use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

const NUM_LIMBS: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FxferCoreCols<T> {
    pub data: [T; NUM_LIMBS],
    pub result: [T; NUM_LIMBS],
    pub fmv_x_flag: T,
    pub fmv_s_flag: T,
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FxferCoreCols<u8>)]
pub struct FxferCoreAir {
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FxferCoreAir {
    fn width(&self) -> usize {
        FxferCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FxferCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FxferCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FxferCoreCols<AB::Var> = local_core.borrow();
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.fmv_x_flag);
        builder.assert_bool(cols.fmv_s_flag);

        builder
            .when(cols.is_valid)
            .assert_one(cols.fmv_x_flag + cols.fmv_s_flag);

        for i in 0..NUM_LIMBS {
            builder.assert_eq(cols.result[i], cols.data[i]);
        }

        let expected_opcode = cols.fmv_x_flag * AB::F::from_u32(FpOpcode::FMV_X_S as u32)
            + cols.fmv_s_flag * AB::F::from_u32(FpOpcode::FMV_S_X as u32)
            + AB::F::from_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.data.map(Into::into), [AB::Expr::ZERO; NUM_LIMBS]].into(),
            writes: [cols.result.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FxferCoreRecord {
    pub data: [u8; NUM_LIMBS],
    pub opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct FxferExecutor<A> {
    adapter: A,
    pub offset: usize,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FxferExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FxferCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", FpOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [src, _] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.data = src;
        core_record.opcode = instruction.opcode.local_opcode_idx(self.offset) as u8;
        self.adapter
            .write(state.memory, instruction, [src].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

pub struct FxferFiller<A> {
    adapter: A,
    pub offset: usize,
}

impl<A> FxferFiller<A> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self { adapter, offset }
    }
}

impl<A, F> TraceFiller<F> for FxferFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FxferCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        // NOTE: `record` aliases the start of the core region (it is only
        // 5 bytes while `core_row.data` spans 4 field elements), so the fields
        // must be copied to locals *before* any `core_row` writes, which would
        // otherwise clobber them.
        let data = record.data;
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let core_row: &mut FxferCoreCols<F> = core_row.borrow_mut();
        for (i, b) in data.into_iter().enumerate() {
            core_row.data[i] = F::from_u8(b);
            core_row.result[i] = F::from_u8(b);
        }
        match opcode {
            FpOpcode::FMV_X_S => {
                core_row.fmv_x_flag = F::ONE;
                core_row.fmv_s_flag = F::ZERO;
            }
            FpOpcode::FMV_S_X => {
                core_row.fmv_x_flag = F::ZERO;
                core_row.fmv_s_flag = F::ONE;
            }
            other => panic!("fxfer filler: unexpected opcode {other:?}"),
        }
        core_row.is_valid = F::ONE;
    }
}
