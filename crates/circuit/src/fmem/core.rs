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

use crate::adapters::FpLoadStoreInstruction;

const NUM_LIMBS: usize = 4;

/// FLW / FSW core: the adapter moves 4 bytes between AS 2 and the FP register
/// file; this core only pins `is_load` to the opcode and passes the value
/// through.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FmemCoreCols<T> {
    pub data: [T; NUM_LIMBS],
    pub flw_flag: T,
    pub fsw_flag: T,
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FmemCoreCols<u8>)]
pub struct FmemCoreAir {
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FmemCoreAir {
    fn width(&self) -> usize {
        FmemCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FmemCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FmemCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<FpLoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FmemCoreCols<AB::Var> = local_core.borrow();
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.flw_flag);
        builder.assert_bool(cols.fsw_flag);
        builder
            .when(cols.is_valid)
            .assert_one(cols.flw_flag + cols.fsw_flag);

        // The adapter moves the value between AS 2 and the FP file: reads[0]
        // (the value) and writes[0] are both `data`.
        let expected_opcode = cols.flw_flag * AB::F::from_u32(FpOpcode::FLW as u32)
            + cols.fsw_flag * AB::F::from_u32(FpOpcode::FSW as u32)
            + AB::F::from_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.data.map(Into::into)].into(),
            writes: [cols.data.map(Into::into)].into(),
            instruction: FpLoadStoreInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                is_load: cols.flw_flag.into(),
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
pub struct FmemCoreRecord {
    pub data: [u8; NUM_LIMBS],
    pub opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct FmemExecutor<A> {
    adapter: A,
    pub offset: usize,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FmemExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FmemCoreRecord),
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

        // The adapter already performs the AS 2 / FP register access; the core
        // just records the moved value and the opcode.
        let [value] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.data = value;
        core_record.opcode = instruction.opcode.local_opcode_idx(self.offset) as u8;
        self.adapter.write(
            state.memory,
            instruction,
            [value].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

pub struct FmemFiller<A> {
    adapter: A,
    pub offset: usize,
}

impl<A> FmemFiller<A> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self { adapter, offset }
    }
}

impl<A, F> TraceFiller<F> for FmemFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FmemCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        // NOTE: `record` aliases the start of the core region (5 bytes vs the
        // 4 field elements of `cols.data`), so copy the fields to locals before
        // any `core_row` writes clobber them.
        let data = record.data;
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let core_row: &mut FmemCoreCols<F> = core_row.borrow_mut();
        for (i, b) in data.into_iter().enumerate() {
            core_row.data[i] = F::from_u8(b);
        }
        match opcode {
            FpOpcode::FLW => {
                core_row.flw_flag = F::ONE;
                core_row.fsw_flag = F::ZERO;
            }
            FpOpcode::FSW => {
                core_row.flw_flag = F::ZERO;
                core_row.fsw_flag = F::ONE;
            }
            other => panic!("fmem filler: unexpected opcode {other:?}"),
        }
        core_row.is_valid = F::ONE;
    }
}
