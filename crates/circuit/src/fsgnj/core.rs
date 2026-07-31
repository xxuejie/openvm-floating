use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
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

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FsgnjCoreCols<T> {
    pub a: [T; 4],
    pub b: [T; 4],
    pub a_sign: T,
    pub b_sign: T,
    pub a_byte3_lo: T,
    pub b_byte3_lo: T,
    pub fsgnj_flag: T,
    pub fsgnjn_flag: T,
    pub fsgnjx_flag: T,
    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FsgnjCoreCols<u8>)]
pub struct FsgnjCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FsgnjCoreAir {
    fn width(&self) -> usize {
        FsgnjCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FsgnjCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FsgnjCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 4]; 2]>,
    I::Writes: From<[[AB::Expr; 4]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FsgnjCoreCols<AB::Var> = local_core.borrow();
        let f128: AB::F = AB::F::from_u32(128);

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.fsgnj_flag);
        builder.assert_bool(cols.fsgnjn_flag);
        builder.assert_bool(cols.fsgnjx_flag);

        // Sign extraction: a[3] = a_byte3_lo + a_sign * 128
        builder.assert_eq(cols.a[3], cols.a_byte3_lo + cols.a_sign * f128);
        builder.assert_eq(cols.b[3], cols.b_byte3_lo + cols.b_sign * f128);
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // a_sign / b_sign are free field choices.
        for v in [cols.a_byte3_lo, cols.b_byte3_lo] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

        // Exactly one opcode flag when valid
        let flag_sum = cols.fsgnj_flag + cols.fsgnjn_flag + cols.fsgnjx_flag;
        builder.when(cols.is_valid).assert_one(flag_sum);

        // Result bytes 0-2 are unchanged from a
        builder.assert_eq(cols.result[0], cols.a[0]);
        builder.assert_eq(cols.result[1], cols.a[1]);
        builder.assert_eq(cols.result[2], cols.a[2]);

        // Compute result sign based on opcode
        // fsgnj:  result_sign = b_sign
        // fsgnjn: result_sign = 1 - b_sign
        // fsgnjx: result_sign = a_sign XOR b_sign = a_sign + b_sign - 2*a_sign*b_sign
        let one = AB::Expr::ONE;
        let xor_sign = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * AB::F::from_u32(2);
        let result_sign = cols.fsgnj_flag * cols.b_sign
            + cols.fsgnjn_flag * (one.clone() - cols.b_sign)
            + cols.fsgnjx_flag * xor_sign;

        builder.assert_eq(cols.result[3], cols.a_byte3_lo + result_sign * f128);

        let expected_opcode = cols.fsgnj_flag * AB::F::from_u32(FpOpcode::FSGNJ_S as u32)
            + cols.fsgnjn_flag * AB::F::from_u32(FpOpcode::FSGNJN_S as u32)
            + cols.fsgnjx_flag * AB::F::from_u32(FpOpcode::FSGNJX_S as u32)
            + AB::F::from_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
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
pub struct FsgnjCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
    pub opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct FsgnjExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FsgnjExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FsgnjCoreRecord),
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
        let local = instruction.opcode.local_opcode_idx(self.offset);
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.b = b;
        core_record.opcode = local as u8;

        let result = match FpOpcode::from_usize(local) {
            FpOpcode::FSGNJ_S => {
                (u32::from_le_bytes(a) & 0x7FFFFFFF) | (u32::from_le_bytes(b) & 0x80000000)
            }
            FpOpcode::FSGNJN_S => {
                (u32::from_le_bytes(a) & 0x7FFFFFFF) | (!u32::from_le_bytes(b) & 0x80000000)
            }
            FpOpcode::FSGNJX_S => u32::from_le_bytes(a) ^ (u32::from_le_bytes(b) & 0x80000000),
            _ => unreachable!(),
        };
        self.adapter.write(
            state.memory,
            instruction,
            [result.to_le_bytes()].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

pub struct FsgnjFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FsgnjFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FsgnjFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FsgnjCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FsgnjCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let a_sign = (a_bits >> 31) & 1;
        let b_sign = (b_bits >> 31) & 1;

        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let result_bits = match opcode {
            FpOpcode::FSGNJ_S => (a_bits & 0x7FFFFFFF) | (b_bits & 0x80000000),
            FpOpcode::FSGNJN_S => (a_bits & 0x7FFFFFFF) | (!b_bits & 0x80000000),
            FpOpcode::FSGNJX_S => a_bits ^ (b_bits & 0x80000000),
            _ => 0,
        };

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result_bits >> (i * 8)) & 0xFF);
        }
        core_row.a_sign = F::from_u32(a_sign);
        core_row.b_sign = F::from_u32(b_sign);
        core_row.a_byte3_lo = F::from_u32((a_bits >> 24) & 0x7F);
        core_row.b_byte3_lo = F::from_u32((b_bits >> 24) & 0x7F);
        self.range_checker.add_count((a_bits >> 24) & 0x7F, 7);
        self.range_checker.add_count((b_bits >> 24) & 0x7F, 7);
        core_row.fsgnj_flag = F::from_bool(opcode == FpOpcode::FSGNJ_S);
        core_row.fsgnjn_flag = F::from_bool(opcode == FpOpcode::FSGNJN_S);
        core_row.fsgnjx_flag = F::from_bool(opcode == FpOpcode::FSGNJX_S);
        core_row.is_valid = F::ONE;
    }
}

#[cfg(test)]
mod air_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;
    type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;

    fn build_row(a_bits: u32, b_bits: u32, opcode: FpOpcode) -> Vec<F> {
        let _chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let mut buf = vec![F::ZERO; FsgnjCoreCols::<F>::width()];
        let row: &mut FsgnjCoreCols<F> = buf.as_mut_slice().borrow_mut();
        let a_sign = (a_bits >> 31) & 1;
        let b_sign = (b_bits >> 31) & 1;
        let result_bits = match opcode {
            FpOpcode::FSGNJ_S => (a_bits & 0x7FFFFFFF) | (b_bits & 0x80000000),
            FpOpcode::FSGNJN_S => (a_bits & 0x7FFFFFFF) | (!b_bits & 0x80000000),
            FpOpcode::FSGNJX_S => a_bits ^ (b_bits & 0x80000000),
            _ => unreachable!(),
        };
        for i in 0..4 {
            row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result_bits >> (i * 8)) & 0xFF);
        }
        row.a_sign = F::from_u32(a_sign);
        row.b_sign = F::from_u32(b_sign);
        row.a_byte3_lo = F::from_u32((a_bits >> 24) & 0x7F);
        row.b_byte3_lo = F::from_u32((b_bits >> 24) & 0x7F);
        row.fsgnj_flag = F::from_bool(opcode == FpOpcode::FSGNJ_S);
        row.fsgnjn_flag = F::from_bool(opcode == FpOpcode::FSGNJN_S);
        row.fsgnjx_flag = F::from_bool(opcode == FpOpcode::FSGNJX_S);
        row.is_valid = F::ONE;
        buf
    }

    fn eval_row(buf: &[F]) -> NumericBuilder<F> {
        let air = FsgnjCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut b = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
        b
    }

    #[test]
    fn honest_rows_pass() {
        for op in [FpOpcode::FSGNJ_S, FpOpcode::FSGNJN_S, FpOpcode::FSGNJX_S] {
            let buf = build_row(1.0f32.to_bits(), (-2.0f32).to_bits(), op);
            assert!(eval_row(&buf).ok(), "fsgnj {op:?} honest failed");
        }
    }

    // NOTE: decomposition-flip attacks (sign/exp_lo with wrapped lo bytes)
    // are caught only by the 7-bit range-check *bus sends*, which
    // NumericBuilder ignores. They must be covered by prover-level
    // bite-checks (hardening item 2), not numeric tests.
}
