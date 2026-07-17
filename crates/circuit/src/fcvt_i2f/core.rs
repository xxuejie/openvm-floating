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

/// IEEE 754 / RISC-V FCVT.S.W: convert i32 to f32 with RNE rounding.
#[inline(always)]
pub fn fcvt_s_w(x: u32) -> u32 {
    ((x as i32) as f32).to_bits()
}

/// IEEE 754 / RISC-V FCVT.S.WU: convert u32 to f32 with RNE rounding.
#[inline(always)]
pub fn fcvt_s_wu(x: u32) -> u32 {
    (x as f32).to_bits()
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FcvtI2FCoreCols<T> {
    pub a: [T; 4], // integer input bytes

    // === Input decomposition ===
    pub a_sign: T, // bit 31 (signed interpretation)
    pub a_hi: T,   // low 7 bits of byte 3
    pub a_sum_inv: T,
    pub a_is_zero: T,

    // === Result float decomposition ===
    pub r_sign: T,
    pub r_exp_lo: T,
    pub r_mant2: T,
    pub r_hi: T,
    pub r_exp_zero_inv: T,
    pub r_exp_max_inv: T,
    pub r_mant_sum_inv: T,
    pub r_exp_is_zero: T,
    pub r_mant_is_zero: T,
    pub r_is_zero: T,

    pub ws_flag: T,
    pub wu_flag: T,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FcvtI2FCoreCols<u8>)]
pub struct FcvtI2FCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FcvtI2FCoreAir {
    fn width(&self) -> usize {
        FcvtI2FCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FcvtI2FCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FcvtI2FCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 4]; 1]>,
    I::Writes: From<[[AB::Expr; 4]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FcvtI2FCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_is_zero);
        builder.assert_bool(cols.r_sign);
        builder.assert_bool(cols.r_exp_lo);
        builder.assert_bool(cols.r_exp_is_zero);
        builder.assert_bool(cols.r_mant_is_zero);
        builder.assert_bool(cols.r_is_zero);
        builder.assert_bool(cols.ws_flag);
        builder.assert_bool(cols.wu_flag);

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.ws_flag + cols.wu_flag);

        // ── Input decomposition ──────────────────────────────────────────
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        let a_sum = cols.a[0] + cols.a[1] + cols.a[2] + cols.a[3];
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_is_zero, one.clone() - a_sum.clone() * cols.a_sum_inv);
        builder.assert_zero(a_sum * cols.a_is_zero);

        // ── Result decomposition ─────────────────────────────────────────
        builder.assert_eq(cols.result[2], cols.r_mant2 + cols.r_exp_lo * f128);
        builder.assert_eq(cols.result[3], cols.r_hi + cols.r_sign * f128);
        let r_exp = cols.r_exp_lo + cols.r_hi * f2;
        let r_mant_sum = cols.result[0] + cols.result[1] + cols.r_mant2;

        // The result of int -> float conversion is never NaN or infinite:
        // prove r_exp != 255 via an inverse.
        builder
            .when(cols.is_valid)
            .assert_one((AB::Expr::from(f255) - r_exp.clone()) * cols.r_exp_max_inv);

        // Zero input iff zero result (+0.0)
        builder.when(cols.is_valid).assert_eq(
            cols.r_exp_is_zero,
            one.clone() - r_exp.clone() * cols.r_exp_zero_inv,
        );
        builder.assert_zero(r_exp * cols.r_exp_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.r_mant_is_zero,
            one.clone() - r_mant_sum.clone() * cols.r_mant_sum_inv,
        );
        builder.assert_zero(r_mant_sum * cols.r_mant_is_zero);
        builder.assert_eq(cols.r_is_zero, cols.r_exp_is_zero * cols.r_mant_is_zero);
        builder.assert_eq(cols.r_is_zero, cols.a_is_zero);

        // Result sign: 0 for .wu and for zero input; a_sign for .w otherwise.
        builder.assert_eq(
            cols.r_sign,
            (one.clone() - cols.wu_flag) * cols.a_sign * (one.clone() - cols.a_is_zero),
        );

        // TODO: verify the RNE rounding relation between the integer input
        // and the result's exponent/mantissa (needs shift/round machinery
        // similar to fadd normalization).

        // Range check input and result bytes.
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.result[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        let expected_opcode = cols.ws_flag * AB::F::from_u32(FpOpcode::FCVT_S_W as u32)
            + cols.wu_flag * AB::F::from_u32(FpOpcode::FCVT_S_WU as u32)
            + AB::F::from_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.a.map(Into::into)].into(),
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

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FcvtI2FCoreRecord {
    pub a: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FcvtI2FExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub read_as: u32,
    pub write_as: u32,
}

pub(crate) fn opcode_is_wu(opcode: FpOpcode) -> Option<bool> {
    match opcode {
        FpOpcode::FCVT_S_W => Some(false),
        FpOpcode::FCVT_S_WU => Some(true),
        _ => None,
    }
}

/// Result of fcvt.s.w (is_wu = false) or fcvt.s.wu (is_wu = true).
#[inline(always)]
pub fn fcvt_i2f_result(x: u32, is_wu: bool) -> u32 {
    if is_wu {
        fcvt_s_wu(x)
    } else {
        fcvt_s_w(x)
    }
}

impl<A, F, RA> PreflightExecutor<F, RA> for FcvtI2FExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FcvtI2FCoreRecord),
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

        let [a] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.opcode = local as u8;

        let is_wu = opcode_is_wu(FpOpcode::from_usize(local)).unwrap();
        let result = fcvt_i2f_result(u32::from_le_bytes(a), is_wu);
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

// === Filler ===

pub struct FcvtI2FFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FcvtI2FFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FcvtI2FFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FcvtI2FCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FcvtI2FCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let is_wu = opcode_is_wu(opcode).unwrap();
        let result = fcvt_i2f_result(a_bits, is_wu);

        let r_sign = (result >> 31) & 1;
        let r_exp = (result >> 23) & 0xFF;
        let r_mant = result & 0x7FFFFF;

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((result >> (i * 8)) & 0xFF, 8);
        }
        core_row.a_sign = F::from_u32((a_bits >> 31) & 1);
        core_row.a_hi = F::from_u32(((a_bits >> 24) & 0xFF) & 0x7F);
        let a_sum: u32 = (a_bits & 0xFF)
            + ((a_bits >> 8) & 0xFF)
            + ((a_bits >> 16) & 0xFF)
            + ((a_bits >> 24) & 0xFF);
        core_row.a_is_zero = F::from_bool(a_bits == 0);
        core_row.a_sum_inv = if a_bits != 0 {
            F::from_u32(a_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };

        core_row.r_sign = F::from_u32(r_sign);
        core_row.r_exp_lo = F::from_u32(r_exp & 1);
        core_row.r_mant2 = F::from_u32((r_mant >> 16) & 0x7F);
        core_row.r_hi = F::from_u32(r_exp >> 1);

        let r_exp_f = F::from_u32(r_exp);
        core_row.r_exp_is_zero = F::from_bool(r_exp == 0);
        core_row.r_exp_zero_inv = if r_exp != 0 {
            r_exp_f.try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        // result is never NaN/inf: r_exp != 255 always holds
        core_row.r_exp_max_inv = (F::from_u32(255) - r_exp_f)
            .try_inverse()
            .unwrap_or(F::ZERO);
        let r_mant_sum = (r_mant & 0xFF) + ((r_mant >> 8) & 0xFF) + ((r_mant >> 16) & 0x7F);
        core_row.r_mant_is_zero = F::from_bool(r_mant == 0);
        core_row.r_mant_sum_inv = if r_mant != 0 {
            F::from_u32(r_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.r_is_zero = F::from_bool(r_exp == 0 && r_mant == 0);

        core_row.ws_flag = F::from_bool(!is_wu);
        core_row.wu_flag = F::from_bool(is_wu);
        core_row.is_valid = F::ONE;
    }
}
