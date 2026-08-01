use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    is_less_than_array::{
        IsLtArrayAuxCols, IsLtArrayAuxColsMut, IsLtArrayAuxColsRef, IsLtArrayIo, IsLtArraySubAir,
    },
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
    TraceSubRowGenerator,
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

use crate::fminmax::fp_key;

/// Number of `lt_decomp` limbs in [IsLtArrayAuxCols]: the comparison uses
/// `max_bits = 8` per key limb, so `8.div_ceil(range_max_bits)` limbs are needed.
/// Warning: This requires that `(8 + range_max_bits - 1) / range_max_bits = LT_AUX_LEN`,
/// i.e. the system range checker must have `range_max_bits >= 8`.
pub const LT_AUX_LEN: usize = 1;

/// Maximum bits per key limb for the lexicographic less-than check.
pub const KEY_LIMB_BITS: usize = 8;

/// Which comparison an feq.s / flt.s / fle.s instruction performs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FcmpOp {
    Le,
    Lt,
    Eq,
}

/// IEEE 754 / RISC-V FEQ.S / FLT.S / FLE.S: returns 0 or 1.
/// If either operand is NaN, the result is 0. -0.0 and +0.0 compare equal.
#[inline(always)]
pub fn fcmp_result(a: u32, b: u32, op: FcmpOp) -> u32 {
    let a_f = f32::from_bits(a);
    let b_f = f32::from_bits(b);
    if a_f.is_nan() || b_f.is_nan() {
        return 0;
    }
    match op {
        FcmpOp::Eq => (a_f == b_f) as u32,
        FcmpOp::Lt => (a_f < b_f) as u32,
        FcmpOp::Le => (a_f <= b_f) as u32,
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FcmpCoreCols<T> {
    pub a: [T; 4],
    pub b: [T; 4],

    // === Decomposition of inputs ===
    pub a_sign: T,
    pub b_sign: T,
    pub a_exp_lo: T, // bit 0 of exponent (top bit of byte 2)
    pub b_exp_lo: T,
    pub a_mant2: T, // low 7 bits of byte 2
    pub b_mant2: T,
    pub a_hi: T, // low 7 bits of byte 3 (high 7 bits of exponent)
    pub b_hi: T,

    // === Monotone comparison keys (little-endian limbs) ===
    pub key_a: [T; 4],
    pub key_b: [T; 4],

    // === Classification inverse witnesses and flags ===
    pub a_exp_zero_inv: T,
    pub b_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub b_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub b_mant_sum_inv: T,
    pub a_exp_is_zero: T,
    pub b_exp_is_zero: T,
    pub a_exp_is_max: T,
    pub b_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub b_mant_is_zero: T,
    pub a_is_zero: T,
    pub b_is_zero: T,
    pub both_zero: T, // both operands are zero (any sign): -0.0 == +0.0
    pub a_nan: T,
    pub b_nan: T,
    pub any_nan: T,

    // === Less-than: key_a < key_b and key_b < key_a (big-endian limb order) ===
    pub lt_ab_aux: IsLtArrayAuxCols<T, 4, LT_AUX_LEN>,
    pub lt_ab: T,
    pub lt_ba_aux: IsLtArrayAuxCols<T, 4, LT_AUX_LEN>,
    pub lt_ba: T,
    pub key_eq: T, // 1 - lt_ab - lt_ba, i.e. keys bitwise equal

    // === Selection ===
    pub sel: T, // comparison result before NaN masking
    pub feq_flag: T,
    pub flt_flag: T,
    pub fle_flag: T,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FcmpCoreCols<u8>)]
pub struct FcmpCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FcmpCoreAir {
    fn width(&self) -> usize {
        FcmpCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FcmpCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FcmpCoreAir
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
        let cols: &FcmpCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.b_exp_lo);
        builder.assert_bool(cols.a_exp_is_zero);
        builder.assert_bool(cols.b_exp_is_zero);
        builder.assert_bool(cols.a_exp_is_max);
        builder.assert_bool(cols.b_exp_is_max);
        builder.assert_bool(cols.a_mant_is_zero);
        builder.assert_bool(cols.b_mant_is_zero);
        builder.assert_bool(cols.a_is_zero);
        builder.assert_bool(cols.b_is_zero);
        builder.assert_bool(cols.both_zero);
        builder.assert_bool(cols.a_nan);
        builder.assert_bool(cols.b_nan);
        builder.assert_bool(cols.any_nan);
        builder.assert_bool(cols.key_eq);
        builder.assert_bool(cols.sel);
        builder.assert_bool(cols.feq_flag);
        builder.assert_bool(cols.flt_flag);
        builder.assert_bool(cols.fle_flag);

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.feq_flag + cols.flt_flag + cols.fle_flag);

        // ── Byte decomposition ───────────────────────────────────────────
        // a[2] = a_mant2 + a_exp_lo * 128, a[3] = a_hi + a_sign * 128
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        builder.assert_eq(cols.b[2], cols.b_mant2 + cols.b_exp_lo * f128);
        builder.assert_eq(cols.b[3], cols.b_hi + cols.b_sign * f128);
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // a_exp_lo / a_sign are free field choices.
        for v in [cols.a_mant2, cols.b_mant2, cols.a_hi, cols.b_hi] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

        // ── Monotone comparison keys ─────────────────────────────────────
        // Bytes 0-2: key = sign ? 255 - x : x
        for i in 0..3 {
            builder.assert_eq(
                cols.key_a[i],
                cols.a[i] + cols.a_sign * (AB::Expr::from(f255) - cols.a[i] * f2),
            );
            builder.assert_eq(
                cols.key_b[i],
                cols.b[i] + cols.b_sign * (AB::Expr::from(f255) - cols.b[i] * f2),
            );
        }
        // Byte 3: key = sign ? 127 - hi : hi + 128. Gated by is_valid so that
        // the all-zero padding row (key = 0 vs constant 128) is unconstrained.
        builder.when(cols.is_valid).assert_eq(
            cols.key_a[3],
            cols.a_hi + f128 - cols.a_sign * (cols.a_hi * f2 + one.clone()),
        );
        builder.when(cols.is_valid).assert_eq(
            cols.key_b[3],
            cols.b_hi + f128 - cols.b_sign * (cols.b_hi * f2 + one.clone()),
        );
        // Range check key limbs to 8 bits so the lexicographic comparison is sound.
        for i in 0..4 {
            self.range_bus
                .range_check(cols.key_a[i].into(), KEY_LIMB_BITS)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.key_b[i].into(), KEY_LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }

        // ── Classification ───────────────────────────────────────────────
        // exp = exp_lo + 2 * hi, mant_sum = a[0] + a[1] + a_mant2
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;
        let a_exp_max_delta = AB::Expr::from(f255) - a_exp.clone();
        let b_exp_max_delta = AB::Expr::from(f255) - b_exp.clone();
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_max,
            one.clone() - a_exp_max_delta.clone() * cols.a_exp_max_inv,
        );
        builder.assert_zero(a_exp_max_delta * cols.a_exp_is_max);
        builder.when(cols.is_valid).assert_eq(
            cols.b_exp_is_max,
            one.clone() - b_exp_max_delta.clone() * cols.b_exp_max_inv,
        );
        builder.assert_zero(b_exp_max_delta * cols.b_exp_is_max);
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_zero,
            one.clone() - a_exp.clone() * cols.a_exp_zero_inv,
        );
        builder.assert_zero(a_exp * cols.a_exp_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.b_exp_is_zero,
            one.clone() - b_exp.clone() * cols.b_exp_zero_inv,
        );
        builder.assert_zero(b_exp * cols.b_exp_is_zero);

        let a_mant_sum = cols.a[0] + cols.a[1] + cols.a_mant2;
        let b_mant_sum = cols.b[0] + cols.b[1] + cols.b_mant2;
        builder.when(cols.is_valid).assert_eq(
            cols.a_mant_is_zero,
            one.clone() - a_mant_sum.clone() * cols.a_mant_sum_inv,
        );
        builder.assert_zero(a_mant_sum * cols.a_mant_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.b_mant_is_zero,
            one.clone() - b_mant_sum.clone() * cols.b_mant_sum_inv,
        );
        builder.assert_zero(b_mant_sum * cols.b_mant_is_zero);

        // NaN = (exp == 255) && (mant != 0); zero = (exp == 0) && (mant == 0)
        builder.assert_eq(
            cols.a_nan,
            cols.a_exp_is_max * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(
            cols.b_nan,
            cols.b_exp_is_max * (one.clone() - cols.b_mant_is_zero),
        );
        builder.assert_eq(
            cols.any_nan,
            cols.a_nan + cols.b_nan - cols.a_nan * cols.b_nan,
        );
        builder.assert_eq(cols.a_is_zero, cols.a_exp_is_zero * cols.a_mant_is_zero);
        builder.assert_eq(cols.b_is_zero, cols.b_exp_is_zero * cols.b_mant_is_zero);
        builder.assert_eq(cols.both_zero, cols.a_is_zero * cols.b_is_zero);

        // ── Comparisons: lt_ab = (key_a < key_b), lt_ba = (key_b < key_a) ──
        let lt_air = IsLtArraySubAir::<4>::new(self.range_bus, KEY_LIMB_BITS);
        let key_a_be = [
            cols.key_a[3].into(),
            cols.key_a[2].into(),
            cols.key_a[1].into(),
            cols.key_a[0].into(),
        ];
        let key_b_be = [
            cols.key_b[3].into(),
            cols.key_b[2].into(),
            cols.key_b[1].into(),
            cols.key_b[0].into(),
        ];
        let io_ab = IsLtArrayIo {
            x: key_a_be.clone(),
            y: key_b_be.clone(),
            out: cols.lt_ab.into(),
            count: cols.is_valid.into(),
        };
        let aux_ab = IsLtArrayAuxColsRef {
            diff_marker: &cols.lt_ab_aux.diff_marker,
            diff_inv: &cols.lt_ab_aux.diff_inv,
            lt_decomp: &cols.lt_ab_aux.lt_aux.lower_decomp,
        };
        lt_air.eval(builder, (io_ab, aux_ab));

        let io_ba = IsLtArrayIo {
            x: key_b_be,
            y: key_a_be,
            out: cols.lt_ba.into(),
            count: cols.is_valid.into(),
        };
        let aux_ba = IsLtArrayAuxColsRef {
            diff_marker: &cols.lt_ba_aux.diff_marker,
            diff_inv: &cols.lt_ba_aux.diff_inv,
            lt_decomp: &cols.lt_ba_aux.lt_aux.lower_decomp,
        };
        lt_air.eval(builder, (io_ba, aux_ba));

        // key_eq = !(lt_ab || lt_ba); gated so padding rows are unconstrained
        builder
            .when(cols.is_valid)
            .assert_eq(cols.key_eq, one.clone() - cols.lt_ab - cols.lt_ba);

        // ── Selection ────────────────────────────────────────────────────
        // Key order matches float order except that key(-0.0) < key(+0.0);
        // both_zero masks that case since -0.0 == +0.0 for feq/flt/fle.
        let not_both_zero = one.clone() - cols.both_zero;
        let feq_val = cols.key_eq + cols.both_zero - cols.key_eq * cols.both_zero;
        let flt_val = cols.lt_ab * not_both_zero.clone();
        let fle_val = one.clone() - cols.lt_ba * not_both_zero;
        builder.assert_eq(
            cols.sel,
            cols.feq_flag * feq_val + cols.flt_flag * flt_val + cols.fle_flag * fle_val,
        );

        // Any NaN input makes the comparison false; result is 0 or 1 in byte 0.
        builder.assert_eq(cols.result[0], (one.clone() - cols.any_nan) * cols.sel);
        builder.assert_zero(cols.result[1]);
        builder.assert_zero(cols.result[2]);
        builder.assert_zero(cols.result[3]);

        let expected_opcode = cols.feq_flag * AB::F::from_u32(FpOpcode::FEQ_S as u32)
            + cols.flt_flag * AB::F::from_u32(FpOpcode::FLT_S as u32)
            + cols.fle_flag * AB::F::from_u32(FpOpcode::FLE_S as u32)
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

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FcmpCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FcmpExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

pub(crate) fn opcode_to_fcmp_op(opcode: FpOpcode) -> Option<FcmpOp> {
    match opcode {
        FpOpcode::FLE_S => Some(FcmpOp::Le),
        FpOpcode::FLT_S => Some(FcmpOp::Lt),
        FpOpcode::FEQ_S => Some(FcmpOp::Eq),
        _ => None,
    }
}

impl<A, F, RA> PreflightExecutor<F, RA> for FcmpExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FcmpCoreRecord),
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

        let op = opcode_to_fcmp_op(FpOpcode::from_usize(local)).unwrap();
        let result = fcmp_result(u32::from_le_bytes(a), u32::from_le_bytes(b), op);
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

pub struct FcmpFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub lt_air: IsLtArraySubAir<4>,
}

impl<A> FcmpFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        let lt_air = IsLtArraySubAir::<4>::new(range_checker.bus(), KEY_LIMB_BITS);
        Self {
            adapter,
            offset,
            range_checker,
            lt_air,
        }
    }
}

impl<A, F> TraceFiller<F> for FcmpFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FcmpCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FcmpCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let op = opcode_to_fcmp_op(opcode).unwrap();
        let result = fcmp_result(a_bits, b_bits, op);

        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        core_row.a_sign = F::from_u32(a_sign);
        core_row.b_sign = F::from_u32(b_sign);
        core_row.a_exp_lo = F::from_u32(a_exp & 1);
        core_row.b_exp_lo = F::from_u32(b_exp & 1);
        core_row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        core_row.b_mant2 = F::from_u32((b_mant >> 16) & 0x7F);
        core_row.a_hi = F::from_u32(a_exp >> 1);
        core_row.b_hi = F::from_u32(b_exp >> 1);
        self.range_checker.add_count((a_mant >> 16) & 0x7F, 7);
        self.range_checker.add_count((b_mant >> 16) & 0x7F, 7);
        self.range_checker.add_count(a_exp >> 1, 7);
        self.range_checker.add_count(b_exp >> 1, 7);

        // Comparison keys
        let (ka, kb) = (fp_key(a_bits), fp_key(b_bits));
        let key_a: [F; 4] = core::array::from_fn(|i| F::from_u32((ka >> (i * 8)) & 0xFF));
        let key_b: [F; 4] = core::array::from_fn(|i| F::from_u32((kb >> (i * 8)) & 0xFF));
        for i in 0..4 {
            core_row.key_a[i] = key_a[i];
            core_row.key_b[i] = key_b[i];
            self.range_checker
                .add_count(key_a[i].as_canonical_u32(), KEY_LIMB_BITS);
            self.range_checker
                .add_count(key_b[i].as_canonical_u32(), KEY_LIMB_BITS);
        }

        // Classification witnesses
        core_row.a_exp_is_zero = F::from_bool(a_exp == 0);
        core_row.b_exp_is_zero = F::from_bool(b_exp == 0);
        core_row.a_exp_is_max = F::from_bool(a_exp == 255);
        core_row.b_exp_is_max = F::from_bool(b_exp == 255);
        let (a_exp_f, b_exp_f) = (F::from_u32(a_exp), F::from_u32(b_exp));
        core_row.a_exp_zero_inv = if a_exp != 0 {
            a_exp_f.try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.b_exp_zero_inv = if b_exp != 0 {
            b_exp_f.try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.a_exp_max_inv = if a_exp != 255 {
            (F::from_u32(255) - a_exp_f)
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.b_exp_max_inv = if b_exp != 255 {
            (F::from_u32(255) - b_exp_f)
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
        let b_mant_sum = (b_mant & 0xFF) + ((b_mant >> 8) & 0xFF) + ((b_mant >> 16) & 0x7F);
        core_row.a_mant_is_zero = F::from_bool(a_mant == 0);
        core_row.b_mant_is_zero = F::from_bool(b_mant == 0);
        core_row.a_mant_sum_inv = if a_mant != 0 {
            F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.b_mant_sum_inv = if b_mant != 0 {
            F::from_u32(b_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_is_zero = a_exp == 0 && a_mant == 0;
        let b_is_zero = b_exp == 0 && b_mant == 0;
        let both_zero = a_is_zero && b_is_zero;
        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        core_row.a_is_zero = F::from_bool(a_is_zero);
        core_row.b_is_zero = F::from_bool(b_is_zero);
        core_row.both_zero = F::from_bool(both_zero);
        core_row.a_nan = F::from_bool(a_nan);
        core_row.b_nan = F::from_bool(b_nan);
        core_row.any_nan = F::from_bool(a_nan || b_nan);

        // Less-than auxiliary columns (also updates range checker multiplicities)
        let x_ab = [key_a[3], key_a[2], key_a[1], key_a[0]];
        let y_ab = [key_b[3], key_b[2], key_b[1], key_b[0]];
        let aux_ab = IsLtArrayAuxColsMut {
            diff_marker: &mut core_row.lt_ab_aux.diff_marker,
            diff_inv: &mut core_row.lt_ab_aux.diff_inv,
            lt_decomp: &mut core_row.lt_ab_aux.lt_aux.lower_decomp,
        };
        self.lt_air.generate_subrow(
            (self.range_checker.as_ref(), &x_ab, &y_ab),
            (aux_ab, &mut core_row.lt_ab),
        );
        let aux_ba = IsLtArrayAuxColsMut {
            diff_marker: &mut core_row.lt_ba_aux.diff_marker,
            diff_inv: &mut core_row.lt_ba_aux.diff_inv,
            lt_decomp: &mut core_row.lt_ba_aux.lt_aux.lower_decomp,
        };
        self.lt_air.generate_subrow(
            (self.range_checker.as_ref(), &y_ab, &x_ab),
            (aux_ba, &mut core_row.lt_ba),
        );

        let lt_ab = ka < kb;
        let lt_ba = kb < ka;
        core_row.key_eq = F::from_bool(!lt_ab && !lt_ba);
        let sel = match op {
            FcmpOp::Eq => (!lt_ab && !lt_ba) || both_zero,
            FcmpOp::Lt => lt_ab && !both_zero,
            FcmpOp::Le => !lt_ba || both_zero,
        };
        core_row.sel = F::from_bool(sel);
        core_row.feq_flag = F::from_bool(op == FcmpOp::Eq);
        core_row.flt_flag = F::from_bool(op == FcmpOp::Lt);
        core_row.fle_flag = F::from_bool(op == FcmpOp::Le);
        core_row.is_valid = F::ONE;
    }
}

// === Witness computation helpers ===

fn decompose(bits: u32) -> (u32, u32, u32) {
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    (sign, exp, mant)
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

    fn build_row(a_bits: u32, b_bits: u32, op: FcmpOp) -> Vec<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt_air = IsLtArraySubAir::<4>::new(chip.bus(), KEY_LIMB_BITS);
        let mut buf = vec![F::ZERO; FcmpCoreCols::<F>::width()];
        let row: &mut FcmpCoreCols<F> = buf.as_mut_slice().borrow_mut();
        let result = fcmp_result(a_bits, b_bits, op);
        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);

        for i in 0..4 {
            row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        row.a_sign = F::from_u32(a_sign);
        row.b_sign = F::from_u32(b_sign);
        row.a_exp_lo = F::from_u32(a_exp & 1);
        row.b_exp_lo = F::from_u32(b_exp & 1);
        row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        row.b_mant2 = F::from_u32((b_mant >> 16) & 0x7F);
        row.a_hi = F::from_u32(a_exp >> 1);
        row.b_hi = F::from_u32(b_exp >> 1);

        let (ka, kb) = (fp_key(a_bits), fp_key(b_bits));
        let key_a: [F; 4] = core::array::from_fn(|i| F::from_u32((ka >> (i * 8)) & 0xFF));
        let key_b: [F; 4] = core::array::from_fn(|i| F::from_u32((kb >> (i * 8)) & 0xFF));
        row.key_a = key_a;
        row.key_b = key_b;

        row.a_exp_is_zero = F::from_bool(a_exp == 0);
        row.b_exp_is_zero = F::from_bool(b_exp == 0);
        row.a_exp_is_max = F::from_bool(a_exp == 255);
        row.b_exp_is_max = F::from_bool(b_exp == 255);
        row.a_exp_zero_inv = if a_exp != 0 {
            F::from_u32(a_exp).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.b_exp_zero_inv = if b_exp != 0 {
            F::from_u32(b_exp).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.a_exp_max_inv = if a_exp != 255 {
            (F::from_u32(255) - F::from_u32(a_exp))
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.b_exp_max_inv = if b_exp != 255 {
            (F::from_u32(255) - F::from_u32(b_exp))
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
        let b_mant_sum = (b_mant & 0xFF) + ((b_mant >> 8) & 0xFF) + ((b_mant >> 16) & 0x7F);
        row.a_mant_is_zero = F::from_bool(a_mant == 0);
        row.b_mant_is_zero = F::from_bool(b_mant == 0);
        row.a_mant_sum_inv = if a_mant != 0 {
            F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.b_mant_sum_inv = if b_mant != 0 {
            F::from_u32(b_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_is_zero = a_exp == 0 && a_mant == 0;
        let b_is_zero = b_exp == 0 && b_mant == 0;
        row.a_is_zero = F::from_bool(a_is_zero);
        row.b_is_zero = F::from_bool(b_is_zero);
        row.both_zero = F::from_bool(a_is_zero && b_is_zero);
        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        row.a_nan = F::from_bool(a_nan);
        row.b_nan = F::from_bool(b_nan);
        row.any_nan = F::from_bool(a_nan || b_nan);

        let x_ab = [key_a[3], key_a[2], key_a[1], key_a[0]];
        let y_ab = [key_b[3], key_b[2], key_b[1], key_b[0]];
        let aux_ab = IsLtArrayAuxColsMut {
            diff_marker: &mut row.lt_ab_aux.diff_marker,
            diff_inv: &mut row.lt_ab_aux.diff_inv,
            lt_decomp: &mut row.lt_ab_aux.lt_aux.lower_decomp,
        };
        lt_air.generate_subrow((chip.as_ref(), &x_ab, &y_ab), (aux_ab, &mut row.lt_ab));
        let aux_ba = IsLtArrayAuxColsMut {
            diff_marker: &mut row.lt_ba_aux.diff_marker,
            diff_inv: &mut row.lt_ba_aux.diff_inv,
            lt_decomp: &mut row.lt_ba_aux.lt_aux.lower_decomp,
        };
        lt_air.generate_subrow((chip.as_ref(), &y_ab, &x_ab), (aux_ba, &mut row.lt_ba));

        let lt_ab = ka < kb;
        let lt_ba = kb < ka;
        row.key_eq = F::from_bool(!lt_ab && !lt_ba);
        let both_zero = a_is_zero && b_is_zero;
        let sel = match op {
            FcmpOp::Eq => (!lt_ab && !lt_ba) || both_zero,
            FcmpOp::Lt => lt_ab && !both_zero,
            FcmpOp::Le => !lt_ba || both_zero,
        };
        row.sel = F::from_bool(sel);
        row.feq_flag = F::from_bool(op == FcmpOp::Eq);
        row.flt_flag = F::from_bool(op == FcmpOp::Lt);
        row.fle_flag = F::from_bool(op == FcmpOp::Le);
        row.is_valid = F::ONE;
        buf
    }

    fn eval_row(buf: &[F]) -> NumericBuilder<F> {
        let air = FcmpCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut b = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
        b
    }

    #[test]
    fn honest_rows_pass() {
        for (a, b) in [
            (1.0f32.to_bits(), 2.0f32.to_bits()),
            (0x7FC00000, 0x3F800000), // qNaN vs 1.0
            (0x80000000, 0x00000000), // -0 vs +0
            (0x7F800000, 0xFF800000), // inf vs -inf
            (0x00400000, 0x3F800000), // subnormal vs 1.0
        ] {
            for op in [FcmpOp::Eq, FcmpOp::Lt, FcmpOp::Le] {
                let row = build_row(a, b, op);
                assert!(
                    eval_row(&row).ok(),
                    "fcmp {op:?} honest failed: a={a:08x} b={b:08x}"
                );
            }
        }
    }

    // NOTE: decomposition-flip attacks (exp_lo/sign with wrapped lo bytes)
    // are caught only by the 7-bit range-check *bus sends*, which
    // NumericBuilder ignores. They must be covered by prover-level
    // bite-checks (hardening item 2), not numeric tests.

    /// Every column must be constrained: perturbing any single column by +1
    /// must break at least one gate constraint (except columns whose only
    /// constraint is a range-check bus send — those are whitelisted here).
    #[test]
    fn corruption_sweep_finds_no_unconstrained_witness() {
        use crate::test_utils::corruption_sweep;
        use openvm_circuit_primitives::StructReflectionHelper;

        let names = FcmpCoreCols::<F>::struct_reflection().unwrap_or_default();
        // Columns whose only constraint is a range-check bus send: NumericBuilder
        // ignores interactions, so a +1 perturbation is accepted even though the
        // real prover rejects it. Verified constrained by the bus (see eval).
        // The `*_inv` inverse witnesses follow the uniform pattern
        // `flag = 1 - delta * inv` + `assert_zero(delta * flag)`: when `delta == 0`
        // the flag is pinned to 1 by the constraint and `inv` is multiplied by 0
        // (hence free); when `delta != 0` the equation pins `inv = (1 - flag)/delta`.
        // The sweep matrix covers delta == 0 (exp 0/255, zero mantissa) and delta != 0
        // (normal exp, non-zero mantissa) classes, confirming the pin on the latter.
        let range_bus_only: std::collections::HashSet<usize> = {
            let mut s = std::collections::HashSet::new();
            for name in [
                "a_mant2",
                "b_mant2",
                "a_hi",
                "b_hi",
                "key_a",
                "key_b",
                "lt_ab_aux",
                "lt_ba_aux",
                "a_exp_zero_inv",
                "b_exp_zero_inv",
                "a_exp_max_inv",
                "b_exp_max_inv",
                "a_mant_sum_inv",
                "b_mant_sum_inv",
            ] {
                for (i, n) in names.iter().enumerate() {
                    if n == name || n.starts_with(&format!("{name}__")) {
                        s.insert(i);
                    }
                }
            }
            s
        };

        let classes: &[(u32, u32, FcmpOp)] = &[
            (1.0f32.to_bits(), 2.0f32.to_bits(), FcmpOp::Lt),
            (1.5f32.to_bits(), 2.5f32.to_bits(), FcmpOp::Eq),
            (0x7FC00000, 0x3F800000, FcmpOp::Eq), // qNaN vs 1.0
            (0x80000000, 0x00000000, FcmpOp::Le), // -0 vs +0
            (0x7F800000, 0xFF800000, FcmpOp::Eq), // inf vs -inf
            (0x00400000, 0x3F800000, FcmpOp::Lt), // subnormal vs 1.0
        ];

        for (a, b, op) in classes {
            let row = build_row(*a, *b, *op);
            let accepted = corruption_sweep(&row, |builder, cols| {
                VmCoreAir::<NumericBuilder<F>, TestI>::eval(
                    &FcmpCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET),
                    builder,
                    cols,
                    F::ZERO,
                );
            });
            let unexpected: Vec<usize> = accepted
                .into_iter()
                .filter(|i| !range_bus_only.contains(i))
                .collect();
            assert!(
                unexpected.is_empty(),
                "fcmp a={a:08x} b={b:08x} op={op:?}: unconstrained columns {:?}",
                unexpected
                    .iter()
                    .map(|&i| (i, names.get(i).cloned().unwrap_or_default()))
                    .collect::<Vec<_>>()
            );
        }
    }
}
