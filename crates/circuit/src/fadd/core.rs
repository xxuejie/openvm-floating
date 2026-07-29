use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_guest::CANONICAL_NAN;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::fp_shift::{
    run_shln, run_shrn, run_shrn_round, ShlnCols, ShlnGadget, ShrnCols, ShrnGadget, SHLIMB_BITS,
};

pub const SIG_NUM_LIMBS: usize = 3;
pub const EXT_SIG_LIMBS: usize = 4; // extended significand: 3 mantissa bytes + 1 guard/sticky byte
pub const LIMB_BITS: usize = 8;
/// Working significand width in bytes: 3 significand bytes + 1 guard/sticky byte.
pub const WORK_LIMBS: usize = 4;
/// ediff >= 25 ⇒ the smaller operand is strictly below 0.5 ULP of the larger, so the
/// result equals the larger operand (no rounding). Below this, align exactly.
pub const DEEP_EDIFF: u32 = 25;

/// IEEE 754 / RISC-V FADD.S: correctly-rounded (RNE) addition.
/// Returns the canonical NaN whenever the result is any NaN (NaN inputs, or
/// inf + (-inf)) — the result is deterministic across platforms.
#[inline(always)]
pub fn fadd_s(a: u32, b: u32) -> u32 {
    let r = f32::from_bits(a) + f32::from_bits(b);
    if r.is_nan() {
        CANONICAL_NAN
    } else {
        r.to_bits()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FaddCoreCols<T> {
    // === Decomposition of input a ===
    pub a_sign: T,
    pub a_exp_lo: T,
    pub a_exp_hi: T,
    pub a_mant0: T,
    pub a_mant1: T,
    pub a_mant2: T,

    // === Decomposition of input b ===
    pub b_sign: T,
    pub b_exp_lo: T,
    pub b_exp_hi: T,
    pub b_mant0: T,
    pub b_mant1: T,
    pub b_mant2: T,

    // === Classification inverse witnesses ===
    pub a_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub b_exp_zero_inv: T,
    pub b_exp_max_inv: T,
    pub b_mant_sum_inv: T,

    // === Classification flags (columns; see eval) ===
    pub a_exp_is_zero: T,
    pub a_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub b_exp_is_zero: T,
    pub b_exp_is_max: T,
    pub b_mant_is_zero: T,
    pub a_is_nan: T,
    pub a_is_inf: T,
    pub a_is_zero: T,
    pub b_is_nan: T,
    pub b_is_inf: T,
    pub b_is_zero: T,

    // === Special-case selectors (input-derived; see eval_special_cases) ===
    pub inf_in: T,      // a_is_inf | b_is_inf
    pub s_nan: T,       // canonical NaN (NaN input, or inf + (-inf))
    pub s_inf: T,       // +-inf due to an infinite input
    pub s_zero_both: T, // +-0 from 0 + 0

    // === Alignment ===
    pub a_is_larger: T, // bool: a_exp >= b_exp
    pub ediff: T,       // |a_exp - b_exp|, in [0, 255]

    // Both-normal indicator columns (degree-1 witnesses, used by selectors).
    pub a_is_normal: T,
    pub b_is_normal: T,
    // Selected significands (3 bytes each). large = the operand with the larger exp.
    pub large_sig: [T; SIG_NUM_LIMBS],
    pub small_sig: [T; SIG_NUM_LIMBS],

    // Variable right-shift gadget for alignment of the smaller significand. Its `x`
    // input is `small_sig << 8` (4 bytes), `shift` = ediff, `y` = aligned significand.
    pub align: ShrnCols<T, WORK_LIMBS>,
    pub is_deep: T, // bool: ediff >= 25 (smaller operand < 0.5 ULP ⇒ result = larger)
    pub deep_delta: T, // byte witness used to soundly derive is_deep
    pub lost_sticky: T, // OR of align.guard / align.sticky (bits lost during alignment)
    pub align_enable: T, // bool: s_add AND !is_deep (gates the ShrnGadget)

    // === Sum / difference ===
    pub is_add: T,                // bool: same sign → add, different sign → subtract
    pub raw_sum: [T; WORK_LIMBS], // significand sum (before normalization)
    pub sum_c1: T,                // carry out of raw_sum byte 1
    pub sum_c2: T,                // carry out of raw_sum byte 2
    pub carry_out: T,             // carry out of raw_sum byte 3 (= bit 32 of the sum)

    // === Normalization ===
    pub norm_sig: [T; WORK_LIMBS], // normalized significand
    pub rs_high7: [T; WORK_LIMBS], // raw_sum[i] = rs_high7[i]*2 + rs_lowbit[i]
    pub rs_lowbit: [T; WORK_LIMBS],
    pub result_exp: T, // final exponent after normalization

    // === Rounding ===
    pub guard: T,           // bool: guard bit (bit 7 of norm_sig[0])
    pub ns0_lo: T,          // norm_sig[0] & 0x7F
    pub ns1_high: T,        // (norm_sig[1] - mant_lsb) / 2
    pub mant_lsb: T,        // bit 0 of norm_sig[1]
    pub below_guard: T,     // ns0_lo + dropped + lost_sticky (nonzero ⇒ sticky)
    pub below_guard_inv: T, // inverse for the sticky nonzero check
    pub sticky: T,          // bool: RNE sticky
    pub round_up: T,        // bool: RNE round-up decision

    // === Output reconstruction ===
    pub rc0: T,      // rounding carry out of mantissa byte 0
    pub rc1: T,      // rounding carry out of mantissa byte 1
    pub rc2: T,      // rounding carry out of mantissa top 7 bits (into exponent)
    pub rc2_rem: T,  // (norm_sig[3]-128 + rc1) mod 128
    pub r_exp_lo: T, // bit 0 of result exponent field
    pub r_exp_hi: T, // bits [7:1] of result exponent field
    pub r_exp_hi_zero_inv: T,
    pub r_exp_hi_max_inv: T,
    pub s_add: T,    // finite AND is_add AND both-normal (sound selector for this path)
    pub s_normal: T, // s_add AND !is_deep AND result exp field in [1, 254]
    pub s_overflow: T, // s_add AND !is_deep AND result exp field == 255

    // === Cancellation path (effective subtraction, 6-byte field) ===
    pub cancel_align: ShrnCols<T, 6>, // align small_sig<<24 >> ediff into 48-bit field
    pub cancel_diff: [T; 6],          // raw_diff = large_working - aligned (6 bytes)
    pub cancel_borrow: [T; 5],        // borrow chain (5 intermediate borrows)
    pub cancel_shln: ShlnCols<T, 6>,  // norm = raw_diff << lshift (normalize)
    pub cancel_lshift: T,             // left-shift amount (leading-zero count)
    pub ediff_nz: T,                  // ediff != 0 (exclude same-exp subtraction)
    pub ediff_nz_inv: T,
    pub s_fb: T,            // finite AND both-normal (common finite precondition)
    pub s_cancel: T,        // finite AND !is_add AND both-normal AND !is_deep AND ediff!=0
    pub s_cancel_normal: T, // s_cancel AND result exp field in [1, 254]

    // === Result ===
    pub result_sign: T,
    pub result: [T; 4], // output f32 bits (4 limbs)
    pub is_special: T,
    pub is_nan_result: T,
    pub is_inf_result: T,
    pub is_zero_result: T,

    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FaddCoreCols<u8>)]
pub struct FaddCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FaddCoreAir {
    fn width(&self) -> usize {
        FaddCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FaddCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FaddCoreAir
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
        let cols: &FaddCoreCols<AB::Var> = local_core.borrow();

        builder.assert_bool(cols.is_valid);

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);

        // ── Phase 1: Decomposition ──────────────────────────────────────
        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;
        let a_mant_sum = cols.a_mant0 + cols.a_mant1 + cols.a_mant2;
        let b_mant_sum = cols.b_mant0 + cols.b_mant1 + cols.b_mant2;

        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.b_exp_lo);
        builder.assert_bool(cols.result_sign);

        // Range-check input mantissa/exponent bytes and the result bytes.
        for b in [cols.a_mant0, cols.a_mant1, cols.b_mant0, cols.b_mant1] {
            self.range_bus
                .range_check(b.into(), LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }
        for b in [cols.a_mant2, cols.b_mant2, cols.a_exp_hi, cols.b_exp_hi] {
            self.range_bus
                .range_check(b.into(), 7)
                .eval(builder, cols.is_valid.into());
        }
        for i in 0..4 {
            self.range_bus
                .range_check(cols.result[i].into(), LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }

        // ── Phase 2: Classification (stored as columns, fdiv-style) ──────────
        // exp_is_zero / exp_is_max / mant_is_zero are constrained only on valid
        // rows; on padding rows they stay zero, so the unconditional nan/inf/zero
        // products below vanish there. Keeps every constraint degree <= 3.
        let one = AB::Expr::ONE;
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_zero,
            one.clone() - a_exp.clone() * cols.a_exp_zero_inv,
        );
        builder.assert_zero(a_exp.clone() * cols.a_exp_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_max,
            one.clone() - (AB::Expr::from(f255) - a_exp.clone()) * cols.a_exp_max_inv,
        );
        builder.assert_zero((AB::Expr::from(f255) - a_exp.clone()) * cols.a_exp_is_max);
        builder.assert_zero(cols.a_exp_is_zero * cols.a_exp_is_max);
        builder.when(cols.is_valid).assert_eq(
            cols.a_mant_is_zero,
            one.clone() - a_mant_sum.clone() * cols.a_mant_sum_inv,
        );
        builder.assert_zero(a_mant_sum.clone() * cols.a_mant_is_zero);

        builder.when(cols.is_valid).assert_eq(
            cols.b_exp_is_zero,
            one.clone() - b_exp.clone() * cols.b_exp_zero_inv,
        );
        builder.assert_zero(b_exp.clone() * cols.b_exp_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.b_exp_is_max,
            one.clone() - (AB::Expr::from(f255) - b_exp.clone()) * cols.b_exp_max_inv,
        );
        builder.assert_zero((AB::Expr::from(f255) - b_exp.clone()) * cols.b_exp_is_max);
        builder.assert_zero(cols.b_exp_is_zero * cols.b_exp_is_max);
        builder.when(cols.is_valid).assert_eq(
            cols.b_mant_is_zero,
            one.clone() - b_mant_sum.clone() * cols.b_mant_sum_inv,
        );
        builder.assert_zero(b_mant_sum.clone() * cols.b_mant_is_zero);

        // nan / inf / zero derived from the classification columns (unconditional).
        builder.assert_eq(
            cols.a_is_nan,
            cols.a_exp_is_max * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(cols.a_is_inf, cols.a_exp_is_max * cols.a_mant_is_zero);
        builder.assert_eq(cols.a_is_zero, cols.a_exp_is_zero * cols.a_mant_is_zero);
        builder.assert_eq(
            cols.b_is_nan,
            cols.b_exp_is_max * (one.clone() - cols.b_mant_is_zero),
        );
        builder.assert_eq(cols.b_is_inf, cols.b_exp_is_max * cols.b_mant_is_zero);
        builder.assert_eq(cols.b_is_zero, cols.b_exp_is_zero * cols.b_mant_is_zero);

        // Verify result classification flags
        builder.assert_bool(cols.is_nan_result);
        builder.assert_bool(cols.is_inf_result);
        builder.assert_bool(cols.is_zero_result);
        builder.assert_bool(cols.is_special);
        builder.assert_bool(cols.is_add);
        builder.assert_bool(cols.a_is_larger);
        builder.assert_bool(cols.carry_out);
        builder.assert_bool(cols.sum_c1);
        builder.assert_bool(cols.sum_c2);
        builder.assert_bool(cols.rc0);
        builder.assert_bool(cols.rc1);
        builder.assert_bool(cols.rc2);
        builder.assert_bool(cols.is_deep);
        for i in 0..WORK_LIMBS {
            builder.assert_bool(cols.rs_lowbit[i]);
        }

        // ── Special-case selectors + result constraints ──────────────────
        self.eval_special_cases(builder, cols);

        // ── Phase 3: Alignment selection (ediff / a_is_larger) ───────────
        // ediff = |a_exp - b_exp|. Range-checking ediff to a byte forces a_is_larger
        // to be consistent with the true comparison (a wrong choice makes ediff wrap
        // to a huge field value, failing the range check).
        self.range_bus
            .range_check(cols.ediff.into(), LIMB_BITS)
            .eval(builder, cols.is_valid.into());
        let diff = a_exp.clone() - b_exp.clone();
        let expected_ediff =
            cols.a_is_larger * diff.clone() + (one.clone() - cols.a_is_larger) * (-diff.clone());
        builder.assert_eq(cols.ediff, expected_ediff);

        // is_add = (a_sign == b_sign) on valid rows.
        let sign_xor = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;
        builder.assert_eq(
            cols.is_add * cols.is_valid,
            (one.clone() - sign_xor) * cols.is_valid,
        );

        // ── Phase 4..7: finite same-sign (both-normal) result ─────────────
        self.eval_finite_add(builder, cols);
        // Cancellation (effective subtraction, 6-byte field).
        self.eval_finite_cancel(builder, cols);

        // ── Output ───────────────────────────────────────────────────────
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FADD_S);

        AdapterAirContext {
            to_pc: None,
            reads: [
                [
                    cols.a_mant0.into(),
                    cols.a_mant1.into(),
                    cols.a_mant2 + cols.a_exp_lo * f128,
                    cols.a_exp_hi + cols.a_sign * f128,
                ],
                [
                    cols.b_mant0.into(),
                    cols.b_mant1.into(),
                    cols.b_mant2 + cols.b_exp_lo * f128,
                    cols.b_exp_hi + cols.b_sign * f128,
                ],
            ]
            .into(),
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

// === Special-case constraints (testable standalone) ===

impl FaddCoreAir {
    /// Constrain the input-derived special-case selectors and the result bytes
    /// for NaN / +-inf (infinite input) / +-0 (0 + 0) results.
    pub fn eval_special_cases<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FaddCoreCols<AB::Var>,
    ) {
        let one = AB::Expr::ONE;
        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f127: AB::F = AB::F::from_u32(127);

        // inf_in = a_is_inf | b_is_inf
        builder.assert_eq(
            cols.inf_in,
            cols.a_is_inf + cols.b_is_inf - cols.a_is_inf * cols.b_is_inf,
        );
        // sign_diff = a_sign XOR b_sign
        let sign_diff = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;

        // s_nan: NaN input, or inf + (-inf).
        builder.assert_eq(
            cols.s_nan,
            cols.a_is_nan + cols.b_is_nan - cols.a_is_nan * cols.b_is_nan
                + cols.a_is_inf * cols.b_is_inf * sign_diff,
        );
        builder.assert_bool(cols.s_nan);

        // s_inf = !s_nan & inf_in ; s_zero_both = !s_nan & !inf_in & a_is_zero & b_is_zero
        builder.assert_eq(cols.s_inf, (one.clone() - cols.s_nan) * cols.inf_in);
        builder.assert_eq(
            cols.s_zero_both,
            (one.clone() - cols.s_nan)
                * (one.clone() - cols.inf_in)
                * cols.a_is_zero
                * cols.b_is_zero,
        );
        builder.assert_bool(cols.s_inf);
        builder.assert_bool(cols.s_zero_both);

        // is_nan_result must equal s_nan.
        builder.assert_eq(cols.is_nan_result, cols.s_nan);

        // Canonical NaN result.
        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &cb) in canon.iter().enumerate() {
            builder
                .when(cols.s_nan)
                .assert_eq(cols.result[i], AB::F::from_u32(cb as u32));
        }
        // +-inf result: sign = the infinite operand's sign. If both are inf
        // they have equal sign here (opposite-sign inf+inf is s_nan), so picking
        // a's sign when a is inf, else b's, is well-defined.
        let inf_sign = cols.a_is_inf * cols.a_sign
            + (one.clone() - cols.a_is_inf) * cols.b_is_inf * cols.b_sign;
        builder.when(cols.s_inf).assert_zero(cols.result[0]);
        builder.when(cols.s_inf).assert_zero(cols.result[1]);
        builder.when(cols.s_inf).assert_eq(cols.result[2], f128);
        builder
            .when(cols.s_inf)
            .assert_eq(cols.result[3], AB::Expr::from(f127) + inf_sign * f128);
        // +-0 result from 0 + 0: sign = a_sign AND b_sign.
        let zero_sign = cols.a_sign * cols.b_sign;
        builder.when(cols.s_zero_both).assert_zero(cols.result[0]);
        builder.when(cols.s_zero_both).assert_zero(cols.result[1]);
        builder.when(cols.s_zero_both).assert_zero(cols.result[2]);
        builder
            .when(cols.s_zero_both)
            .assert_eq(cols.result[3], zero_sign * f128);
    }

    /// Constrain the finite, same-sign, both-normal result path.
    ///
    /// Selector `s_add = s_finite AND is_add AND a_is_normal AND b_is_normal`
    /// (both operands normal, same sign). The result is then either a normal
    /// float (exp field in [1, 254]) or +-inf (overflow, exp field 255).
    ///
    /// Layout (working significand, 4 little-endian bytes): the significand
    /// `1.mantissa` occupies bits [31..8] (implicit bit at bit 31), bit 7 is the
    /// guard bit, bits [6..0] are the sticky region, and `large_working =
    /// large_sig << 8`.
    pub fn eval_finite_add<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FaddCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f24: AB::F = AB::F::from_u32(DEEP_EDIFF - 1);
        let f25: AB::F = AB::F::from_u32(DEEP_EDIFF);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(1 << LIMB_BITS);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;

        // ── Significand construction ────────────────────────────────────
        // Gated on is_valid: on padding rows the classification columns are all
        // zero, which would make the RHS evaluate to 1.
        builder.when(cols.is_valid).assert_eq(
            cols.a_is_normal,
            (one.clone() - cols.a_exp_is_zero) * (one.clone() - cols.a_exp_is_max),
        );
        builder.when(cols.is_valid).assert_eq(
            cols.b_is_normal,
            (one.clone() - cols.b_exp_is_zero) * (one.clone() - cols.b_exp_is_max),
        );
        let a_sig2 = cols.a_mant2 + cols.a_is_normal * f128;
        let b_sig2 = cols.b_mant2 + cols.b_is_normal * f128;
        let sel = cols.a_is_larger.into();
        let notsel = one.clone() - sel.clone();
        // large/small significand selection (gated on is_valid; padding stays 0).
        builder.when(cols.is_valid).assert_eq(
            cols.large_sig[0],
            sel.clone() * cols.a_mant0 + notsel.clone() * cols.b_mant0,
        );
        builder.when(cols.is_valid).assert_eq(
            cols.large_sig[1],
            sel.clone() * cols.a_mant1 + notsel.clone() * cols.b_mant1,
        );
        builder.when(cols.is_valid).assert_eq(
            cols.large_sig[2],
            sel.clone() * a_sig2.clone() + notsel.clone() * b_sig2.clone(),
        );
        builder.when(cols.is_valid).assert_eq(
            cols.small_sig[0],
            sel.clone() * cols.b_mant0 + notsel.clone() * cols.a_mant0,
        );
        builder.when(cols.is_valid).assert_eq(
            cols.small_sig[1],
            sel.clone() * cols.b_mant1 + notsel.clone() * cols.a_mant1,
        );
        builder.when(cols.is_valid).assert_eq(
            cols.small_sig[2],
            sel.clone() * b_sig2.clone() + notsel.clone() * a_sig2.clone(),
        );
        let large_exp = sel.clone() * a_exp.clone() + notsel.clone() * b_exp.clone();

        // ── Sound selector s_add ────────────────────────────────────────
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero_both;
        builder.assert_eq(
            cols.s_add,
            s_finite.clone() * cols.is_add * cols.a_is_normal * cols.b_is_normal,
        );
        builder.assert_bool(cols.s_add);

        // ── Deep clamp: is_deep = (ediff >= 25) ─────────────────────────
        //   !is_deep: ediff + deep_delta == 24   (deep_delta = 24 - ediff ∈ [0, 24])
        //   is_deep:  ediff - deep_delta == 25   (deep_delta = ediff - 25 ∈ [0, 230])
        // Both branches range-check deep_delta to a byte, which forces is_deep to
        // correctly indicate ediff >= 25.
        self.range_bus
            .range_check(cols.deep_delta.into(), LIMB_BITS)
            .eval(builder, cols.is_valid.into());
        builder
            .when(cols.is_valid)
            .when(cols.is_deep)
            .assert_eq(cols.ediff - cols.deep_delta, AB::Expr::from(f25));
        builder
            .when(cols.is_valid)
            .when(not(cols.is_deep))
            .assert_eq(cols.ediff + cols.deep_delta, AB::Expr::from(f24));

        // Alignment gadget enable = s_add AND !is_deep (stored as a degree-1 column
        // so it can be used as a low-degree `when` filter).
        builder.assert_eq(cols.align_enable, cols.s_add * (one.clone() - cols.is_deep));
        builder.assert_bool(cols.align_enable);
        let s_shallow = cols.align_enable;
        let s_deep = cols.s_add * cols.is_deep;

        // ── Alignment via ShrnGadget (shallow path) ─────────────────────
        // x = small_sig << 8 = [0, small_sig[0], small_sig[1], small_sig[2]].
        builder.when(cols.is_valid).assert_zero(cols.align.x[0]);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.align.x[1], cols.small_sig[0]);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.align.x[2], cols.small_sig[1]);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.align.x[3], cols.small_sig[2]);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.align);
        builder
            .when(cols.align_enable)
            .assert_eq(cols.align.shift, cols.ediff);
        // Alignment sticky = bits of the smaller operand lost below the field
        // (only meaningful when the shift gadget is enabled).
        builder.when(cols.align_enable).assert_eq(
            cols.lost_sticky,
            cols.align.guard + cols.align.sticky - cols.align.guard * cols.align.sticky,
        );
        builder.assert_bool(cols.lost_sticky);

        // ── Deep path: result = larger operand (no rounding) ────────────
        let large_b0 = sel.clone() * cols.a_mant0 + notsel.clone() * cols.b_mant0;
        let large_b1 = sel.clone() * cols.a_mant1 + notsel.clone() * cols.b_mant1;
        let large_b2 = sel.clone() * (cols.a_mant2 + cols.a_exp_lo * f128)
            + notsel.clone() * (cols.b_mant2 + cols.b_exp_lo * f128);
        let large_b3 = sel.clone() * (cols.a_exp_hi + cols.a_sign * f128)
            + notsel.clone() * (cols.b_exp_hi + cols.b_sign * f128);
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result[0], large_b0.clone());
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result[1], large_b1.clone());
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result[2], large_b2.clone());
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result[3], large_b3.clone());
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result_sign, cols.a_sign);
        builder
            .when(s_deep.clone())
            .assert_eq(cols.result_exp, large_exp.clone());

        // ── Sum (shallow): raw_sum = (large_sig << 8) + aligned ─────────
        // large_working = [0, large_sig[0], large_sig[1], large_sig[2]].
        // byte 0: raw_sum[0] = aligned[0] (no carry in, large_working[0] = 0).
        builder
            .when(s_shallow)
            .assert_eq(cols.raw_sum[0], cols.align.y[0]);
        builder.when(s_shallow).assert_eq(
            cols.raw_sum[1] + cols.sum_c1 * f256,
            cols.large_sig[0] + cols.align.y[1],
        );
        builder.when(s_shallow).assert_eq(
            cols.raw_sum[2] + cols.sum_c2 * f256,
            cols.large_sig[1] + cols.align.y[2] + cols.sum_c1,
        );
        builder.when(s_shallow).assert_eq(
            cols.raw_sum[3] + cols.carry_out * f256,
            cols.large_sig[2] + cols.align.y[3] + cols.sum_c2,
        );

        // ── Normalize: norm_sig = raw_sum >> carry_out ──────────────────
        for i in 0..WORK_LIMBS {
            builder
                .when(s_shallow)
                .assert_eq(cols.raw_sum[i], cols.rs_high7[i] * f2 + cols.rs_lowbit[i]);
            self.range_bus
                .range_check(cols.rs_high7[i].into(), 7)
                .eval(builder, s_shallow);
            // no carry out: norm_sig = raw_sum
            builder
                .when(s_shallow)
                .when(not(cols.carry_out))
                .assert_eq(cols.norm_sig[i], cols.raw_sum[i]);
            self.range_bus
                .range_check(cols.norm_sig[i].into(), LIMB_BITS)
                .eval(builder, s_shallow);
        }
        // carry out: norm_sig[i] = rs_high7[i] + rs_lowbit[i+1]*128 (i<3);
        // norm_sig[3] = rs_high7[3] + 128 (carry_out bit becomes the new implicit 1).
        for i in 0..(WORK_LIMBS - 1) {
            builder.when(s_shallow).when(cols.carry_out).assert_eq(
                cols.norm_sig[i],
                cols.rs_high7[i] + cols.rs_lowbit[i + 1] * f128,
            );
        }
        builder
            .when(s_shallow)
            .when(cols.carry_out)
            .assert_eq(cols.norm_sig[3], cols.rs_high7[3] + f128);
        // Exponent gains the carry.
        // (result_exp relation is constrained together with output reconstruction below.)

        // ── RNE rounding ────────────────────────────────────────────────
        // guard = bit 7 of norm_sig[0]; mant_lsb = bit 0 of norm_sig[1].
        builder
            .when(s_shallow)
            .assert_eq(cols.norm_sig[0], cols.ns0_lo + cols.guard * f128);
        builder.assert_bool(cols.guard);
        self.range_bus
            .range_check(cols.ns0_lo.into(), 7)
            .eval(builder, s_shallow);
        builder
            .when(s_shallow)
            .assert_eq(cols.norm_sig[1], cols.mant_lsb + cols.ns1_high * f2);
        builder.assert_bool(cols.mant_lsb);
        self.range_bus
            .range_check(cols.ns1_high.into(), 7)
            .eval(builder, s_shallow);
        // below_guard = ns0_lo + dropped + lost_sticky; dropped = carry_out * rs_lowbit[0].
        let dropped = cols.carry_out * cols.rs_lowbit[0];
        builder.when(s_shallow).assert_eq(
            cols.below_guard,
            cols.ns0_lo + dropped.clone() + cols.lost_sticky,
        );
        let bgz = one.clone() - cols.below_guard * cols.below_guard_inv;
        builder.assert_zero(cols.below_guard * bgz.clone());
        builder
            .when(s_shallow)
            .assert_eq(cols.sticky, one.clone() - bgz);
        builder.assert_bool(cols.sticky);
        let sticky_or_lsb = cols.sticky + cols.mant_lsb - cols.sticky * cols.mant_lsb;
        builder
            .when(s_shallow)
            .assert_eq(cols.round_up, cols.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Output reconstruction ───────────────────────────────────────
        // result_exp = r_exp_lo + r_exp_hi*2 = large_exp + carry_out + rc2.
        builder
            .when(s_shallow)
            .assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);
        builder.when(s_shallow).assert_eq(
            cols.r_exp_lo + cols.r_exp_hi * f2,
            large_exp + cols.carry_out + cols.rc2,
        );
        // result_sign = a_sign (same sign on this path).
        builder
            .when(s_shallow)
            .assert_eq(cols.result_sign, cols.a_sign);
        // Rounding carry chain over the 23-bit mantissa. rc0/rc1 are tied to the
        // result bytes on the normal-result path (s_normal) below; rc2 (and the
        // exponent) are tied here on the whole shallow path.
        builder.when(s_shallow).assert_eq(
            (cols.norm_sig[3] - f128) + cols.rc1,
            cols.rc2_rem + cols.rc2 * f128,
        );
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s_shallow);
        // result[3] = r_exp_hi + result_sign*128 (shared by normal and overflow).
        builder
            .when(s_shallow)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.result_sign * f128);

        // Sound "(r_exp_hi == 0)" / "(r_exp_hi == 127)" indicators (only needed on
        // the shallow path; their inverse witnesses are don't-care elsewhere).
        let he0 = one.clone() - cols.r_exp_hi * cols.r_exp_hi_zero_inv;
        builder
            .when(s_shallow)
            .assert_zero(cols.r_exp_hi * he0.clone());
        let delta127 = AB::Expr::from(f127) - cols.r_exp_hi;
        let he127 = one.clone() - delta127.clone() * cols.r_exp_hi_max_inv;
        builder
            .when(s_shallow)
            .assert_zero(delta127 * he127.clone());
        let exp_field_zero = (one.clone() - cols.r_exp_lo) * he0;
        let exp_field_max = cols.r_exp_lo * he127;
        // s_normal / s_overflow stored as columns (degree 1) so they can gate the
        // result-byte constraints without exceeding the max constraint degree.
        builder.assert_eq(
            cols.s_normal,
            s_shallow * (one.clone() - exp_field_zero.clone() - exp_field_max.clone()),
        );
        builder.assert_eq(cols.s_overflow, s_shallow * exp_field_max.clone());
        builder.assert_bool(cols.s_normal);
        builder.assert_bool(cols.s_overflow);

        // Normal result: tie mantissa bytes to the rounded significand.
        builder.when(cols.s_normal).assert_eq(
            cols.result[0] + cols.rc0 * f256,
            cols.norm_sig[1] + cols.round_up,
        );
        builder.when(cols.s_normal).assert_eq(
            cols.result[1] + cols.rc1 * f256,
            cols.norm_sig[2] + cols.rc0,
        );
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        // Overflow result: +-inf (exp 255, mantissa 0).
        builder.when(cols.s_overflow).assert_zero(cols.result[0]);
        builder.when(cols.s_overflow).assert_zero(cols.result[1]);
        builder
            .when(cols.s_overflow)
            .assert_eq(cols.result[2], f128);
    }

    /// Constrain the cancellation (effective subtraction) result path using a
    /// 6-byte working field. For ediff ∈ [1, 24], `aligned = (small_sig << 24) >>
    /// ediff` via `ShrnGadget<6>` (no precision loss), `raw_diff = large_working -
    /// aligned` (6-byte borrow chain), then `norm = raw_diff << lshift` via
    /// `ShlnGadget<6>` (implicit bit → position 47). RNE rounding + output recon
    /// reuse the existing columns under `s_cancel` / `s_cancel_normal`.
    pub fn eval_finite_cancel<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FaddCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(1 << LIMB_BITS);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;

        // ── Sound selectors ─────────────────────────────────────────────
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero_both;
        builder.assert_eq(
            cols.s_fb,
            s_finite.clone() * cols.a_is_normal * cols.b_is_normal,
        );
        builder.assert_bool(cols.s_fb);
        // ediff != 0 indicator (excludes same-exp subtraction).
        let ediff_is_zero = one.clone() - cols.ediff * cols.ediff_nz_inv;
        builder.assert_zero(cols.ediff * ediff_is_zero.clone());
        builder
            .when(cols.is_valid)
            .assert_eq(cols.ediff_nz, one.clone() - ediff_is_zero);
        builder.assert_bool(cols.ediff_nz);
        // s_cancel = s_fb AND (NOT is_add) AND (NOT is_deep) AND ediff!=0.
        builder.assert_eq(
            cols.s_cancel,
            cols.s_fb * (one.clone() - cols.is_add) * (one.clone() - cols.is_deep) * cols.ediff_nz,
        );
        builder.assert_bool(cols.s_cancel);
        let s = cols.s_cancel;

        // ── Alignment via ShrnGadget<6> ──────────────────────────────────
        // x = small_sig << 24 = [0, 0, 0, small_sig bytes]. No precision loss
        // for ediff <= 24 (all bits remain within the 48-bit field).
        for i in 0..3 {
            builder.when(s).assert_zero(cols.cancel_align.x[i]);
        }
        builder
            .when(s)
            .assert_eq(cols.cancel_align.x[3], cols.small_sig[0]);
        builder
            .when(s)
            .assert_eq(cols.cancel_align.x[4], cols.small_sig[1]);
        builder
            .when(s)
            .assert_eq(cols.cancel_align.x[5], cols.small_sig[2]);
        builder
            .when(s)
            .assert_eq(cols.cancel_align.shift, cols.ediff);
        builder.assert_eq(cols.cancel_align.enable, s);
        let align_shrn = ShrnGadget::new(self.range_bus);
        align_shrn.eval_shrn(builder, &cols.cancel_align);

        // ── 6-byte borrow chain: raw_diff = large_working - aligned ──────
        // large_working = [0, 0, 0, large_sig[0], large_sig[1], large_sig[2]].
        // raw_diff >= 0 for ediff >= 1 (large dominates).
        let lw = |i: usize| -> AB::Expr {
            match i {
                0..=2 => AB::Expr::ZERO,
                3 => cols.large_sig[0].into(),
                4 => cols.large_sig[1].into(),
                5 => cols.large_sig[2].into(),
                _ => unreachable!(),
            }
        };
        for i in 0..5 {
            builder.assert_bool(cols.cancel_borrow[i]);
        }
        // byte 0: raw_diff[0] + aligned[0] + 0 = 0 + borrow_out0*256
        builder.when(s).assert_eq(
            cols.cancel_diff[0] + cols.cancel_align.y[0],
            cols.cancel_borrow[0] * f256,
        );
        // bytes 1..5: raw_diff[i] + aligned[i] + borrow_in = lw[i] + borrow_out*256
        for i in 1..5 {
            let bin = cols.cancel_borrow[i - 1];
            builder.when(s).assert_eq(
                cols.cancel_diff[i] + cols.cancel_align.y[i] + bin,
                lw(i) + cols.cancel_borrow[i] * f256,
            );
        }
        // byte 5 (top): no borrow out (raw_diff >= 0).
        builder.when(s).assert_eq(
            cols.cancel_diff[5] + cols.cancel_align.y[5] + cols.cancel_borrow[4],
            lw(5),
        );

        // ── Left-shift normalize via ShlnGadget<6> ──────────────────────
        // norm = raw_diff << lshift, implicit bit at position 47.
        for i in 0..6 {
            builder
                .when(s)
                .assert_eq(cols.cancel_shln.x[i], cols.cancel_diff[i]);
        }
        builder
            .when(s)
            .assert_eq(cols.cancel_shln.shift, cols.cancel_lshift);
        builder.assert_eq(cols.cancel_shln.enable, s);
        let shln = ShlnGadget::new(self.range_bus);
        shln.eval_shln(builder, &cols.cancel_shln);
        // Range-check norm bytes (shln.y) to 8 bits.
        for i in 0..6 {
            self.range_bus
                .range_check(cols.cancel_shln.y[i].into(), LIMB_BITS)
                .eval(builder, s.into());
        }

        // ── RNE rounding ────────────────────────────────────────────────
        // 6-byte norm layout: implicit at 47 (byte5 bit7), mantissa [46..24]
        // (bytes 3,4,5&0x7F), guard = bit 23 (byte2 bit7), sticky = bytes 0,1,2&0x7F.
        let norm = &cols.cancel_shln.y;
        // guard = bit7 of norm[2].
        builder
            .when(s)
            .assert_eq(norm[2], cols.ns0_lo + cols.guard * f128);
        builder.assert_bool(cols.guard);
        self.range_bus
            .range_check(cols.ns0_lo.into(), 7)
            .eval(builder, s.into());
        // mant_lsb = bit0 of norm[3].
        builder
            .when(s)
            .assert_eq(norm[3], cols.mant_lsb + cols.ns1_high * f2);
        builder.assert_bool(cols.mant_lsb);
        self.range_bus
            .range_check(cols.ns1_high.into(), 7)
            .eval(builder, s.into());
        // below_guard = ns0_lo + norm[0] + norm[1] (no dropped, no lost_sticky for cancellation).
        builder
            .when(s)
            .assert_eq(cols.below_guard, cols.ns0_lo + norm[0] + norm[1]);
        let bgz = one.clone() - cols.below_guard * cols.below_guard_inv;
        builder.assert_zero(cols.below_guard * bgz.clone());
        builder.when(s).assert_eq(cols.sticky, one.clone() - bgz);
        builder.assert_bool(cols.sticky);
        let sticky_or_lsb = cols.sticky + cols.mant_lsb - cols.sticky * cols.mant_lsb;
        builder
            .when(s)
            .assert_eq(cols.round_up, cols.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Output reconstruction ───────────────────────────────────────
        let large_exp =
            cols.a_is_larger * a_exp.clone() + (one.clone() - cols.a_is_larger) * b_exp.clone();
        let large_sign =
            cols.a_is_larger * cols.a_sign + (one.clone() - cols.a_is_larger) * cols.b_sign;
        // result_exp = r_exp_lo + 2*r_exp_hi = large_exp - lshift + rc2.
        builder
            .when(s)
            .assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);
        builder.when(s).assert_eq(
            cols.r_exp_lo + cols.r_exp_hi * f2,
            large_exp.clone() - cols.cancel_lshift + cols.rc2,
        );
        builder
            .when(s)
            .assert_eq(cols.result_sign, large_sign.clone());
        // result mantissa from norm[3..5] + round_up.
        builder
            .when(s)
            .assert_eq(cols.result[0] + cols.rc0 * f256, norm[3] + cols.round_up);
        builder
            .when(s)
            .assert_eq(cols.result[1] + cols.rc1 * f256, norm[4] + cols.rc0);
        // top7c = (norm[5] - 128) + rc1 = rc2_rem + rc2*128.
        builder
            .when(s)
            .assert_eq((norm[5] - f128) + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s.into());
        builder
            .when(s)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        builder
            .when(s)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.result_sign * f128);

        // ── Normal-result selector ──────────────────────────────────────
        let he0 = one.clone() - cols.r_exp_hi * cols.r_exp_hi_zero_inv;
        builder.when(s).assert_zero(cols.r_exp_hi * he0.clone());
        let delta127 = AB::Expr::from(f127) - cols.r_exp_hi;
        let he127 = one.clone() - delta127.clone() * cols.r_exp_hi_max_inv;
        builder.when(s).assert_zero(delta127 * he127.clone());
        let exp_field_zero = (one.clone() - cols.r_exp_lo) * he0;
        let exp_field_max = cols.r_exp_lo * he127;
        builder.assert_eq(
            cols.s_cancel_normal,
            s * (one.clone() - exp_field_zero.clone() - exp_field_max.clone()),
        );
        builder.assert_bool(cols.s_cancel_normal);
        let scn = cols.s_cancel_normal;
        // Tie result bytes under s_cancel_normal (same recon as above, gated
        // separately for the normal-exp subset).
        builder
            .when(scn)
            .assert_eq(cols.result[0] + cols.rc0 * f256, norm[3] + cols.round_up);
        builder
            .when(scn)
            .assert_eq(cols.result[1] + cols.rc1 * f256, norm[4] + cols.rc0);
        builder
            .when(scn)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FaddCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FaddExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FaddExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FaddCoreRecord),
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
        let Instruction { opcode, .. } = instruction;
        debug_assert_eq!(
            FpOpcode::from_usize(opcode.local_opcode_idx(self.offset)),
            FpOpcode::FADD_S
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        // Native f32 add = IEEE 754 correctly-rounded (RNE), canonical NaN
        let result = fadd_s(u32::from_le_bytes(a), u32::from_le_bytes(b)).to_le_bytes();

        core_record.a = a;
        core_record.b = b;

        self.adapter.write(
            state.memory,
            instruction,
            [result].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

// === Filler ===

pub struct FaddFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FaddFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FaddFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &FaddCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FaddCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let result_bits = fadd_s(a_bits, b_bits);

        fill_fadd_witness(core_row, a_bits, b_bits, result_bits, &self.range_checker);
    }
}

// === Witness computation ===

fn decompose(bits: u32) -> (u32, u32, u32) {
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    (sign, exp, mant)
}

fn mant_to_limbs(mant: u32) -> [u32; 3] {
    [mant & 0xFF, (mant >> 8) & 0xFF, (mant >> 16) & 0x7F]
}

fn fill_fadd_witness<F: PrimeField32>(
    row: &mut FaddCoreCols<F>,
    a_bits: u32,
    b_bits: u32,
    result_bits: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let (a_sign, a_exp, a_mant) = decompose(a_bits);
    let (b_sign, b_exp, b_mant) = decompose(b_bits);

    // Decomposition
    let a_ml = mant_to_limbs(a_mant);
    let b_ml = mant_to_limbs(b_mant);

    row.a_sign = F::from_u32(a_sign);
    row.a_exp_lo = F::from_u32(a_exp & 1);
    row.a_exp_hi = F::from_u32(a_exp >> 1);
    row.a_mant0 = F::from_u32(a_ml[0]);
    row.a_mant1 = F::from_u32(a_ml[1]);
    row.a_mant2 = F::from_u32(a_ml[2]);

    row.b_sign = F::from_u32(b_sign);
    row.b_exp_lo = F::from_u32(b_exp & 1);
    row.b_exp_hi = F::from_u32(b_exp >> 1);
    row.b_mant0 = F::from_u32(b_ml[0]);
    row.b_mant1 = F::from_u32(b_ml[1]);
    row.b_mant2 = F::from_u32(b_ml[2]);

    // Balance the input-byte range checks emitted in eval (gated on is_valid).
    range_checker.add_count(a_ml[0], LIMB_BITS);
    range_checker.add_count(a_ml[1], LIMB_BITS);
    range_checker.add_count(b_ml[0], LIMB_BITS);
    range_checker.add_count(b_ml[1], LIMB_BITS);
    range_checker.add_count(a_ml[2], 7);
    range_checker.add_count(b_ml[2], 7);
    range_checker.add_count(a_exp >> 1, 7);
    range_checker.add_count(b_exp >> 1, 7);

    // Classification inverse witnesses
    let a_ef = F::from_u32(a_exp);
    let b_ef = F::from_u32(b_exp);
    row.a_exp_zero_inv = if a_exp != 0 {
        a_ef.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.a_exp_max_inv = if a_exp != 255 {
        (F::from_u32(255) - a_ef).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let a_ms = F::from_u32(a_ml[0] + a_ml[1] + a_ml[2]);
    row.a_mant_sum_inv = if a_ml[0] + a_ml[1] + a_ml[2] != 0 {
        a_ms.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.b_exp_zero_inv = if b_exp != 0 {
        b_ef.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.b_exp_max_inv = if b_exp != 255 {
        (F::from_u32(255) - b_ef).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let b_ms = F::from_u32(b_ml[0] + b_ml[1] + b_ml[2]);
    row.b_mant_sum_inv = if b_ml[0] + b_ml[1] + b_ml[2] != 0 {
        b_ms.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // Classification
    let a_is_nan = a_exp == 255 && a_mant != 0;
    let a_is_inf = a_exp == 255 && a_mant == 0;
    let a_is_zero = a_exp == 0 && a_mant == 0;
    let a_is_normal = (1..=254).contains(&a_exp);
    let b_is_nan = b_exp == 255 && b_mant != 0;
    let b_is_inf = b_exp == 255 && b_mant == 0;
    let b_is_zero = b_exp == 0 && b_mant == 0;
    let b_is_normal = (1..=254).contains(&b_exp);

    // Classification columns (constrained in eval).
    row.a_exp_is_zero = F::from_bool(a_exp == 0);
    row.a_exp_is_max = F::from_bool(a_exp == 255);
    row.a_mant_is_zero = F::from_bool(a_mant == 0);
    row.b_exp_is_zero = F::from_bool(b_exp == 0);
    row.b_exp_is_max = F::from_bool(b_exp == 255);
    row.b_mant_is_zero = F::from_bool(b_mant == 0);
    row.a_is_nan = F::from_bool(a_is_nan);
    row.a_is_inf = F::from_bool(a_is_inf);
    row.a_is_zero = F::from_bool(a_is_zero);
    row.b_is_nan = F::from_bool(b_is_nan);
    row.b_is_inf = F::from_bool(b_is_inf);
    row.b_is_zero = F::from_bool(b_is_zero);
    row.a_is_normal = F::from_bool(a_is_normal);
    row.b_is_normal = F::from_bool(b_is_normal);

    // Result
    let (r_sign, r_exp, _r_mant) = decompose(result_bits);
    row.result_sign = F::from_u32(r_sign);
    row.result_exp = F::from_u32(r_exp);
    row.result[0] = F::from_u32(result_bits & 0xFF);
    row.result[1] = F::from_u32((result_bits >> 8) & 0xFF);
    row.result[2] = F::from_u32((result_bits >> 16) & 0xFF);
    row.result[3] = F::from_u32((result_bits >> 24) & 0xFF);
    for i in 0..4 {
        range_checker.add_count((result_bits >> (i * 8)) & 0xFF, LIMB_BITS);
    }

    let is_nan_res = a_is_nan || b_is_nan || (a_is_inf && b_is_inf && a_sign != b_sign);
    let is_inf_res = !is_nan_res && (a_is_inf || b_is_inf || r_exp == 255);
    let is_zero_res = !is_nan_res && (r_exp == 0 && result_bits & 0x7FFFFF == 0);
    row.is_nan_result = F::from_bool(is_nan_res);
    row.is_inf_result = F::from_bool(is_inf_res);
    row.is_zero_result = F::from_bool(is_zero_res);
    row.is_special = F::from_bool(is_nan_res || is_inf_res || is_zero_res);

    // Input-derived special-case selectors (constrained in eval_special_cases).
    let inf_in = a_is_inf || b_is_inf;
    let s_inf = !is_nan_res && inf_in;
    let s_zero_both = !is_nan_res && !inf_in && a_is_zero && b_is_zero;
    row.inf_in = F::from_bool(inf_in);
    row.s_nan = F::from_bool(is_nan_res);
    row.s_inf = F::from_bool(s_inf);
    row.s_zero_both = F::from_bool(s_zero_both);

    // Alignment selection
    let same_sign = a_sign == b_sign;
    let a_is_larger = a_exp >= b_exp;
    let ediff = if a_is_larger {
        a_exp - b_exp
    } else {
        b_exp - a_exp
    };
    row.is_add = F::from_bool(same_sign);
    row.a_is_larger = F::from_bool(a_is_larger);
    row.ediff = F::from_u32(ediff);
    range_checker.add_count(ediff, LIMB_BITS);

    // Sound selector: finite, same-sign, both-normal.
    let s_add = !is_nan_res && !inf_in && !s_zero_both && same_sign && a_is_normal && b_is_normal;
    row.s_add = F::from_bool(s_add);

    // Selected significands (with implicit bit, since both normal on the s_add path;
    // off-path values are still well-defined mantissa selections).
    let a_sig = if a_is_normal {
        a_mant | 0x800000
    } else {
        a_mant
    };
    let b_sig = if b_is_normal {
        b_mant | 0x800000
    } else {
        b_mant
    };
    let (large_sig, small_sig, large_exp, large_sign) = if a_is_larger {
        (a_sig, b_sig, a_exp, a_sign)
    } else {
        (b_sig, a_sig, b_exp, b_sign)
    };
    for i in 0..SIG_NUM_LIMBS {
        row.large_sig[i] = F::from_u32((large_sig >> (i * LIMB_BITS)) & 0xFF);
        row.small_sig[i] = F::from_u32((small_sig >> (i * LIMB_BITS)) & 0xFF);
    }

    let is_deep = ediff >= DEEP_EDIFF;
    row.is_deep = F::from_bool(is_deep);
    row.deep_delta = F::from_u32(if is_deep {
        ediff - DEEP_EDIFF
    } else {
        DEEP_EDIFF - 1 - ediff
    });
    range_checker.add_count(row.deep_delta.as_canonical_u32(), LIMB_BITS);

    let s_shallow = s_add && !is_deep;
    row.align_enable = F::from_bool(s_shallow);

    // Cancellation selectors.
    let s_finite_bool = !is_nan_res && !inf_in && !s_zero_both;
    row.ediff_nz = F::from_bool(ediff != 0);
    row.ediff_nz_inv = if ediff != 0 {
        F::from_u32(ediff).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let s_cancel =
        s_finite_bool && !same_sign && a_is_normal && b_is_normal && !is_deep && ediff != 0;
    row.s_fb = F::from_bool(s_finite_bool && a_is_normal && b_is_normal);
    row.s_cancel = F::from_bool(s_cancel);
    row.s_cancel_normal = F::ZERO;

    // ── Alignment via ShrnGadget (shallow), else disabled ───────────────
    let x: [u32; WORK_LIMBS] = [
        0,
        small_sig & 0xFF,
        (small_sig >> 8) & 0xFF,
        (small_sig >> 16) & 0xFF,
    ];
    if s_shallow {
        let shift = ediff as usize;
        row.align = make_shrn_cols::<F>(x, shift, range_checker);
        let ag = row.align.guard.as_canonical_u32();
        let as_ = row.align.sticky.as_canonical_u32();
        let lost_sticky = (ag != 0) || (as_ != 0);
        row.lost_sticky = F::from_bool(lost_sticky);
    } else {
        // Disabled gadget: do NOT register range-checker counts (the gadget emits
        // no sends when enable = 0). Only align.x must match small_sig<<8, which
        // is constrained unconditionally on is_valid.
        let mut a = ShrnCols::<F, WORK_LIMBS>::default();
        a.x[1] = F::from_u32(x[1]);
        a.x[2] = F::from_u32(x[2]);
        a.x[3] = F::from_u32(x[3]);
        row.align = a;
        row.lost_sticky = F::from_bool(is_deep && (small_sig != 0));
    }

    if !s_shallow {
        if s_cancel {
            fill_cancel_fadd(
                row,
                large_sig,
                small_sig,
                large_exp,
                large_sign,
                r_exp,
                result_bits,
                a_bits,
                b_bits,
                ediff,
                range_checker,
            );
            row.is_valid = F::ONE;
            return;
        }
        // Deep / ediff=0 opposite-sign / subnormal-input / special: zero finite
        // columns (both shallow and cancel) so unconditional assertions hold.
        zero_finite_columns(row);
        row.cancel_align = ShrnCols::<F, 6>::default();
        for i in 0..6 {
            row.cancel_diff[i] = F::ZERO;
        }
        for i in 0..5 {
            row.cancel_borrow[i] = F::ZERO;
        }
        row.cancel_shln = ShlnCols::<F, 6>::default();
        row.cancel_lshift = F::ZERO;
        row.result_exp = F::from_u32(r_exp);
        row.is_valid = F::ONE;
        return;
    }
    // Shallow path: zero the cancellation columns.
    row.cancel_align = ShrnCols::<F, 6>::default();
    for i in 0..6 {
        row.cancel_diff[i] = F::ZERO;
    }
    for i in 0..5 {
        row.cancel_borrow[i] = F::ZERO;
    }
    row.cancel_shln = ShlnCols::<F, 6>::default();
    row.cancel_lshift = F::ZERO;

    // ── Sum: raw_sum = (large_sig << 8) + aligned ───────────────────────
    let large_working = (large_sig as u64) << 8; // [0, large_sig bytes...]
    let aligned = (0..WORK_LIMBS).fold(0u64, |acc, i| {
        acc | ((row.align.y[i].as_canonical_u32() as u64) << (i * LIMB_BITS))
    });
    let raw = large_working + aligned;
    let raw_bytes: [u32; WORK_LIMBS] =
        core::array::from_fn(|i| ((raw >> (i * LIMB_BITS)) & 0xFF) as u32);
    let carry_out = ((raw >> 32) & 1) as u32;
    let sum_c1 =
        ((row.large_sig[0].as_canonical_u32() + row.align.y[1].as_canonical_u32()) >> 8) & 1;
    let sum_c2 =
        ((row.large_sig[1].as_canonical_u32() + row.align.y[2].as_canonical_u32() + sum_c1) >> 8)
            & 1;
    for (i, &b) in raw_bytes.iter().enumerate() {
        row.raw_sum[i] = F::from_u32(b);
    }
    row.sum_c1 = F::from_u32(sum_c1);
    row.sum_c2 = F::from_u32(sum_c2);
    row.carry_out = F::from_u32(carry_out);

    // ── Normalize: norm = raw >> carry_out ──────────────────────────────
    let norm = raw >> carry_out;
    for (i, &b) in raw_bytes.iter().enumerate() {
        row.rs_high7[i] = F::from_u32(b >> 1);
        row.rs_lowbit[i] = F::from_u32(b & 1);
        row.norm_sig[i] = F::from_u32(((norm >> (i * LIMB_BITS)) & 0xFF) as u32);
        range_checker.add_count(row.rs_high7[i].as_canonical_u32(), 7);
        range_checker.add_count(row.norm_sig[i].as_canonical_u32(), LIMB_BITS);
    }

    // ── RNE rounding ────────────────────────────────────────────────────
    let ns0 = (norm & 0xFF) as u32;
    let ns1 = ((norm >> 8) & 0xFF) as u32;
    let guard = (ns0 >> 7) & 1;
    let ns0_lo = ns0 & 0x7F;
    let mant_lsb = ns1 & 1;
    let ns1_high = ns1 >> 1;
    let dropped = carry_out * (raw_bytes[0] & 1);
    let below_guard = ns0_lo + dropped + row.lost_sticky.as_canonical_u32();
    let sticky = below_guard != 0;
    let round_up = guard == 1 && (sticky || mant_lsb == 1);

    row.guard = F::from_u32(guard);
    row.ns0_lo = F::from_u32(ns0_lo);
    row.ns1_high = F::from_u32(ns1_high);
    row.mant_lsb = F::from_u32(mant_lsb);
    row.below_guard = F::from_u32(below_guard);
    row.below_guard_inv = if below_guard != 0 {
        F::from_u32(below_guard).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.sticky = F::from_bool(sticky);
    row.round_up = F::from_bool(round_up);
    range_checker.add_count(ns0_lo, 7);
    range_checker.add_count(ns1_high, 7);

    // ── Output reconstruction ───────────────────────────────────────────
    let s0 = ((norm >> 8) & 0xFF) as u32 + round_up as u32; // norm_sig[1] + round_up
    let rc0 = s0 >> 8;
    let s1 = ((norm >> 16) & 0xFF) as u32 + rc0; // norm_sig[2] + rc0
    let rc1 = s1 >> 8;
    let top7c = (((norm >> 24) & 0x7F) as u32) + rc1; // (norm_sig[3]-128) + rc1
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    let result_exp_calc = large_exp + carry_out + rc2;

    row.rc0 = F::from_u32(rc0);
    row.rc1 = F::from_u32(rc1);
    row.rc2 = F::from_u32(rc2);
    row.rc2_rem = F::from_u32(rc2_rem);
    row.r_exp_lo = F::from_u32(r_exp & 1);
    row.r_exp_hi = F::from_u32(r_exp >> 1);
    row.r_exp_hi_zero_inv = if r_exp >> 1 != 0 {
        F::from_u32(r_exp >> 1).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.r_exp_hi_max_inv = if r_exp >> 1 != 127 {
        (F::from_u32(127) - F::from_u32(r_exp >> 1))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    range_checker.add_count(rc2_rem, 7);

    // Sanity: witness must agree with the native result on this path (only for a
    // normal result; overflow -> inf does not match the significand reconstruction).
    if (1u32..=254).contains(&r_exp) {
        debug_assert_eq!(
            result_exp_calc, r_exp,
            "fadd finite exponent mismatch: a={:08x} b={:08x}",
            a_bits, b_bits
        );
        let recon0 = (((norm >> 8) & 0xFF) as u32 + round_up as u32) & 0xFF;
        let recon1 = (((norm >> 16) & 0xFF) as u32 + rc0) & 0xFF;
        let recon2 = rc2_rem + (r_exp & 1) * 128;
        debug_assert_eq!(
            recon0,
            result_bits & 0xFF,
            "fadd finite mantissa byte0 mismatch: a={:08x} b={:08x}",
            a_bits,
            b_bits
        );
        debug_assert_eq!(recon1, (result_bits >> 8) & 0xFF);
        debug_assert_eq!(recon2, (result_bits >> 16) & 0xFF);
    }

    let _ = result_exp_calc;
    row.s_normal = F::from_bool(s_shallow && (1u32..=254).contains(&r_exp));
    row.s_overflow = F::from_bool(s_shallow && r_exp == 255);
    row.is_valid = F::ONE;
}

/// Zero the finite-path columns (used when the s_shallow gate is off, so the
/// unconditional bool assertions still hold).
fn zero_finite_columns<F: PrimeField32>(row: &mut FaddCoreCols<F>) {
    for i in 0..WORK_LIMBS {
        row.raw_sum[i] = F::ZERO;
        row.norm_sig[i] = F::ZERO;
        row.rs_high7[i] = F::ZERO;
        row.rs_lowbit[i] = F::ZERO;
    }
    row.sum_c1 = F::ZERO;
    row.sum_c2 = F::ZERO;
    row.carry_out = F::ZERO;
    row.guard = F::ZERO;
    row.ns0_lo = F::ZERO;
    row.ns1_high = F::ZERO;
    row.mant_lsb = F::ZERO;
    row.below_guard = F::ZERO;
    row.below_guard_inv = F::ZERO;
    row.sticky = F::ZERO;
    row.round_up = F::ZERO;
    row.rc0 = F::ZERO;
    row.rc1 = F::ZERO;
    row.rc2 = F::ZERO;
    row.rc2_rem = F::ZERO;
    row.r_exp_lo = F::ZERO;
    row.r_exp_hi = F::ZERO;
    row.r_exp_hi_zero_inv = F::ZERO;
    row.r_exp_hi_max_inv = F::ZERO;
    row.s_normal = F::ZERO;
    row.s_overflow = F::ZERO;
}

/// Build a `ShrnCols` witness (mirrors `fp_shift::tests::make_cols`) and register
/// the corresponding range-checker counts. `enable` defaults to 1.
fn make_shrn_cols<F: PrimeField32>(
    x: [u32; WORK_LIMBS],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShrnCols<F, WORK_LIMBS> {
    let shift = shift.min(WORK_LIMBS * SHLIMB_BITS - 1);
    let (y, q, r) = run_shrn(&x, shift);
    let (guard, sticky) = run_shrn_round(&x, shift);
    let mut limb_shift_marker = [F::ZERO; WORK_LIMBS];
    let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
    limb_shift_marker[q] = F::ONE;
    bit_shift_marker[r] = F::ONE;

    let pc = x[q] & ((1u32 << r) - 1);
    let pmsb = if r >= 1 { (pc >> (r - 1)) & 1 } else { 0 };
    let prest = if r >= 1 {
        pc - pmsb * (1u32 << (r - 1))
    } else {
        0
    };
    let hf = if q >= 1 { x[q - 1] } else { 0 };
    let hmsb = if r == 0 { (hf >> 7) & 1 } else { 0 };
    let hrest = if r == 0 { hf - hmsb * 128 } else { 0 };
    let full_rge1: u32 = (0..q).map(|k| x[k]).sum();
    let full_r0: u32 = (0..q.saturating_sub(1)).map(|k| x[k]).sum();
    let lost_combined = if r >= 1 {
        prest + full_rge1
    } else {
        hrest + full_r0
    };
    let combined_inv = if lost_combined != 0 {
        F::from_u32(lost_combined).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // Register range-checker counts to balance ShrnGadget's interaction sends.
    for i in 0..WORK_LIMBS {
        range_checker.add_count(x[i] & ((1u32 << r) - 1), r); // bit_shift_carry[i] @ r bits
        range_checker.add_count(y[i], LIMB_BITS);
    }
    if r >= 1 {
        range_checker.add_count(prest, r - 1);
    } else {
        range_checker.add_count(hrest, 7);
    }

    ShrnCols {
        x: core::array::from_fn(|i| F::from_u32(x[i])),
        shift: F::from_u32(shift as u32),
        enable: F::ONE,
        limb_shift_marker,
        bit_shift_marker,
        bit_mult: F::from_u32(1u32 << r),
        bit_shift_carry: core::array::from_fn(|i| F::from_u32(x[i] & ((1u32 << r) - 1))),
        y: core::array::from_fn(|i| F::from_u32(y[i])),
        guard: F::from_u32(guard),
        sticky: F::from_bool(sticky),
        pmsb: F::from_u32(pmsb),
        prest: F::from_u32(prest),
        hmsb: F::from_u32(hmsb),
        hrest: F::from_u32(hrest),
        lost_combined: F::from_u32(lost_combined),
        combined_inv,
    }
}

/// Fill the cancellation (effective subtraction) witness via the 6-byte field.
#[allow(clippy::too_many_arguments)]
fn fill_cancel_fadd<F: PrimeField32>(
    row: &mut FaddCoreCols<F>,
    large_sig: u32,
    small_sig: u32,
    large_exp: u32,
    large_sign: u32,
    r_exp: u32,
    result_bits: u32,
    a_bits: u32,
    b_bits: u32,
    ediff: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    // ── 6-byte alignment: aligned = (small_sig << 24) >> ediff ─────────
    let x: [u32; 6] = [
        0,
        0,
        0,
        small_sig & 0xFF,
        (small_sig >> 8) & 0xFF,
        (small_sig >> 16) & 0xFF,
    ];
    row.cancel_align = make_shrn_cols6::<F>(x, ediff as usize, range_checker);

    // ── 6-byte subtraction: raw_diff = large_working - aligned ─────────
    let aligned: u64 = (0..6).fold(0u64, |acc, i| {
        acc | ((row.cancel_align.y[i].as_canonical_u32() as u64) << (i * 8))
    });
    let large_working6: u64 = (large_sig as u64) << 24;
    let raw_diff = large_working6.wrapping_sub(aligned);
    debug_assert!(
        raw_diff < (1u64 << 48),
        "fadd cancel raw_diff overflow: a={:08x} b={:08x}",
        a_bits,
        b_bits
    );
    // Borrow chain: recover borrows from the byte-level subtraction.
    let lw_bytes: [u32; 6] = [
        (large_working6 & 0xFF) as u32,
        ((large_working6 >> 8) & 0xFF) as u32,
        ((large_working6 >> 16) & 0xFF) as u32,
        ((large_working6 >> 24) & 0xFF) as u32,
        ((large_working6 >> 32) & 0xFF) as u32,
        ((large_working6 >> 40) & 0xFF) as u32,
    ];
    let al_bytes: [u32; 6] = [
        (aligned & 0xFF) as u32,
        ((aligned >> 8) & 0xFF) as u32,
        ((aligned >> 16) & 0xFF) as u32,
        ((aligned >> 24) & 0xFF) as u32,
        ((aligned >> 32) & 0xFF) as u32,
        ((aligned >> 40) & 0xFF) as u32,
    ];
    let mut borrow = 0i32;
    for i in 0..6 {
        let d = lw_bytes[i] as i32 - al_bytes[i] as i32 - borrow;
        if d < 0 {
            row.cancel_diff[i] = F::from_u32((d + 256) as u32);
            borrow = 1;
        } else {
            row.cancel_diff[i] = F::from_u32(d as u32);
            borrow = 0;
        }
        if i < 5 {
            row.cancel_borrow[i] = F::from_u32(borrow as u32);
        }
    }
    debug_assert_eq!(borrow, 0, "fadd cancel final borrow != 0");

    // ── Left-shift normalize: norm = raw_diff << lshift ────────────────
    let raw_diff_val: u64 = (0..6).fold(0u64, |acc, i| {
        acc | ((row.cancel_diff[i].as_canonical_u32() as u64) << (i * 8))
    });
    let lshift = if raw_diff_val == 0 {
        0
    } else {
        raw_diff_val.leading_zeros() - 16
    }; // 48-bit CLZ
       // leading_zeros gives 64-bit CLZ; raw_diff < 2^48 so CLZ_48 = 64-bit CLZ - 16.
    row.cancel_lshift = F::from_u32(lshift);
    let raw_bytes6: [u32; 6] = core::array::from_fn(|i| ((raw_diff_val >> (i * 8)) & 0xFF) as u32);
    row.cancel_shln = make_shln_cols6::<F>(raw_bytes6, lshift as usize, range_checker);

    // ── RNE rounding (6-byte norm layout) ──────────────────────────────
    let norm = raw_diff_val << lshift;
    let ns0 = ((norm >> 16) & 0xFF) as u32; // byte 2
    let ns3 = ((norm >> 24) & 0xFF) as u32; // byte 3
    let guard = (ns0 >> 7) & 1;
    let ns0_lo = ns0 & 0x7F;
    let mant_lsb = ns3 & 1;
    let ns1_high = ns3 >> 1;
    let below_guard = ns0_lo + (norm & 0xFF) as u32 + ((norm >> 8) & 0xFF) as u32; // ns0_lo + byte0 + byte1
    let sticky = below_guard != 0;
    let round_up = guard == 1 && (sticky || mant_lsb == 1);

    row.guard = F::from_u32(guard);
    row.ns0_lo = F::from_u32(ns0_lo);
    row.ns1_high = F::from_u32(ns1_high);
    row.mant_lsb = F::from_u32(mant_lsb);
    row.below_guard = F::from_u32(below_guard);
    row.below_guard_inv = if below_guard != 0 {
        F::from_u32(below_guard).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.sticky = F::from_bool(sticky);
    row.round_up = F::from_bool(round_up);
    range_checker.add_count(ns0_lo, 7);
    range_checker.add_count(ns1_high, 7);
    for i in 0..6 {
        range_checker.add_count(((norm >> (i * 8)) & 0xFF) as u32, 8);
    }

    // ── Output reconstruction ───────────────────────────────────────────
    let nb3 = ((norm >> 24) & 0xFF) as u32;
    let nb4 = ((norm >> 32) & 0xFF) as u32;
    let nb5 = ((norm >> 40) & 0xFF) as u32;
    let s0 = nb3 + round_up as u32;
    let rc0 = s0 >> 8;
    let s1 = nb4 + rc0;
    let rc1 = s1 >> 8;
    let top7c = (nb5 - 128) + rc1;
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    let result_exp_calc = large_exp.wrapping_sub(lshift).wrapping_add(rc2);

    row.carry_out = F::ZERO;
    row.sum_c1 = F::ZERO;
    row.sum_c2 = F::ZERO;
    for i in 0..WORK_LIMBS {
        row.raw_sum[i] = F::ZERO;
        row.rs_high7[i] = F::ZERO;
        row.rs_lowbit[i] = F::ZERO;
    }
    row.rc0 = F::from_u32(rc0);
    row.rc1 = F::from_u32(rc1);
    row.rc2 = F::from_u32(rc2);
    row.rc2_rem = F::from_u32(rc2_rem);
    row.r_exp_lo = F::from_u32(r_exp & 1);
    row.r_exp_hi = F::from_u32(r_exp >> 1);
    row.r_exp_hi_zero_inv = if r_exp >> 1 != 0 {
        F::from_u32(r_exp >> 1).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.r_exp_hi_max_inv = if r_exp >> 1 != 127 {
        (F::from_u32(127) - F::from_u32(r_exp >> 1))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.result_exp = F::from_u32(r_exp);
    row.result_sign = F::from_u32(large_sign);
    range_checker.add_count(rc2_rem, 7);
    row.s_cancel_normal = F::from_bool((1u32..=254).contains(&r_exp));

    if (1u32..=254).contains(&r_exp) {
        debug_assert_eq!(
            result_exp_calc, r_exp,
            "fadd cancel exp: a={:08x} b={:08x}",
            a_bits, b_bits
        );
        let recon0 = (nb3 + round_up as u32) & 0xFF;
        let recon1 = (nb4 + rc0) & 0xFF;
        let recon2 = rc2_rem + (r_exp & 1) * 128;
        debug_assert_eq!(recon0, result_bits & 0xFF, "fadd cancel byte0");
        debug_assert_eq!(recon1, (result_bits >> 8) & 0xFF);
        debug_assert_eq!(recon2, (result_bits >> 16) & 0xFF);
    }
}

/// Build a `ShrnCols<6>` witness for ShrnGadget and register range-checker counts.
fn make_shrn_cols6<F: PrimeField32>(
    x: [u32; 6],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShrnCols<F, 6> {
    const N: usize = 6;
    let shift = shift.min(N * SHLIMB_BITS - 1);
    let (y, q, r) = run_shrn(&x, shift);
    let (guard, sticky) = run_shrn_round(&x, shift);
    let mut limb_shift_marker = [F::ZERO; N];
    let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
    limb_shift_marker[q] = F::ONE;
    bit_shift_marker[r] = F::ONE;
    let pc = x[q] & ((1u32 << r) - 1);
    let pmsb = if r >= 1 { (pc >> (r - 1)) & 1 } else { 0 };
    let prest = if r >= 1 {
        pc - pmsb * (1u32 << (r - 1))
    } else {
        0
    };
    let hf = if q >= 1 { x[q - 1] } else { 0 };
    let hmsb = if r == 0 { (hf >> 7) & 1 } else { 0 };
    let hrest = if r == 0 { hf - hmsb * 128 } else { 0 };
    let full_rge1: u32 = (0..q).map(|k| x[k]).sum();
    let full_r0: u32 = (0..q.saturating_sub(1)).map(|k| x[k]).sum();
    let lost_combined = if r >= 1 {
        prest + full_rge1
    } else {
        hrest + full_r0
    };
    let combined_inv = if lost_combined != 0 {
        F::from_u32(lost_combined).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    for i in 0..N {
        range_checker.add_count(x[i] & ((1u32 << r) - 1), r);
        range_checker.add_count(y[i], 8);
    }
    if r >= 1 {
        range_checker.add_count(prest, r - 1);
    } else {
        range_checker.add_count(hrest, 7);
    }
    ShrnCols {
        x: core::array::from_fn(|i| F::from_u32(x[i])),
        shift: F::from_u32(shift as u32),
        enable: F::ONE,
        limb_shift_marker,
        bit_shift_marker,
        bit_mult: F::from_u32(1u32 << r),
        bit_shift_carry: core::array::from_fn(|i| F::from_u32(x[i] & ((1u32 << r) - 1))),
        y: core::array::from_fn(|i| F::from_u32(y[i])),
        guard: F::from_u32(guard),
        sticky: F::from_bool(sticky),
        pmsb: F::from_u32(pmsb),
        prest: F::from_u32(prest),
        hmsb: F::from_u32(hmsb),
        hrest: F::from_u32(hrest),
        lost_combined: F::from_u32(lost_combined),
        combined_inv,
    }
}

/// Build a `ShlnCols<6>` witness for ShlnGadget and register range-checker counts.
fn make_shln_cols6<F: PrimeField32>(
    x: [u32; 6],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShlnCols<F, 6> {
    const N: usize = 6;
    let shift = shift.min(N * SHLIMB_BITS);
    let (y, q, r) = run_shln(&x, shift);
    let mut limb_shift_marker = [F::ZERO; N];
    let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
    limb_shift_marker[q] = F::ONE;
    bit_shift_marker[r] = F::ONE;
    let carry_val = |v: u32| if r == 0 { 0 } else { v >> (SHLIMB_BITS - r) };
    for i in 0..N {
        range_checker.add_count(carry_val(x[i]), r);
        range_checker.add_count(y[i], 8);
    }
    ShlnCols {
        x: core::array::from_fn(|i| F::from_u32(x[i])),
        shift: F::from_u32(shift as u32),
        enable: F::ONE,
        limb_shift_marker,
        bit_shift_marker,
        bit_mult: F::from_u32(1u32 << r),
        bit_shift_carry: core::array::from_fn(|i| F::from_u32(carry_val(x[i]))),
        y: core::array::from_fn(|i| F::from_u32(y[i])),
    }
}

#[cfg(test)]
mod special_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    type F = BabyBear;

    fn classify(bits: u32) -> (bool, bool, bool) {
        let exp = (bits >> 23) & 0xFF;
        let mant = bits & 0x7FFFFF;
        (
            exp == 255 && mant != 0,
            exp == 255 && mant == 0,
            exp == 0 && mant == 0,
        )
    }

    fn make_cols(a: u32, b: u32) -> FaddCoreCols<F> {
        let (a_nan, a_inf, a_zero) = classify(a);
        let (b_nan, b_inf, b_zero) = classify(b);
        let result = fadd_s(a, b);
        let is_nan_res = a_nan || b_nan || (a_inf && b_inf && ((a >> 31) & 1) != ((b >> 31) & 1));
        let inf_in = a_inf || b_inf;
        let s_inf = !is_nan_res && inf_in;
        let s_zero_both = !is_nan_res && !inf_in && a_zero && b_zero;

        FaddCoreCols {
            a_sign: F::from_u32((a >> 31) & 1),
            b_sign: F::from_u32((b >> 31) & 1),
            a_is_nan: F::from_bool(a_nan),
            b_is_nan: F::from_bool(b_nan),
            a_is_inf: F::from_bool(a_inf),
            b_is_inf: F::from_bool(b_inf),
            a_is_zero: F::from_bool(a_zero),
            b_is_zero: F::from_bool(b_zero),
            inf_in: F::from_bool(a_inf || b_inf),
            s_nan: F::from_bool(is_nan_res),
            s_inf: F::from_bool(s_inf),
            s_zero_both: F::from_bool(s_zero_both),
            is_nan_result: F::from_bool(is_nan_res),
            result: core::array::from_fn(|i| F::from_u32((result >> (i * 8)) & 0xFF)),
            ..Default::default()
        }
    }

    fn run(c: &FaddCoreCols<F>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        FaddCoreAir::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerBus::new(0, 16),
            0,
        )
        .eval_special_cases(&mut b, c);
        b
    }

    #[test]
    fn fadd_special_honest_passes() {
        let inputs: &[u32] = &[
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            0.0f32.to_bits(),
            (-0.0f32).to_bits(),
            1.0f32.to_bits(),
            (-1.0f32).to_bits(),
            f32::MAX.to_bits(),
        ];
        for &a in inputs {
            for &bb in inputs {
                let c = make_cols(a, bb);
                let b = run(&c);
                assert!(b.ok(), "a={:08x} b={:08x}: {:?}", a, bb, b.failures);
            }
        }
    }

    #[test]
    fn fadd_corrupt_nan_rejected() {
        let mut c = make_cols(f32::NAN.to_bits(), 1.0f32.to_bits());
        c.result[0] = F::from_u32(0xFF);
        assert!(!run(&c).ok());
    }

    #[test]
    fn fadd_corrupt_inf_rejected() {
        let mut c = make_cols(f32::INFINITY.to_bits(), 3.0f32.to_bits());
        c.result[2] = F::ZERO; // inf has result[2] = 128
        assert!(!run(&c).ok());
    }

    #[test]
    fn fadd_corrupt_zero_sign_rejected() {
        // (-0) + (-0) = -0; flipping the sign bit must fail.
        let mut c = make_cols((-0.0f32).to_bits(), (-0.0f32).to_bits());
        c.result[3] = F::ZERO; // should be 0x80
        assert!(!run(&c).ok());
    }

    #[test]
    fn fadd_finite_unconstrained() {
        let mut c = make_cols(3.0f32.to_bits(), 4.0f32.to_bits());
        c.result[0] = F::from_u32(0xAB);
        assert!(run(&c).ok(), "finite case must not be constrained here");
    }
}

#[cfg(test)]
mod finite_add_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn rc() -> SharedVariableRangeCheckerChip {
        Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )))
    }

    fn air() -> FaddCoreAir {
        FaddCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0)
    }

    /// Build a full witness row via the production filler, then run eval_finite_add.
    fn run(a: u32, b: u32) -> (NumericBuilder<F>, FaddCoreCols<F>) {
        run_with(a, b, &rc())
    }

    fn run_with(
        a: u32,
        b: u32,
        chip: &SharedVariableRangeCheckerChip,
    ) -> (NumericBuilder<F>, FaddCoreCols<F>) {
        let mut row = FaddCoreCols::<F>::default();
        let result = fadd_s(a, b);
        fill_fadd_witness(&mut row, a, b, result, chip);
        let mut builder = NumericBuilder::new(vec![]);
        air().eval_finite_add(&mut builder, &row);
        air().eval_finite_cancel(&mut builder, &row);
        (builder, row)
    }

    fn run_row(row: &FaddCoreCols<F>) -> NumericBuilder<F> {
        let mut builder = NumericBuilder::new(vec![]);
        air().eval_finite_add(&mut builder, row);
        builder
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        // Same-sign (matching the caller's sign), both-normal f32 bits.
        fn normal_same_sign(&mut self, sign: u32) -> u32 {
            let exp = (self.next() as u32 % 254) + 1;
            let mant = (self.next() as u32) & 0x7FFFFF;
            (sign << 31) | (exp << 23) | mant
        }
    }

    #[test]
    fn finite_add_honest_passes() {
        let chip = rc();
        let mut rng = Rng(0xA15A_5F1D_C0DE_BEEF);
        let mut carry_seen = false;
        let mut round_up_seen = false;
        let mut deep_seen = false;
        let mut overflow_seen = false;
        let mut trials = 0;
        while trials < 40000 {
            let sign = ((rng.next() as u32) & 1) << 31;
            let a = rng.normal_same_sign(sign);
            let b = rng.normal_same_sign(sign);
            trials += 1;
            let (builder, row) = run_with(a, b, &chip);
            assert!(
                builder.ok(),
                "honest finite-add failed: {:?}\n a={:08x} b={:08x}",
                builder.failures,
                a,
                b
            );
            if row.carry_out == F::ONE {
                carry_seen = true;
            }
            if row.round_up == F::ONE {
                round_up_seen = true;
            }
            if row.is_deep == F::ONE {
                deep_seen = true;
            }
            let r_exp = (fadd_s(a, b) >> 23) & 0xFF;
            if r_exp == 255 {
                overflow_seen = true;
            }
        }
        let _ = trials;
        assert!(carry_seen, "never exercised a carry-out normalization");
        assert!(round_up_seen, "never exercised a round_up=1 case");
        assert!(deep_seen, "never exercised the deep (ediff>=25) path");
        assert!(overflow_seen, "never exercised an overflow->inf result");
    }

    #[test]
    fn finite_add_known_values() {
        // 1.0 + 1.0 = 2.0 (carry-out normalization)
        let (b, _) = run(1.0f32.to_bits(), 1.0f32.to_bits());
        assert!(b.ok(), "{:?}", b.failures);
        // 1.0 + 2.0 = 3.0 (no carry)
        let (b, _) = run(1.0f32.to_bits(), 2.0f32.to_bits());
        assert!(b.ok(), "{:?}", b.failures);
        // 1.0 + (tiny, ediff=25) = 1.0 (deep path)
        let tiny = f32::from_bits(1 << 0); // smallest subnormal is not normal; use exp=102
        let large = 1.0f32;
        // build a normal small operand 25 exp below 1.0 (exp 127): exp 102
        let small = f32::from_bits((102u32 << 23) | 1).to_bits();
        let _ = tiny;
        let (b, _) = run(large.to_bits(), small);
        assert!(b.ok(), "deep path: {:?}", b.failures);
        // MAX + MAX = +inf (overflow)
        let (b, _) = run(f32::MAX.to_bits(), f32::MAX.to_bits());
        assert!(b.ok(), "overflow: {:?}", b.failures);
    }

    #[test]
    fn finite_add_corrupt_result_rejected() {
        let (_, mut row) = run(1.5f32.to_bits(), 2.5f32.to_bits());
        row.result[0] = F::from_u32(row.result[0].as_canonical_u32() ^ 1);
        assert!(!run_row(&row).ok(), "corrupted mantissa must be rejected");
    }

    #[test]
    fn finite_add_corrupt_exponent_rejected() {
        let (_, mut row) = run(1.5f32.to_bits(), 2.5f32.to_bits());
        // Flip the result exponent low bit without fixing r_exp_lo.
        row.result_exp = F::from_u32(row.result_exp.as_canonical_u32() ^ 1);
        assert!(!run_row(&row).ok(), "corrupted exponent must be rejected");
    }

    #[test]
    fn finite_add_corrupt_round_up_rejected() {
        let (_, mut row) = run(1.0f32.to_bits(), f32::from_bits(0x3f800001).to_bits());
        // Force an incorrect rounding decision.
        let flipped = F::from_u32(1 - row.round_up.as_canonical_u32());
        row.round_up = flipped;
        // This may or may not flip the result enough to fail depending on the
        // value; corrupt the result to match the flipped round_up so the only
        // inconsistency is round_up vs guard/sticky.
        // Simpler robust check: flip guard -> round_up relation breaks.
        row.guard = F::from_u32(1 - row.guard.as_canonical_u32());
        assert!(!run_row(&row).ok(), "corrupted guard must be rejected");
    }

    #[test]
    fn finite_add_corrupt_carry_out_rejected() {
        let (_, mut row) = run(1.0f32.to_bits(), 1.0f32.to_bits()); // forces carry_out = 1
        assert_eq!(row.carry_out, F::ONE);
        row.carry_out = F::ZERO; // wrongly claim no carry
        assert!(!run_row(&row).ok(), "wrong carry_out must be rejected");
    }

    #[test]
    fn finite_add_special_inputs_unconstrained() {
        // Special inputs have s_add = 0, so eval_finite_add must not fire any
        // result-tying constraint (and must not spuriously fail).
        for &(a, b) in &[
            (f32::NAN.to_bits(), 1.0f32.to_bits()),
            (f32::INFINITY.to_bits(), 1.0f32.to_bits()),
            (0.0f32.to_bits(), 0.0f32.to_bits()),
            (1.0f32.to_bits(), f32::NAN.to_bits()),
        ] {
            let (builder, _) = run(a, b);
            assert!(
                builder.ok(),
                "special input must not fail finite-add: a={:08x} b={:08x}: {:?}",
                a,
                b,
                builder.failures
            );
        }
    }

    #[test]
    fn finite_add_subtraction_unconstrained() {
        // Opposite-sign (subtraction) finite inputs are NOT covered by this path
        // (s_add = 0), so the constraints must not fire / fail.
        let (builder, _) = run(5.0f32.to_bits(), (-3.0f32).to_bits());
        assert!(builder.ok(), "subtraction must not be constrained here");
    }
}
