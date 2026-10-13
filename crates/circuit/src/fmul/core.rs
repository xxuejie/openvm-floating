use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    is_less_than::{IsLessThanIo, IsLtSubAir},
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
    TraceSubRowGenerator,
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
    make_shln_cols3, run_shrn, run_shrn_round, ShlnCols, ShlnGadget, ShrnCols, ShrnGadget,
    SHLIMB_BITS,
};

pub const SIG_NUM_LIMBS: usize = 3;
pub const PRODUCT_NUM_LIMBS: usize = 6;
pub const LIMB_BITS: usize = 8;

/// IEEE 754 / RISC-V FFMUL.S: correctly-rounded (RNE) multiplication.
/// Returns the canonical NaN whenever the result is any NaN (NaN inputs, or
/// inf * (-inf)) — the result is deterministic across platforms.
#[inline(always)]
pub fn fmul_s(a: u32, b: u32) -> u32 {
    let r = f32::from_bits(a) * f32::from_bits(b);
    if r.is_nan() {
        CANONICAL_NAN
    } else {
        r.to_bits()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FmulCoreCols<T> {
    // Decomposition of input a
    pub a_sign: T,
    pub a_exp_lo: T,
    pub a_exp_hi: T,
    pub a_mant0: T,
    pub a_mant1: T,
    pub a_mant2: T,

    // Decomposition of input b
    pub b_sign: T,
    pub b_exp_lo: T,
    pub b_exp_hi: T,
    pub b_mant0: T,
    pub b_mant1: T,
    pub b_mant2: T,

    // Classification inverse witnesses
    pub a_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub b_exp_zero_inv: T,
    pub b_exp_max_inv: T,
    pub b_mant_sum_inv: T,

    // Classification flags (stored as columns so downstream result-flag and
    // output constraints stay low degree; mirrors the fdiv/fma pattern).
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

    // Significand construction (stored as columns to keep carry chain degree low)
    pub a_sig: [T; SIG_NUM_LIMBS],
    pub b_sig: [T; SIG_NUM_LIMBS],

    // Subnormal operand pre-normalization (leading-zeros left shift; the
    // exponent absorbs the shift, so the downstream pipeline is unchanged).
    pub a_sub: T,
    pub b_sub: T,
    pub a_lshift: T,
    pub b_lshift: T,
    pub a_shln: ShlnCols<T, 3>,
    pub b_shln: ShlnCols<T, 3>,
    pub a_y2_lo7: T,
    pub b_y2_lo7: T,

    // Product
    pub product: [T; PRODUCT_NUM_LIMBS],

    // Normalization
    pub is_high: T,
    pub p5_lo: T,
    pub norm_sig: [T; PRODUCT_NUM_LIMBS],

    // Rounding
    pub guard: T,
    pub norm2_lo: T,
    pub slack_sum_inv: T,
    pub norm3_high: T,
    pub round_up: T,

    // Result
    pub result_sign: T,
    pub result_exp: T,
    pub result: [T; 4], // output f32 bits (4 limbs) — goes to adapter writes
    pub is_nan_result: T,

    // Special-case selectors (input-derived; see eval_special_cases).
    pub inf_in: T,  // a_is_inf | b_is_inf
    pub zero_in: T, // a_is_zero | b_is_zero
    pub s_nan: T,   // result is canonical NaN (NaN input or 0*inf)
    pub s_inf: T,   // result is +-inf due to an infinite input
    pub s_zero: T,  // result is +-0 due to a zero input

    // Finite normal-result output reconstruction (see eval_finite_output).
    pub c0: T,       // carry out of mantissa byte 0 when adding round_up
    pub c1: T,       // carry out of mantissa byte 1
    pub c2: T,       // carry out of mantissa top 7 bits -> into exponent
    pub r_exp_lo: T, // bit 0 of result exponent field
    pub r_exp_hi: T, // bits [7:1] of result exponent field
    pub r_exp_hi_zero_inv: T,
    pub r_exp_hi_max_inv: T,
    pub s_normal: T, // finite AND not overflow AND not subnormal (LT-routed)

    // Overflow detection (see eval_finite_overflow).
    pub m3_255_inv: T,            // inverse for "norm_sig[3] == 255"
    pub m4_255_inv: T,            // inverse for "norm_sig[4] == 255"
    pub m5_255_inv: T,            // inverse for "norm_sig[5] == 255"
    pub s_overflow: T,            // finite inputs AND exp_normal + c2 >= 255 (result is +-inf)
    pub route_ovf_decomp: [T; 1], // LT aux for s_overflow
    pub route_sub_decomp: [T; 1], // LT aux for s_subnormal

    // === Subnormal / underflow (see eval_finite_subnormal) ===
    pub s_subnormal: T, // finite inputs AND exp_normal <= 0 (subnormal or underflow)
    pub sub_efz: T,     // (result exp field == 0) indicator, gated on is_valid (debug/aux)
    pub sub_capped: T,  // bool: true shift >= 56 (capped to 55 ⇒ deep underflow → 0)
    pub sub_lt_decomp: [T; 1], // lower decomposition for sub_capped = (sum < 97)
    pub sub_shift: T,   // capped shift cshift ∈ [25, 55]
    pub sub_shrn: ShrnCols<T, 7>, // result significand = norm_sig >> cshift (+ guard/sticky)
    pub sub_y0_high7: T, // sub_shrn.y[0] >> 1
    pub sub_lsb: T,     // bit 0 of sub_shrn.y[0]
    pub sub_round_up: T,
    pub sub_rc0: T, // rounding carry out of mantissa byte 0
    pub sub_rc1: T, // rounding carry out of mantissa byte 1

    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FmulCoreCols<u8>)]
pub struct FmulCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FmulCoreAir {
    fn width(&self) -> usize {
        FmulCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FmulCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FmulCoreAir
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
        let cols: &FmulCoreCols<AB::Var> = local_core.borrow();

        builder.assert_bool(cols.is_valid);

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let f256: AB::F = AB::F::from_u32(256);
        let f64: AB::F = AB::F::from_u32(64);

        // ── Phase 1: Decomposition ──────────────────────────────────────
        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let a_mant_sum = cols.a_mant0 + cols.a_mant1 + cols.a_mant2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;
        let b_mant_sum = cols.b_mant0 + cols.b_mant1 + cols.b_mant2;

        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.b_exp_lo);

        // Pin the decomposition: the high pieces must be range-checked, or
        // exp_lo / sign are free field choices (classification re-routing).
        for v in [cols.a_mant2, cols.b_mant2, cols.a_exp_hi, cols.b_exp_hi] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }
        // Low mantissa bytes are adapter-read limbs; range-check them like
        // fdiv's input bytes (memory does not byte-check written data).
        for v in [cols.a_mant0, cols.a_mant1, cols.b_mant0, cols.b_mant1] {
            self.range_bus
                .range_check(v.into(), 8)
                .eval(builder, cols.is_valid.into());
        }
        // Result bytes: without RC, free bool carries (sub_rc0/sub_rc1) on the
        // subnormal path can trade ±256 against result limbs and still satisfy
        // the carry equations (memory does not byte-check writes).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.result[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        // ── Phase 2: Classification (stored as columns, fdiv-style) ──────────
        // The exp_is_zero / exp_is_max / mant_is_zero columns are constrained only
        // on valid rows. On padding rows (is_valid = 0) they are left zero-filled,
        // which makes the unconditional nan/inf/zero products below vanish, so no
        // separate gating is needed there. This keeps every constraint degree <= 3.
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

        // nan / inf / zero derived from the classification columns (unconditional;
        // vanish on padding rows because all factor columns are zero there).
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

        let a_is_normal = (one.clone() - cols.a_exp_is_zero) * (one.clone() - cols.a_exp_is_max);
        let b_is_normal = (one.clone() - cols.b_exp_is_zero) * (one.clone() - cols.b_exp_is_max);

        // Verify result classification flags
        builder.assert_bool(cols.is_nan_result);

        // Result sign = a_sign XOR b_sign
        let r_sign = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;
        builder.assert_eq(cols.result_sign, r_sign);

        // ── Phase 3: Significand construction ────────────────────────────
        // Significands are stored as columns (degree 1) to keep the carry chain
        // at degree 2. Verify: a_sig[2] = a_mant2 + is_normal * 128.
        // Gated on is_valid: on padding rows a/b_is_normal evaluate to 1 (all
        // classification columns are zero), so the implicit-bit term must not fire.
        // Subnormal operands are pre-normalized (leading-zeros left shift); the
        // exponent absorbs the shift (see eval_finite_output etc.).
        builder.assert_eq(
            cols.a_sub,
            cols.a_exp_is_zero * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(
            cols.b_sub,
            cols.b_exp_is_zero * (one.clone() - cols.b_mant_is_zero),
        );
        builder.assert_bool(cols.a_sub);
        builder.assert_bool(cols.b_sub);
        let shln = ShlnGadget::new(self.range_bus);
        let mut eval_op_norm = |x_sub: AB::Var,
                                lshift: AB::Var,
                                m0v: AB::Var,
                                m1v: AB::Var,
                                m2v: AB::Var,
                                shln_cols: &ShlnCols<AB::Var, 3>,
                                y2_lo7: AB::Var| {
            builder.assert_eq(shln_cols.x[0], x_sub * m0v);
            builder.assert_eq(shln_cols.x[1], x_sub * m1v);
            builder.assert_eq(shln_cols.x[2], x_sub * m2v);
            builder.assert_eq(shln_cols.shift, lshift);
            builder.assert_eq(shln_cols.enable, x_sub);
            shln.eval_shln(builder, shln_cols);
            // MSB must land at bit 23 (forces the maximal shift).
            builder.when(x_sub).assert_eq(shln_cols.y[2], y2_lo7 + f128);
            self.range_bus
                .range_check(y2_lo7.into(), 7)
                .eval(builder, x_sub.into());
        };
        eval_op_norm(
            cols.a_sub,
            cols.a_lshift,
            cols.a_mant0,
            cols.a_mant1,
            cols.a_mant2,
            &cols.a_shln,
            cols.a_y2_lo7,
        );
        eval_op_norm(
            cols.b_sub,
            cols.b_lshift,
            cols.b_mant0,
            cols.b_mant1,
            cols.b_mant2,
            &cols.b_shln,
            cols.b_y2_lo7,
        );

        let a_ns = one.clone() - cols.a_sub;
        let b_ns = one.clone() - cols.b_sub;
        builder.assert_eq(
            cols.a_sig[0],
            a_ns.clone() * cols.a_mant0 + cols.a_sub * cols.a_shln.y[0],
        );
        builder.assert_eq(
            cols.a_sig[1],
            a_ns.clone() * cols.a_mant1 + cols.a_sub * cols.a_shln.y[1],
        );
        builder.when(cols.is_valid).assert_eq(
            cols.a_sig[2],
            a_ns * (cols.a_mant2 + a_is_normal.clone() * f128) + cols.a_sub * cols.a_shln.y[2],
        );
        builder.assert_eq(
            cols.b_sig[0],
            b_ns.clone() * cols.b_mant0 + cols.b_sub * cols.b_shln.y[0],
        );
        builder.assert_eq(
            cols.b_sig[1],
            b_ns.clone() * cols.b_mant1 + cols.b_sub * cols.b_shln.y[1],
        );
        builder.when(cols.is_valid).assert_eq(
            cols.b_sig[2],
            b_ns * (cols.b_mant2 + b_is_normal.clone() * f128) + cols.b_sub * cols.b_shln.y[2],
        );

        let a_sig: [AB::Var; SIG_NUM_LIMBS] = [cols.a_sig[0], cols.a_sig[1], cols.a_sig[2]];
        let b_sig: [AB::Var; SIG_NUM_LIMBS] = [cols.b_sig[0], cols.b_sig[1], cols.b_sig[2]];

        // ── Phase 4: Product carry chain ─────────────────────────────────
        let carry_divide: AB::F = AB::F::from_u32(1 << LIMB_BITS).inverse();
        let mut carry: Vec<AB::Expr> = vec![AB::Expr::ZERO; PRODUCT_NUM_LIMBS];

        for i in 0..PRODUCT_NUM_LIMBS {
            let prev = if i > 0 {
                carry[i - 1].clone()
            } else {
                AB::Expr::ZERO
            };
            let mut expected = prev;
            for k in 0..SIG_NUM_LIMBS {
                if k <= i && (i - k) < SIG_NUM_LIMBS {
                    expected += a_sig[k] * b_sig[i - k];
                }
            }
            carry[i] = (expected - cols.product[i]) * carry_divide;

            self.range_bus
                .range_check(cols.product[i].into(), LIMB_BITS)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(carry[i].clone(), 10)
                .eval(builder, cols.is_valid.into());
        }
        // Final carry must be zero
        builder
            .when(cols.is_valid)
            .assert_zero(carry[PRODUCT_NUM_LIMBS - 1].clone());

        // Sound finite-input selector: normalization / output reconstruction are
        // only meaningful when both inputs are finite and nonzero. (Selectors are
        // constrained by eval_special_cases below; constraint order is irrelevant
        // since they must all hold simultaneously.)
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero;

        // ── Phase 5: Normalization ───────────────────────────────────────
        builder.assert_bool(cols.is_high);
        // product[5] has bit 47 or bit 46 set (for normal×normal).
        // When is_high=1 (bit 47 set): product[5] = p5_lo + 128
        // When is_high=0 (bit 46 set, 47 clear): product[5] = p5_lo + 64
        // Gated on s_finite: zero/inf/NaN-input results don't have the expected
        // leading bit, so these constraints must not fire for them.
        builder
            .when(cols.is_valid)
            .when(s_finite.clone())
            .when(cols.is_high)
            .assert_eq(cols.product[5], cols.p5_lo + f128);
        builder
            .when(cols.is_valid)
            .when(s_finite.clone())
            .when(not(cols.is_high))
            .assert_eq(cols.product[5], cols.p5_lo + f64);
        // p5_lo must be range-checked, or is_high is a free choice.
        self.range_bus
            .range_check(cols.p5_lo.into(), 7)
            .eval(builder, cols.is_valid.into());

        // Verify norm_sig: when is_high, norm_sig = product; when !is_high, norm_sig = product << 1.
        // Gated on s_finite for the same reason.
        let mut shift_carry = AB::Expr::ZERO;
        for i in 0..PRODUCT_NUM_LIMBS {
            // is_high case: norm_sig[i] == product[i]
            builder
                .when(s_finite.clone())
                .when(cols.is_high)
                .assert_eq(cols.norm_sig[i], cols.product[i]);

            // !is_high case: shift left by 1 (carry chain stays degree 1)
            let shifted = cols.product[i] * f2 + shift_carry.clone();
            let new_carry = (shifted.clone() - cols.norm_sig[i]) * carry_divide;
            builder
                .when(s_finite.clone())
                .when(not(cols.is_high))
                .assert_eq(cols.norm_sig[i], shifted - new_carry.clone() * f256);
            // The per-limb carry must be a bit, or the chain is tautological.
            self.range_bus.range_check(new_carry.clone(), 1).eval(
                builder,
                cols.is_valid.into() * s_finite.clone() * not(cols.is_high),
            );
            shift_carry = new_carry;
        }
        // Only enforce shift carry == 0 when !is_high (skipped for is_high rows)
        builder
            .when(cols.is_valid)
            .when(not(cols.is_high))
            .assert_zero(shift_carry);

        // ── Phase 6: RNE rounding ────────────────────────────────────────
        builder.assert_bool(cols.guard);
        // norm_sig[2] = norm2_lo + guard * 128
        builder.assert_eq(cols.norm_sig[2], cols.norm2_lo + cols.guard * f128);
        self.range_bus
            .range_check(cols.norm2_lo.into(), 7)
            .eval(builder, cols.is_valid.into());

        // Sticky: slack_sum = norm_sig[0] + norm_sig[1] + norm2_lo
        let slack_sum = cols.norm_sig[0] + cols.norm_sig[1] + cols.norm2_lo;
        let slack_is_zero = one.clone() - slack_sum.clone() * cols.slack_sum_inv;
        builder.assert_bool(slack_is_zero.clone());
        builder.assert_zero(slack_sum * slack_is_zero.clone());
        let sticky = one.clone() - slack_is_zero;

        // LSB: bit 0 of norm_sig[3]
        // norm_sig[3] = lsb + norm3_high * 2
        let lsb = cols.norm_sig[3] - cols.norm3_high * f2;
        builder.assert_eq(cols.norm_sig[3], lsb.clone() + cols.norm3_high * f2);
        builder.assert_bool(lsb.clone());
        self.range_bus
            .range_check(cols.norm3_high.into(), 7)
            .eval(builder, cols.is_valid.into());

        // round_up = guard AND (sticky OR lsb)
        let sticky_or_lsb = sticky.clone() + lsb.clone() - sticky * lsb;
        builder.assert_eq(cols.round_up, cols.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Phase 7: Finite normal-result output reconstruction ──────────
        // Ties result bytes / exponent to norm_sig + round_up via a carry chain
        // (rounding carry included). Gated on s_normal.
        self.eval_finite_output(builder, cols);
        // Finite overflow (product exponent >= 255 -> +-inf).
        self.eval_finite_overflow(builder, cols);
        // Finite subnormal / underflow (product exponent <= 0 -> subnormal or 0).
        self.eval_finite_subnormal(builder, cols);

        // ── Phase 8: Special-case selectors + result constraints ─────────
        self.eval_special_cases(builder, cols);

        // ── Output ───────────────────────────────────────────────────────
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FMUL_S);

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

impl FmulCoreAir {
    /// Constrain the input-derived special-case selectors and the result bytes
    /// for NaN / +-inf / +-0 results. Fully determined by the inputs,
    /// independent of the finite-result (product/rounding) path.
    pub fn eval_special_cases<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmulCoreCols<AB::Var>,
    ) {
        let one = AB::Expr::ONE;
        let f128: AB::F = AB::F::from_u32(128);
        let f127: AB::F = AB::F::from_u32(127);

        // inf_in = a_is_inf | b_is_inf ; zero_in = a_is_zero | b_is_zero
        builder.assert_eq(
            cols.inf_in,
            cols.a_is_inf + cols.b_is_inf - cols.a_is_inf * cols.b_is_inf,
        );
        builder.assert_eq(
            cols.zero_in,
            cols.a_is_zero + cols.b_is_zero - cols.a_is_zero * cols.b_is_zero,
        );

        // s_nan: any NaN input, or invalid inf*0 / 0*inf. The inf*0 terms are
        // only nonzero when the NaN-input terms are zero (an operand cannot be
        // NaN and inf/zero simultaneously), so the plain sum is a clean OR.
        builder.assert_eq(
            cols.s_nan,
            cols.a_is_nan + cols.b_is_nan - cols.a_is_nan * cols.b_is_nan
                + cols.a_is_inf * cols.b_is_zero
                + cols.a_is_zero * cols.b_is_inf,
        );
        builder.assert_bool(cols.s_nan);

        // s_inf = !s_nan & inf_in ; s_zero = !s_nan & !inf_in & zero_in
        builder.assert_eq(cols.s_inf, (one.clone() - cols.s_nan) * cols.inf_in);
        builder.assert_eq(
            cols.s_zero,
            (one.clone() - cols.s_nan) * (one.clone() - cols.inf_in) * cols.zero_in,
        );
        builder.assert_bool(cols.s_inf);
        builder.assert_bool(cols.s_zero);

        // is_nan_result must equal s_nan (canonical NaN iff the op is invalid).
        builder.assert_eq(cols.is_nan_result, cols.s_nan);

        // Result bytes per case (gated; result_sign = a ^ b is constrained in eval).
        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &cb) in canon.iter().enumerate() {
            builder
                .when(cols.s_nan)
                .assert_eq(cols.result[i], AB::F::from_u32(cb as u32));
        }
        // +-inf: exponent field 255, sign = result_sign.
        builder.when(cols.s_inf).assert_zero(cols.result[0]);
        builder.when(cols.s_inf).assert_zero(cols.result[1]);
        builder.when(cols.s_inf).assert_eq(cols.result[2], f128);
        builder.when(cols.s_inf).assert_eq(
            cols.result[3],
            AB::Expr::from(f127) + cols.result_sign * f128,
        );
        // +-0: exponent field 0, sign = result_sign.
        builder.when(cols.s_zero).assert_zero(cols.result[0]);
        builder.when(cols.s_zero).assert_zero(cols.result[1]);
        builder.when(cols.s_zero).assert_zero(cols.result[2]);
        builder
            .when(cols.s_zero)
            .assert_eq(cols.result[3], cols.result_sign * f128);
    }

    /// Constrain the finite *normal*-result output: tie the result bytes to the
    /// (already-constrained) normalized product `norm_sig`, the RNE `round_up`,
    /// and the exponent `a_exp + b_exp - 127 + is_high`, via a 3-byte carry
    /// chain whose final carry adjusts the exponent. Gated on `s_normal`
    /// (finite inputs and a normal result exponent field in [1, 254]); overflow
    /// and subnormal/underflow results are handled separately.
    pub fn eval_finite_output<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmulCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;
        // Effective exponents: subnormal operands absorb their normalization
        // shift (value = sig * 2^(exp_eff - 127 - 23) in all cases).
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let exp_normal = a_exp_eff + b_exp_eff - f127 + cols.is_high;

        // Carry / decomposition flags.
        builder.assert_bool(cols.c0);
        builder.assert_bool(cols.c1);
        builder.assert_bool(cols.c2);
        builder.assert_bool(cols.r_exp_lo);
        builder.assert_bool(cols.s_normal);

        // result_exp = r_exp_lo + 2 * r_exp_hi.
        builder.assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);

        // Pin r_exp_hi to 7 bits so result[3] = r_exp_hi + sign*128 is a unique
        // byte decomposition (result limbs are separately range-checked).
        self.range_bus
            .range_check(cols.r_exp_hi.into(), 7)
            .eval(builder, cols.s_normal.into());

        // s_normal is LT-routed in eval_finite_overflow (from exp_normal + c2),
        // never from the prover-chosen result exponent field.

        // Output reconstruction (gated on s_normal):
        //   result[0] = norm_sig[3]            + round_up - c0*256
        //   result[1] = norm_sig[4]            + c0       - c1*256
        //   result[2] = (norm_sig[5] - 128)    + c1       + (r_exp_lo - c2)*128
        //   result[3] = r_exp_hi + result_sign*128
        //   result_exp = exp_normal + c2
        builder.when(cols.s_normal).assert_eq(
            cols.result[0],
            cols.norm_sig[3] + cols.round_up - cols.c0 * f256,
        );
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[1], cols.norm_sig[4] + cols.c0 - cols.c1 * f256);
        builder.when(cols.s_normal).assert_eq(
            cols.result[2],
            cols.norm_sig[5] - f128 + cols.c1 + (cols.r_exp_lo - cols.c2) * f128,
        );
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.result_sign * f128);
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result_exp, exp_normal + cols.c2);
    }

    /// Constrain the finite overflow case: when the product exponent (including
    /// the rounding carry) reaches 255, the result is +-inf. The rounding carry
    /// `c2` is recomputed here via an all-ones cascade so it is available outside
    /// the `s_normal` reconstruction path.
    pub fn eval_finite_overflow<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmulCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f127: AB::F = AB::F::from_u32(127);
        let f255: AB::F = AB::F::from_u32(255);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let exp_normal = a_exp_eff + b_exp_eff - f127 + cols.is_high;
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero;

        // eq_i = (norm_sig[i] == 255) via is-zero on (255 - norm_sig[i]).
        // Gated on is_valid: on padding rows the inverse columns are 0.
        let d3 = AB::Expr::from(f255) - cols.norm_sig[3];
        let eq3 = one.clone() - d3.clone() * cols.m3_255_inv;
        builder.when(cols.is_valid).assert_zero(d3 * eq3.clone());
        let d4 = AB::Expr::from(f255) - cols.norm_sig[4];
        let eq4 = one.clone() - d4.clone() * cols.m4_255_inv;
        builder.when(cols.is_valid).assert_zero(d4 * eq4.clone());
        let d5 = AB::Expr::from(f255) - cols.norm_sig[5];
        let eq5 = one.clone() - d5.clone() * cols.m5_255_inv;
        // Without d5*eq5=0, inv=0 forges eq5=1 whenever norm_sig[5]!=255,
        // which lets c2 spuriously bump the exponent (observable wrong result).
        builder.when(cols.is_valid).assert_zero(d5 * eq5.clone());
        // (norm_sig[5] == 255 iff its low 7 mantissa bits are all ones, since bit
        // 47 is the implicit 1.)

        // Cascade carry chain (gated on s_finite): c0 = round_up & eq3, etc.
        builder
            .when(s_finite.clone())
            .assert_eq(cols.c0, cols.round_up * eq3);
        builder
            .when(s_finite.clone())
            .assert_eq(cols.c1, cols.c0 * eq4);
        builder
            .when(s_finite.clone())
            .assert_eq(cols.c2, cols.c1 * eq5);

        // ── Sound result-class routing via exp_normal (never result field) ─
        // exp_normal in roughly [-250, 380]; exp_b = exp_normal + 256 fits 10 bits.
        // s_overflow:  exp_normal + c2 >= 255  <=>  510 < exp_b + c2.
        // s_subnormal: exp_normal <= 0         <=>  exp_b < 257.
        // Previously s_overflow/s_subnormal were derived from the claimed result
        // exponent field, so a prover could write a non-Inf encoding whenever
        // exp_normal+c2 > 255 (or take the subnormal path on a normal product).
        let exp_b = exp_normal.clone() + AB::Expr::from(f256);
        self.range_bus
            .range_check(exp_b.clone(), 10)
            .eval(builder, s_finite.clone());
        let lt10 = IsLtSubAir::new(self.range_bus, 10);
        let ovf_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(510)),
            exp_b.clone() + cols.c2,
            cols.s_overflow.into(),
            s_finite.clone(),
        );
        lt10.eval(builder, (ovf_io, &cols.route_ovf_decomp));
        let sub_io = IsLessThanIo::new(
            exp_b,
            AB::Expr::from(AB::F::from_u32(257)),
            cols.s_subnormal.into(),
            s_finite.clone(),
        );
        lt10.eval(builder, (sub_io, &cols.route_sub_decomp));
        builder.assert_bool(cols.s_overflow);
        builder.assert_bool(cols.s_subnormal);
        builder.assert_eq(
            cols.s_normal,
            s_finite * (one.clone() - cols.s_overflow) * (one.clone() - cols.s_subnormal),
        );
        builder.assert_bool(cols.s_normal);

        // Result is +-inf (exponent 255, zero mantissa, sign = a ^ b).
        builder.when(cols.s_overflow).assert_zero(cols.result[0]);
        builder.when(cols.s_overflow).assert_zero(cols.result[1]);
        builder
            .when(cols.s_overflow)
            .assert_eq(cols.result[2], f128);
        builder.when(cols.s_overflow).assert_eq(
            cols.result[3],
            AB::Expr::from(f127) + cols.result_sign * f128,
        );
        let _ = f255; // kept for eq_* deltas above
    }

    /// Constrain the finite subnormal / underflow result (product exponent ≤ 0).
    ///
    /// The normalized product `norm_sig` (implicit bit at 47) is shifted right by
    /// `cshift = min(25 - exp_normal, 55)` via `ShrnGadget<7>`; the low bits become
    /// the subnormal mantissa, with RNE rounding from the gadget's guard/sticky +
    /// the mantissa LSB. For `exp_normal = -23` (`cshift = 48`) this yields the
    /// smallest subnormal (or 0); for deeper underflow (`cshift ≥ 49`, or capped at
    /// 55) the result is 0. Rounding up to exactly 2^23 produces the smallest
    /// *normal* (exp field 1) via the carry into result byte 2's top bit.
    pub fn eval_finite_subnormal<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmulCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f64: AB::F = AB::F::from_u32(64);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(1 << LIMB_BITS);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;

        // s_subnormal is LT-routed in eval_finite_overflow from exp_normal <= 0.
        // sub_efz still tracks (result exp field == 0) for filler/debug consistency.
        let he0 = one.clone() - cols.r_exp_hi * cols.r_exp_hi_zero_inv;
        let exp_field_zero = (one.clone() - cols.r_exp_lo) * he0;
        builder
            .when(cols.is_valid)
            .assert_eq(cols.sub_efz, exp_field_zero);
        builder.assert_bool(cols.s_subnormal);
        let s = cols.s_subnormal;

        // Effective exponents (subnormal operands absorb their normalization
        // shift). Biased by +64 so the sum stays non-negative for the
        // less-than gadget: sub_capped = (true shift >= 56) <=>
        // (sum_eff < 97) <=> (sum_b < 161). On s_subnormal rows exp_normal
        // <= 0, so sum_b <= 191 (< 2^8).
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let sum = a_exp_eff + b_exp_eff + cols.is_high + AB::Expr::from(f64);
        let lt_air = IsLtSubAir::new(self.range_bus, 8);
        let io = IsLessThanIo::new(
            sum.clone(),
            AB::Expr::from(AB::F::from_u32(161)),
            cols.sub_capped.into(),
            s.into(),
        );
        lt_air.eval(builder, (io, &cols.sub_lt_decomp));
        builder.assert_bool(cols.sub_capped);

        // cshift = !capped ? (216 - sum_b) : 55.  (216 - sum_b = 25 - exp_normal.)
        builder.when(s * (one.clone() - cols.sub_capped)).assert_eq(
            cols.sub_shift,
            AB::Expr::from(AB::F::from_u32(216)) - sum.clone(),
        );
        builder
            .when(s * cols.sub_capped)
            .assert_eq(cols.sub_shift, AB::F::from_u32(55));
        self.range_bus
            .range_check(cols.sub_shift.into(), 6)
            .eval(builder, s.into());

        // ShrnGadget<7>: y = norm_sig >> cshift (norm_sig in the low 48 bits).
        for i in 0..6 {
            builder
                .when(s)
                .assert_eq(cols.sub_shrn.x[i], cols.norm_sig[i]);
        }
        builder.when(s).assert_zero(cols.sub_shrn.x[6]);
        builder
            .when(s)
            .assert_eq(cols.sub_shrn.shift, cols.sub_shift);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.sub_shrn);
        builder.assert_eq(cols.sub_shrn.enable, s);

        // For cshift >= 25, y < 2^24, so bytes [3..7] must be zero.
        for i in 3..7 {
            builder.when(s).assert_zero(cols.sub_shrn.y[i]);
        }

        // RNE rounding (guard/sticky from the gadget; lsb = bit 0 of y[0]).
        builder
            .when(s)
            .assert_eq(cols.sub_shrn.y[0], cols.sub_y0_high7 * f2 + cols.sub_lsb);
        builder.assert_bool(cols.sub_lsb);
        self.range_bus
            .range_check(cols.sub_y0_high7.into(), 7)
            .eval(builder, s.into());
        builder.assert_bool(cols.sub_shrn.guard);
        builder.assert_bool(cols.sub_shrn.sticky);
        let sticky_or_lsb =
            cols.sub_shrn.sticky + cols.sub_lsb - cols.sub_shrn.sticky * cols.sub_lsb;
        builder
            .when(s)
            .assert_eq(cols.sub_round_up, cols.sub_shrn.guard * sticky_or_lsb);
        builder.assert_bool(cols.sub_round_up);

        // Output reconstruction: result mantissa = y[0..2] + (y[2]&0x7F), rounded.
        builder.assert_bool(cols.sub_rc0);
        builder.assert_bool(cols.sub_rc1);
        builder.when(s).assert_eq(
            cols.result[0] + cols.sub_rc0 * f256,
            cols.sub_shrn.y[0] + cols.sub_round_up,
        );
        builder.when(s).assert_eq(
            cols.result[1] + cols.sub_rc1 * f256,
            cols.sub_shrn.y[1] + cols.sub_rc0,
        );
        // result[2] = y[2] + rc1 (∈ [0, 0x80]; the top bit is the exp field's low
        // bit, set iff rounding carries into the smallest normal).
        builder
            .when(s)
            .assert_eq(cols.result[2], cols.sub_shrn.y[2] + cols.sub_rc1);
        builder
            .when(s)
            .assert_eq(cols.result[3], cols.result_sign * f128);
        builder.when(s).assert_eq(cols.r_exp_hi, AB::Expr::ZERO);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FmulCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FmulExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FmulExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FmulCoreRecord),
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
            FpOpcode::FMUL_S
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        // Native f32 multiply = IEEE 754 correctly-rounded (RNE), canonical NaN
        let result = fmul_s(u32::from_le_bytes(a), u32::from_le_bytes(b)).to_le_bytes();

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

pub struct FmulFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
    lt_air: IsLtSubAir,
}

impl<A> FmulFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), 8);
        Self {
            adapter,
            offset,
            range_checker,
            lt_air,
        }
    }
}

impl<A, F> TraceFiller<F> for FmulFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &FmulCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FmulCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let result_bits = fmul_s(a_bits, b_bits);

        fill_fmul_witness(
            core_row,
            a_bits,
            b_bits,
            result_bits,
            &self.range_checker,
            &self.lt_air,
        );
    }
}

fn decompose(bits: u32) -> (u32, u32, u32, u32, u32, u32, u32) {
    // Returns: sign, exp_lo, exp_hi, mant0, mant1, mant2, exp
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    let exp_lo = exp & 1;
    let exp_hi = exp >> 1;
    let mant0 = mant & 0xFF;
    let mant1 = (mant >> 8) & 0xFF;
    let mant2 = (mant >> 16) & 0x7F;
    (sign, exp_lo, exp_hi, mant0, mant1, mant2, exp)
}

fn fill_fmul_witness<F: PrimeField32>(
    row: &mut FmulCoreCols<F>,
    a_bits: u32,
    b_bits: u32,
    result_bits: u32,
    range_chip: &SharedVariableRangeCheckerChip,
    lt_air: &IsLtSubAir,
) {
    let (a_sign, a_exp_lo, a_exp_hi, a_mant0, a_mant1, a_mant2, a_exp) = decompose(a_bits);
    let (b_sign, b_exp_lo, b_exp_hi, b_mant0, b_mant1, b_mant2, b_exp) = decompose(b_bits);

    row.a_sign = F::from_u32(a_sign);
    row.a_exp_lo = F::from_u32(a_exp_lo);
    row.a_exp_hi = F::from_u32(a_exp_hi);
    row.a_mant0 = F::from_u32(a_mant0);
    row.a_mant1 = F::from_u32(a_mant1);
    row.a_mant2 = F::from_u32(a_mant2);

    row.b_sign = F::from_u32(b_sign);
    row.b_exp_lo = F::from_u32(b_exp_lo);
    row.b_exp_hi = F::from_u32(b_exp_hi);
    row.b_mant0 = F::from_u32(b_mant0);
    row.b_mant1 = F::from_u32(b_mant1);
    row.b_mant2 = F::from_u32(b_mant2);

    range_chip.add_count(a_mant2, 7);
    range_chip.add_count(b_mant2, 7);
    range_chip.add_count(a_exp_hi, 7);
    range_chip.add_count(b_exp_hi, 7);
    range_chip.add_count(a_mant0, 8);
    range_chip.add_count(a_mant1, 8);
    range_chip.add_count(b_mant0, 8);
    range_chip.add_count(b_mant1, 8);

    // Classification inverse witnesses
    let a_exp_f = F::from_u32(a_exp);
    let b_exp_f = F::from_u32(b_exp);
    row.a_exp_zero_inv = if a_exp != 0 {
        a_exp_f.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.a_exp_max_inv = if a_exp != 255 {
        (F::from_u32(255) - a_exp_f)
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let a_msum = F::from_u32(a_mant0 + a_mant1 + a_mant2);
    row.a_mant_sum_inv = if a_mant0 + a_mant1 + a_mant2 != 0 {
        a_msum.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.b_exp_zero_inv = if b_exp != 0 {
        b_exp_f.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.b_exp_max_inv = if b_exp != 255 {
        (F::from_u32(255) - b_exp_f)
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let b_msum = F::from_u32(b_mant0 + b_mant1 + b_mant2);
    row.b_mant_sum_inv = if b_mant0 + b_mant1 + b_mant2 != 0 {
        b_msum.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // Classification
    let a_is_nan = a_exp == 255 && (a_mant0 + a_mant1 + a_mant2) != 0;
    let a_is_inf = a_exp == 255 && (a_mant0 + a_mant1 + a_mant2) == 0;
    let a_is_zero = a_exp == 0 && (a_mant0 + a_mant1 + a_mant2) == 0;
    let a_is_normal = (1..=254).contains(&a_exp);
    let b_is_nan = b_exp == 255 && (b_mant0 + b_mant1 + b_mant2) != 0;
    let b_is_inf = b_exp == 255 && (b_mant0 + b_mant1 + b_mant2) == 0;
    let b_is_zero = b_exp == 0 && (b_mant0 + b_mant1 + b_mant2) == 0;
    let b_is_normal = (1..=254).contains(&b_exp);

    // Classification columns (constrained in eval; witness-provided).
    row.a_exp_is_zero = F::from_bool(a_exp == 0);
    row.a_exp_is_max = F::from_bool(a_exp == 255);
    row.a_mant_is_zero = F::from_bool(a_mant0 + a_mant1 + a_mant2 == 0);
    row.b_exp_is_zero = F::from_bool(b_exp == 0);
    row.b_exp_is_max = F::from_bool(b_exp == 255);
    row.b_mant_is_zero = F::from_bool(b_mant0 + b_mant1 + b_mant2 == 0);
    row.a_is_nan = F::from_bool(a_is_nan);
    row.a_is_inf = F::from_bool(a_is_inf);
    row.a_is_zero = F::from_bool(a_is_zero);
    row.b_is_nan = F::from_bool(b_is_nan);
    row.b_is_inf = F::from_bool(b_is_inf);
    row.b_is_zero = F::from_bool(b_is_zero);

    // Product computation. Subnormal operands are pre-normalized
    // (leading-zeros left shift); the exponent absorbs the shift.
    let a_sub = a_exp == 0 && (a_mant0 + a_mant1 + a_mant2) != 0;
    let b_sub = b_exp == 0 && (b_mant0 + b_mant1 + b_mant2) != 0;
    row.a_sub = F::from_bool(a_sub);
    row.b_sub = F::from_bool(b_sub);
    let a_mant = a_mant0 | (a_mant1 << 8) | (a_mant2 << 16);
    let b_mant = b_mant0 | (b_mant1 << 8) | (b_mant2 << 16);
    let (a_sig, a_exp_eff) = if a_sub {
        let lz = 24 - (32 - a_mant.leading_zeros());
        let bytes = [a_mant & 0xFF, (a_mant >> 8) & 0xFF, (a_mant >> 16) & 0xFF];
        row.a_shln = make_shln_cols3::<F>(bytes, lz as usize, range_chip);
        row.a_lshift = F::from_u32(lz);
        let sig = a_mant << lz;
        row.a_y2_lo7 = F::from_u32(((sig >> 16) & 0xFF) - 128);
        range_chip.add_count(((sig >> 16) & 0xFF) - 128, 7);
        (sig, 1 - lz as i32)
    } else {
        let sig = if a_is_normal {
            a_mant | 0x800000
        } else {
            a_mant
        };
        (sig, a_exp as i32)
    };
    let (b_sig, b_exp_eff) = if b_sub {
        let lz = 24 - (32 - b_mant.leading_zeros());
        let bytes = [b_mant & 0xFF, (b_mant >> 8) & 0xFF, (b_mant >> 16) & 0xFF];
        row.b_shln = make_shln_cols3::<F>(bytes, lz as usize, range_chip);
        row.b_lshift = F::from_u32(lz);
        let sig = b_mant << lz;
        row.b_y2_lo7 = F::from_u32(((sig >> 16) & 0xFF) - 128);
        range_chip.add_count(((sig >> 16) & 0xFF) - 128, 7);
        (sig, 1 - lz as i32)
    } else {
        let sig = if b_is_normal {
            b_mant | 0x800000
        } else {
            b_mant
        };
        (sig, b_exp as i32)
    };

    // Fill significand columns
    row.a_sig[0] = F::from_u32(a_sig & 0xFF);
    row.a_sig[1] = F::from_u32((a_sig >> 8) & 0xFF);
    row.a_sig[2] = F::from_u32((a_sig >> 16) & 0xFF);
    row.b_sig[0] = F::from_u32(b_sig & 0xFF);
    row.b_sig[1] = F::from_u32((b_sig >> 8) & 0xFF);
    row.b_sig[2] = F::from_u32((b_sig >> 16) & 0xFF);

    let prod = (a_sig as u64) * (b_sig as u64);

    // Product limbs + range check sends
    let a_limbs: [u32; 3] = [a_sig & 0xFF, (a_sig >> 8) & 0xFF, (a_sig >> 16) & 0xFF];
    let b_limbs: [u32; 3] = [b_sig & 0xFF, (b_sig >> 8) & 0xFF, (b_sig >> 16) & 0xFF];
    let mut carry: u32 = 0;
    for i in 0..PRODUCT_NUM_LIMBS {
        let prod_limb = (prod >> (i * LIMB_BITS)) as u32 & 0xFF;
        row.product[i] = F::from_u32(prod_limb);

        let mut sum = carry;
        for k in 0..SIG_NUM_LIMBS {
            if k <= i && (i - k) < SIG_NUM_LIMBS {
                sum += a_limbs[k] * b_limbs[i - k];
            }
        }
        let c = sum >> LIMB_BITS;
        carry = c;
        range_chip.add_count(prod_limb, LIMB_BITS);
        range_chip.add_count(c, 10);
    }

    // Normalization
    let is_high = (prod >> 47) & 1 == 1;
    row.is_high = F::from_bool(is_high);
    let p5 = (prod >> 40) as u32 & 0xFF;
    // When the product is 0 or doesn't have the expected leading bit
    // (e.g., zero/subnormal inputs), p5 may be less than 64/128.
    // Clamp to 0 to avoid underflow — these rows are classified as
    // special/zero results, so normalization columns are don't-care.
    let p5_lo = if is_high && p5 >= 128 {
        p5 - 128
    } else if !is_high && p5 >= 64 {
        p5 - 64
    } else {
        0
    };
    row.p5_lo = F::from_u32(p5_lo);
    range_chip.add_count(p5_lo, 7);
    let norm = if is_high { prod } else { prod << 1 };
    for i in 0..PRODUCT_NUM_LIMBS {
        row.norm_sig[i] = F::from_u32((norm >> (i * LIMB_BITS)) as u32 & 0xFF);
    }

    // Rounding
    let guard_bit = ((norm >> 23) & 1) as u32;
    let norm2_lo = ((norm >> 16) & 0x7F) as u32;
    let slack = norm & 0xFFFFFF;
    let sticky = (slack & 0x7FFFFF) != 0;
    let mant_lsb = (norm >> 24) & 1;
    let round_up = guard_bit == 1 && (sticky || mant_lsb == 1);

    row.guard = F::from_u32(guard_bit);
    row.norm2_lo = F::from_u32(norm2_lo);
    range_chip.add_count(norm2_lo, 7);
    let slack_sum = F::from_u32((slack & 0xFF) as u32)
        + F::from_u32(((slack >> 8) & 0xFF) as u32)
        + F::from_u32(norm2_lo);
    row.slack_sum_inv = if slack != 0 {
        slack_sum.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.norm3_high = F::from_u32(((norm >> 25) & 0x7F) as u32);
    range_chip.add_count(((norm >> 25) & 0x7F) as u32, 7);
    row.round_up = F::from_bool(round_up);

    // Result decomposition
    let (r_sign, _, _, _, _, _, r_exp) = decompose(result_bits);
    let r_mant = result_bits & 0x7FFFFF;

    row.result_sign = F::from_u32(r_sign);
    row.result_exp = F::from_u32(r_exp);
    row.result[0] = F::from_u32(result_bits & 0xFF);
    row.result[1] = F::from_u32((result_bits >> 8) & 0xFF);
    row.result[2] = F::from_u32((result_bits >> 16) & 0xFF);
    row.result[3] = F::from_u32((result_bits >> 24) & 0xFF);
    for i in 0..4 {
        range_chip.add_count((result_bits >> (i * 8)) & 0xFF, 8);
    }

    let is_nan_res = a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf);
    row.is_nan_result = F::from_bool(is_nan_res);

    // Input-derived special-case selectors (constrained in eval_special_cases).
    let inf_in = a_is_inf || b_is_inf;
    let zero_in = a_is_zero || b_is_zero;
    let s_inf = !is_nan_res && inf_in;
    let s_zero = !is_nan_res && !inf_in && zero_in;
    row.inf_in = F::from_bool(inf_in);
    row.zero_in = F::from_bool(zero_in);
    row.s_nan = F::from_bool(is_nan_res);
    row.s_inf = F::from_bool(s_inf);
    row.s_zero = F::from_bool(s_zero);

    // 1-bit shift carries for the <<1 normalization (finite, !is_high only).
    let s_finite = !is_nan_res && !s_inf && !s_zero;
    if s_finite && !is_high {
        let mut sc: u32 = 0;
        for i in 0..PRODUCT_NUM_LIMBS {
            let plimb = (prod >> (i * LIMB_BITS)) as u32 & 0xFF;
            let nlimb = (norm >> (i * LIMB_BITS)) as u32 & 0xFF;
            let shifted = 2 * plimb + sc;
            sc = (shifted - nlimb) / 256;
            debug_assert!(sc <= 1, "fmul shift carry not a bit");
            range_chip.add_count(sc, 1);
        }
        debug_assert_eq!(sc, 0, "fmul shift final carry nonzero");
    }

    // Finite normal-result output reconstruction witnesses. (c0/c1/c2 are
    // computed after s_finite below — they are only meaningful for finite
    // inputs, where norm has its implicit bit at 47.)
    let nb3 = ((norm >> 24) & 0xFF) as u32;
    let nb4 = ((norm >> 32) & 0xFF) as u32;
    let nb5 = ((norm >> 40) & 0xFF) as u32;
    let r_exp_hi = r_exp >> 1;
    row.r_exp_lo = F::from_u32(r_exp & 1);
    row.r_exp_hi = F::from_u32(r_exp_hi);
    row.r_exp_hi_zero_inv = if r_exp_hi != 0 {
        F::from_u32(r_exp_hi).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.r_exp_hi_max_inv = if r_exp_hi != 127 {
        (F::from_u32(127) - F::from_u32(r_exp_hi))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let s_finite = !is_nan_res && !inf_in && !zero_in;
    // s_normal / s_overflow / s_subnormal assigned below via LT routing.

    // Rounding carries. Only meaningful for finite inputs (norm has its implicit
    // bit at 47, so nb5 >= 128); for special/zero inputs they are don't-care
    // (the cascade is gated on s_finite), so set them 0 to satisfy the
    // unconditional booleanity assertions.
    let (c0, c1, c2) = if s_finite {
        let s0 = nb3 + round_up as u32;
        let c0 = s0 >> 8;
        let s1 = nb4 + c0;
        let c1 = s1 >> 8;
        let top7 = nb5.wrapping_sub(128);
        let c2 = top7.wrapping_add(c1) >> 7;
        (c0, c1, c2)
    } else {
        (0u32, 0u32, 0u32)
    };
    row.c0 = F::from_u32(c0);
    row.c1 = F::from_u32(c1);
    row.c2 = F::from_u32(c2);

    // Overflow witnesses (eval_finite_overflow).
    let nb3 = ((norm >> 24) & 0xFF) as u32;
    let nb4 = ((norm >> 32) & 0xFF) as u32;
    let nb5 = ((norm >> 40) & 0xFF) as u32;
    row.m3_255_inv = if nb3 != 255 {
        (F::from_u32(255) - F::from_u32(nb3))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.m4_255_inv = if nb4 != 255 {
        (F::from_u32(255) - F::from_u32(nb4))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.m5_255_inv = if nb5 != 255 {
        (F::from_u32(255) - F::from_u32(nb5))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    // Sound result-class routing (matches eval_finite_overflow LT gadgets).
    let exp_normal_i = a_exp_eff + b_exp_eff - 127 + is_high as i32;
    let exp_check_i = exp_normal_i + c2 as i32;
    let s_overflow = s_finite && exp_check_i >= 255;
    let s_subnormal = s_finite && exp_normal_i <= 0;
    let s_normal = s_finite && !s_overflow && !s_subnormal;
    row.s_overflow = F::from_bool(s_overflow);
    row.s_subnormal = F::from_bool(s_subnormal);
    row.s_normal = F::from_bool(s_normal);
    row.sub_efz = F::from_bool(r_exp == 0);
    let exp_b = (exp_normal_i + 256) as u32;
    debug_assert!(exp_b < 1024, "fmul exp_b out of 10-bit range: {exp_b}");
    if s_finite {
        range_chip.add_count(exp_b, 10);
        let lt10 = IsLtSubAir::new(range_chip.bus(), 10);
        lt10.generate_subrow(
            (range_chip.as_ref(), 510u32, exp_b + c2),
            (&mut row.route_ovf_decomp, &mut row.s_overflow),
        );
        lt10.generate_subrow(
            (range_chip.as_ref(), exp_b, 257u32),
            (&mut row.route_sub_decomp, &mut row.s_subnormal),
        );
        debug_assert_eq!(row.s_overflow, F::from_bool(s_overflow));
        debug_assert_eq!(row.s_subnormal, F::from_bool(s_subnormal));
    } else {
        row.route_ovf_decomp = [F::ZERO];
        row.route_sub_decomp = [F::ZERO];
    }
    if s_normal {
        range_chip.add_count(r_exp >> 1, 7);
    }

    // The reconstruction is only meaningful for normal results; sanity-check
    // the exponent carry there so any witness drift surfaces immediately.
    debug_assert!(
        !s_normal || r_exp as i32 == exp_check_i,
        "fmul normal-result exponent reconstruction mismatch"
    );
    debug_assert!(
        !s_overflow || exp_check_i >= 255,
        "fmul overflow exponent mismatch"
    );
    if s_subnormal {
        let exp_normal = a_exp_eff + b_exp_eff - 127 + is_high as i32;
        let sum_b = (a_exp_eff + b_exp_eff + is_high as i32 + 64) as u32;
        let true_shift = 25 - exp_normal; // = 216 - sum_b
        let sub_capped = true_shift >= 56;
        row.sub_capped = F::from_bool(sub_capped);
        lt_air.generate_subrow(
            (range_chip.as_ref(), sum_b, 161),
            (&mut row.sub_lt_decomp, &mut row.sub_capped),
        );
        let cshift = if sub_capped { 55u32 } else { true_shift as u32 };
        row.sub_shift = F::from_u32(cshift);
        range_chip.add_count(cshift, 6);
        let x: [u32; 7] = core::array::from_fn(|i| ((norm >> (i * LIMB_BITS)) & 0xFF) as u32);
        row.sub_shrn = make_shrn_cols7::<F>(x, cshift as usize, range_chip);
        let y0 = row.sub_shrn.y[0].as_canonical_u32();
        let y1 = row.sub_shrn.y[1].as_canonical_u32();
        let y2 = row.sub_shrn.y[2].as_canonical_u32();
        row.sub_y0_high7 = F::from_u32(y0 >> 1);
        row.sub_lsb = F::from_u32(y0 & 1);
        range_chip.add_count(y0 >> 1, 7);
        let guard = row.sub_shrn.guard.as_canonical_u32();
        let sticky = row.sub_shrn.sticky.as_canonical_u32() != 0;
        let round_up = guard == 1 && (sticky || (y0 & 1) == 1);
        row.sub_round_up = F::from_bool(round_up);
        let s0 = y0 + round_up as u32;
        let rc0 = s0 >> 8;
        let s1 = y1 + rc0;
        let rc1 = s1 >> 8;
        row.sub_rc0 = F::from_u32(rc0);
        row.sub_rc1 = F::from_u32(rc1);
        debug_assert_eq!(
            (y2 + rc1) & 0xFF,
            result_bits >> 16 & 0xFF,
            "fmul subnormal byte2: a={:08x} b={:08x}",
            a_bits,
            b_bits
        );
        debug_assert_eq!(s0 & 0xFF, result_bits & 0xFF, "fmul subnormal byte0");
        debug_assert_eq!(s1 & 0xFF, result_bits >> 8 & 0xFF, "fmul subnormal byte1");
    } else {
        row.sub_capped = F::ZERO;
        row.sub_shift = F::ZERO;
        row.sub_shrn = ShrnCols::<F, 7>::default();
        row.sub_y0_high7 = F::ZERO;
        row.sub_lsb = F::ZERO;
        row.sub_round_up = F::ZERO;
        row.sub_rc0 = F::ZERO;
        row.sub_rc1 = F::ZERO;
    }

    row.is_valid = F::ONE;

    let _ = (r_mant, p5_lo);
}

/// Build a `ShrnCols<7>` witness and register range-checker counts for ShrnGadget.
fn make_shrn_cols7<F: PrimeField32>(
    x: [u32; 7],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShrnCols<F, 7> {
    const N: usize = 7;
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

    /// Build a column row whose classification / selector / result fields are
    /// consistent with fmul_s(a, b). All other columns are left zero.
    fn make_cols(a: u32, b: u32) -> FmulCoreCols<F> {
        let (a_nan, a_inf, a_zero) = classify(a);
        let (b_nan, b_inf, b_zero) = classify(b);
        let result = fmul_s(a, b);
        let is_nan_res = a_nan || b_nan || (a_inf && b_zero) || (a_zero && b_inf);
        let inf_in = a_inf || b_inf;
        let zero_in = a_zero || b_zero;
        let s_inf = !is_nan_res && inf_in;
        let s_zero = !is_nan_res && !inf_in && zero_in;
        let r_sign = ((a >> 31) & 1) ^ ((b >> 31) & 1);

        let mut c = FmulCoreCols::<F> {
            a_is_nan: F::from_bool(a_nan),
            b_is_nan: F::from_bool(b_nan),
            a_is_inf: F::from_bool(a_inf),
            b_is_inf: F::from_bool(b_inf),
            a_is_zero: F::from_bool(a_zero),
            b_is_zero: F::from_bool(b_zero),
            inf_in: F::from_bool(inf_in),
            zero_in: F::from_bool(zero_in),
            s_nan: F::from_bool(is_nan_res),
            s_inf: F::from_bool(s_inf),
            s_zero: F::from_bool(s_zero),
            is_nan_result: F::from_bool(is_nan_res),
            result_sign: F::from_u32(r_sign),
            ..Default::default()
        };
        for i in 0..4 {
            c.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        c
    }

    fn run(c: &FmulCoreCols<F>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        FmulCoreAir::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerBus::new(0, 16),
            0,
        )
        .eval_special_cases(&mut b, c);
        b
    }

    #[test]
    fn special_cases_honest_witness_passes() {
        let inputs: &[u32] = &[
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            0.0f32.to_bits(),
            (-0.0f32).to_bits(),
            1.0f32.to_bits(),
            1e38f32.to_bits(),
            1e-38f32.to_bits(),
        ];
        for &a in inputs {
            for &bb in inputs {
                let c = make_cols(a, bb);
                let b = run(&c);
                assert!(
                    b.ok(),
                    "honest special-case witness failed: {:?} {:?}",
                    b.failures,
                    a
                );
            }
        }
    }

    #[test]
    fn special_case_corrupt_nan_result_rejected() {
        // NaN input must yield canonical NaN; corrupting a byte must fail.
        let mut c = make_cols(f32::NAN.to_bits(), 1.0f32.to_bits());
        c.result[0] = F::from_u32(0xFF); // canonical NaN low byte is 0x00
        let b = run(&c);
        assert!(!b.ok(), "corrupt NaN result must be rejected");
    }

    #[test]
    fn special_case_corrupt_inf_result_rejected() {
        // inf * finite must yield +-inf; corrupting the exponent byte must fail.
        let mut c = make_cols(f32::INFINITY.to_bits(), 3.0f32.to_bits());
        c.result[2] = F::ZERO; // inf has result[2] = 128
        let b = run(&c);
        assert!(!b.ok(), "corrupt inf result must be rejected");
    }

    #[test]
    fn special_case_corrupt_zero_result_rejected() {
        // finite * 0 must yield +-0; setting a nonzero byte must fail.
        let mut c = make_cols(5.0f32.to_bits(), 0.0f32.to_bits());
        c.result[0] = F::ONE; // +-0 has all low bytes zero
        let b = run(&c);
        assert!(!b.ok(), "corrupt zero result must be rejected");
    }

    #[test]
    fn finite_case_leaves_result_unconstrained() {
        // Both operands normal: no special gate fires, so any result passes.
        let mut c = make_cols(3.0f32.to_bits(), 4.0f32.to_bits());
        c.result[0] = F::from_u32(0xAB);
        let b = run(&c);
        assert!(
            b.ok(),
            "finite case must not be constrained by special-case gates"
        );
    }
}

#[cfg(test)]
mod finite_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    type F = BabyBear;

    fn is_normal(bits: u32) -> bool {
        let e = (bits >> 23) & 0xFF;
        (1..=254).contains(&e)
    }

    /// Build a row whose finite-output fields are consistent with fmul_s(a, b),
    /// assuming both inputs and the result are normal f32 values.
    fn make_finite_cols(a: u32, b: u32) -> FmulCoreCols<F> {
        let a_exp = (a >> 23) & 0xFF;
        let b_exp = (b >> 23) & 0xFF;
        let a_sig = (a & 0x7FFFFF) | 0x800000;
        let b_sig = (b & 0x7FFFFF) | 0x800000;
        let prod = (a_sig as u64) * (b_sig as u64);
        let is_high = (prod >> 47) & 1 == 1;
        let norm = if is_high { prod } else { prod << 1 };
        let guard_bit = ((norm >> 23) & 1) as u32;
        let slack = norm & 0xFFFFFF;
        let sticky = (slack & 0x7FFFFF) != 0;
        let mant_lsb = (norm >> 24) & 1;
        let round_up = guard_bit == 1 && (sticky || mant_lsb == 1);

        let result = fmul_s(a, b);
        let r_exp = (result >> 23) & 0xFF;
        let r_sign = ((a >> 31) & 1) ^ ((b >> 31) & 1);

        let nb5 = ((norm >> 40) & 0xFF) as u32;
        let s0 = ((norm >> 24) & 0xFF) as u32 + round_up as u32;
        let c0 = s0 >> 8;
        let s1 = ((norm >> 32) & 0xFF) as u32 + c0;
        let c1 = s1 >> 8;
        let top7 = nb5.wrapping_sub(128);
        let c2 = top7.wrapping_add(c1) >> 7;
        let r_exp_hi = r_exp >> 1;

        FmulCoreCols {
            is_high: F::from_bool(is_high),
            round_up: F::from_bool(round_up),
            a_exp_lo: F::from_u32(a_exp & 1),
            a_exp_hi: F::from_u32(a_exp >> 1),
            b_exp_lo: F::from_u32(b_exp & 1),
            b_exp_hi: F::from_u32(b_exp >> 1),
            result_sign: F::from_u32(r_sign),
            result_exp: F::from_u32(r_exp),
            norm_sig: core::array::from_fn(|i| F::from_u32((norm >> (i * 8)) as u32 & 0xFF)),
            result: core::array::from_fn(|i| F::from_u32((result >> (i * 8)) & 0xFF)),
            c0: F::from_u32(c0),
            c1: F::from_u32(c1),
            c2: F::from_u32(c2),
            r_exp_lo: F::from_u32(r_exp & 1),
            r_exp_hi: F::from_u32(r_exp_hi),
            r_exp_hi_zero_inv: if r_exp_hi != 0 {
                F::from_u32(r_exp_hi).try_inverse().unwrap_or(F::ZERO)
            } else {
                F::ZERO
            },
            r_exp_hi_max_inv: if r_exp_hi != 127 {
                (F::from_u32(127) - F::from_u32(r_exp_hi))
                    .try_inverse()
                    .unwrap_or(F::ZERO)
            } else {
                F::ZERO
            },
            s_normal: F::ONE,
            ..Default::default()
        }
    }

    fn run(c: &FmulCoreCols<F>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        FmulCoreAir::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerBus::new(0, 16),
            0,
        )
        .eval_finite_output(&mut b, c);
        b
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
        fn normal_f32_bits(&mut self) -> u32 {
            let sign = ((self.next() as u32) & 1) << 31;
            let exp = (self.next() as u32 % 254) + 1;
            let mant = (self.next() as u32) & 0x7FFFFF;
            sign | (exp << 23) | mant
        }
    }

    #[test]
    fn finite_output_honest_witness_passes() {
        let mut rng = Rng(0x0F1A_1FE5_A5A5_A5A5);
        let mut round_up_seen = false;
        let mut trials = 0;
        while trials < 20000 {
            let a = rng.normal_f32_bits();
            let b = rng.normal_f32_bits();
            let result = fmul_s(a, b);
            // Only test the normal-result path here.
            if !is_normal(result) {
                continue;
            }
            trials += 1;
            let c = make_finite_cols(a, b);
            if c.round_up == F::ONE {
                round_up_seen = true;
            }
            let bld = run(&c);
            assert!(
                bld.ok(),
                "honest finite-output failed: {:?}\n a={:08x} b={:08x}",
                bld.failures,
                a,
                b
            );
        }
        assert!(round_up_seen, "fuzz never exercised a round_up=1 case");
    }

    #[test]
    fn finite_output_corrupt_mantissa_rejected() {
        let mut c = make_finite_cols(3.0f32.to_bits(), 4.0f32.to_bits());
        c.result[0] = F::from_u32(c.result[0].as_canonical_u32() ^ 1);
        let b = run(&c);
        assert!(!b.ok(), "corrupted mantissa byte must be rejected");
    }

    #[test]
    fn finite_output_corrupt_exponent_rejected() {
        let mut c = make_finite_cols(3.0f32.to_bits(), 4.0f32.to_bits());
        // Flip result_exp without changing r_exp_lo/r_exp_hi -> decomposition breaks.
        c.result_exp = F::from_u32(c.result_exp.as_canonical_u32() ^ 2);
        let b = run(&c);
        assert!(!b.ok(), "corrupted exponent must be rejected");
    }

    #[test]
    fn finite_output_wrong_rounding_rejected() {
        // Force an incorrect carry -> reconstruction should fail.
        let mut c = make_finite_cols(1.0f32.to_bits(), 1.0f32.to_bits());
        c.c0 = F::ONE;
        let b = run(&c);
        assert!(!b.ok(), "incorrect carry must be rejected");
    }
}

#[cfg(test)]
mod overflow_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn filler_cols(a: u32, b: u32) -> FmulCoreCols<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt = IsLtSubAir::new(chip.bus(), 8);
        let mut cols = FmulCoreCols::<F>::default();
        fill_fmul_witness(&mut cols, a, b, fmul_s(a, b), &chip, &lt);
        cols
    }

    fn run(c: &FmulCoreCols<F>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        FmulCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0).eval_finite_overflow(&mut b, c);
        b
    }

    #[test]
    fn overflow_honest_passes() {
        let cases: &[(u32, u32)] = &[
            (f32::MAX.to_bits(), f32::MAX.to_bits()),
            (f32::MAX.to_bits(), 2.0f32.to_bits()),
            (f32::MIN.to_bits(), f32::MAX.to_bits()),
            (1e30f32.to_bits(), 1e30f32.to_bits()),
        ];
        for &(a, bb) in cases {
            let c = filler_cols(a, bb);
            assert_eq!(
                fmul_s(a, bb) & 0x7FFF_FFFF,
                f32::INFINITY.to_bits() & 0x7FFF_FFFF
            );
            assert_eq!(c.s_overflow, F::ONE, "expected overflow row");
            let bld = run(&c);
            assert!(
                bld.ok(),
                "overflow a={:08x} b={:08x}: {:?}",
                a,
                bb,
                bld.failures
            );
        }
    }

    #[test]
    fn overflow_corrupt_result_rejected() {
        let mut c = filler_cols(f32::MAX.to_bits(), f32::MAX.to_bits());
        c.result[0] = F::ONE;
        assert!(!run(&c).ok(), "non-inf overflow result must be rejected");
    }

    /// Forging eq5=1 with m5_255_inv=0 while norm_sig[5]!=255 must not bump c2.
    #[test]
    fn eq5_inv_zero_forge_rejected() {
        use openvm_circuit::arch::BasicAdapterInterface;
        use openvm_circuit::arch::MinimalInstruction;
        use openvm_stark_backend::p3_field::PrimeField32;
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt = IsLtSubAir::new(chip.bus(), 8);
        // 1.5 * 1.5 = 2.25 — normal, norm_sig[5] is not 255, c1 may be 0.
        // Use a case where c1=1 so forged eq5 would change c2.
        let a = 0x7f7ffffe_u32; // near MAX so rounding cascade can carry
        let b = 0x3fc00000_u32; // 1.5
        let mut cols = FmulCoreCols::<F>::default();
        fill_fmul_witness(&mut cols, a, b, fmul_s(a, b), &chip, &lt);
        let ns5 = cols.norm_sig[5].as_canonical_u32();
        if ns5 == 255 || cols.c1 == F::ZERO {
            // Fall back to any finite normal row and force the forge shape.
            let a = 3.0f32.to_bits();
            let b = 4.0f32.to_bits();
            fill_fmul_witness(&mut cols, a, b, fmul_s(a, b), &chip, &lt);
        }
        let ns5 = cols.norm_sig[5].as_canonical_u32();
        assert_ne!(ns5, 255, "need norm_sig[5]!=255 for inv=0 forge");
        let honest_c2 = cols.c2;
        cols.m5_255_inv = F::ZERO;
        cols.c2 = cols.c1; // forged: c2 = c1 * eq5 with eq5=1
        if cols.c2 == honest_c2 {
            // c1 was 0 so c2 unchanged; still the is-zero gate must fail.
        }
        let mut buf = vec![F::ZERO; FmulCoreCols::<F>::width()];
        *buf.as_mut_slice().borrow_mut() = cols;
        let mut bld = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(
            &FmulCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0),
            &mut bld,
            &buf,
            F::ZERO,
        );
        assert!(
            !bld.ok(),
            "eq5 forge with m5_255_inv=0 must be rejected: {:?}",
            bld.failures
        );
    }

    /// Prover sets a non-255 result exp encoding equal to exp_check (>255).
    /// Before LT routing this satisfied gates with a finite garbage write.
    #[test]
    fn overflow_cannot_forge_finite_via_large_result_exp() {
        use openvm_circuit::arch::BasicAdapterInterface;
        use openvm_circuit::arch::MinimalInstruction;
        use openvm_stark_backend::p3_field::PrimeField32;
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt = IsLtSubAir::new(chip.bus(), 8);
        let a = f32::MAX.to_bits();
        let b = f32::MAX.to_bits();
        let mut cols = FmulCoreCols::<F>::default();
        fill_fmul_witness(&mut cols, a, b, fmul_s(a, b), &chip, &lt);
        assert_eq!(cols.s_overflow, F::ONE);
        // Forge: claim exp_check as a "normal" result exponent (>255).
        let exp_check = cols.result_exp; // may be 255 on Inf row; use a_exp path
                                         // Derive exp_check from inputs: max*max has large product exp.
        let a_exp = ((a >> 23) & 0xFF) as i32;
        let b_exp = ((b >> 23) & 0xFF) as i32;
        let is_high = cols.is_high.as_canonical_u32();
        let c2 = cols.c2.as_canonical_u32();
        let exp_check_i = a_exp + b_exp - 127 + is_high as i32 + c2 as i32;
        assert!(
            exp_check_i >= 255,
            "need overflow exp_check, got {exp_check_i}"
        );
        if exp_check_i == 255 {
            // Exact-255 overflow cannot use the >255 forge; still ensure Inf is forced.
            return;
        }
        let forged_exp = exp_check_i as u32;
        cols.s_overflow = F::ZERO;
        cols.s_subnormal = F::ZERO;
        cols.s_normal = F::ONE;
        cols.result_exp = F::from_u32(forged_exp);
        cols.r_exp_lo = F::from_u32(forged_exp & 1);
        cols.r_exp_hi = F::from_u32(forged_exp >> 1);
        cols.result[0] = F::from_u32(0x12);
        cols.result[1] = F::from_u32(0x34);
        cols.result[2] = F::from_u32((forged_exp & 1) * 128);
        cols.result[3] = F::from_u32(forged_exp >> 1); // result_sign=0
        let mut buf = vec![F::ZERO; FmulCoreCols::<F>::width()];
        *buf.as_mut_slice().borrow_mut() = cols;
        let mut bld = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(
            &FmulCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0),
            &mut bld,
            &buf,
            F::ZERO,
        );
        assert!(
            !bld.ok(),
            "forged finite encoding for overflow must be rejected: {:?}",
            bld.failures
        );
        let _ = exp_check;
    }
}

/// A numeric builder that records the *value* of each failing constraint, to help
/// pinpoint which constraint a full-eval failure comes from.
#[cfg(test)]
mod full_eval_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn filler_cols(a: u32, b: u32) -> (FmulCoreCols<F>, Arc<VariableRangeCheckerChip>) {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt = IsLtSubAir::new(chip.bus(), 8);
        let mut cols = FmulCoreCols::<F>::default();
        fill_fmul_witness(&mut cols, a, b, fmul_s(a, b), &chip, &lt);
        (cols, chip)
    }

    fn air() -> FmulCoreAir {
        FmulCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0)
    }

    /// Result bytes and routing selectors must be pinned on the full eval.
    #[test]
    fn corruption_sweep_pins_result_and_selectors() {
        use crate::test_utils::assert_columns_constrained;
        use openvm_circuit::arch::BasicAdapterInterface;
        use openvm_circuit::arch::MinimalInstruction;
        use openvm_circuit_primitives::StructReflectionHelper;
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        let names = FmulCoreCols::<F>::struct_reflection().unwrap_or_default();
        let eval_full = |b: &mut NumericBuilder<F>, cols: &[F]| {
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), b, cols, F::ZERO);
        };
        let classes: &[(u32, u32, &[&str])] = &[
            (
                3.0f32.to_bits(),
                4.0f32.to_bits(),
                &["round_up", "is_high"][..],
            ),
            (0x7F7FFFFF, 0x40000000, &[]), // MAX * 2 -> inf (overflow)
            (0x7F800000, 0x3F800000, &[]), // inf * 1
            (0x7FC00000, 0x3F800000, &[]), // NaN * 1
            (0x00000001, 0x00000001, &[]), // min subnormal^2 -> +0
            (0x00000001, 0x3F800000, &[]), // subnormal * 1 -> subnormal
        ];
        for &(a, b, selectors) in classes {
            let (cols, _chip) = filler_cols(a, b);
            let mut buf = vec![F::ZERO; FmulCoreCols::<F>::width()];
            *buf.as_mut_slice().borrow_mut() = cols;
            let mut bld = NumericBuilder::new(vec![]);
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), &mut bld, &buf, F::ZERO);
            assert!(
                bld.ok(),
                "honest fmul a={a:08x} b={b:08x}: {:?}",
                bld.failures
            );
            let mut pinned = vec!["result__0", "result__1", "result__2", "result__3"];
            pinned.extend_from_slice(selectors);
            assert_columns_constrained(
                &buf,
                eval_full,
                &names,
                &pinned,
                &format!("fmul a={a:08x} b={b:08x}"),
            );
        }
    }

    #[test]
    fn full_eval_special_inputs_pass() {
        // Run each eval helper on filler-built witness for special inputs, to see
        // which (if any) fails — the manual numeric tests use hand-built cols.
        for &(a, b) in &[
            (f32::NAN.to_bits(), 1.0f32.to_bits()),
            (f32::INFINITY.to_bits(), 2.0f32.to_bits()),
            (0.0f32.to_bits(), 5.0f32.to_bits()),
        ] {
            let (cols, _chip) = filler_cols(a, b);
            let mut b1 = NumericBuilder::new(vec![]);
            air().eval_special_cases(&mut b1, &cols);
            assert!(
                b1.ok(),
                "special a={:08x} b={:08x}: {:?}",
                a,
                b,
                b1.failures
            );
            let mut b2 = NumericBuilder::new(vec![]);
            air().eval_finite_output(&mut b2, &cols);
            assert!(b2.ok(), "output a={:08x} b={:08x}: {:?}", a, b, b2.failures);
            let mut b3 = NumericBuilder::new(vec![]);
            air().eval_finite_overflow(&mut b3, &cols);
            assert!(
                b3.ok(),
                "overflow a={:08x} b={:08x}: {:?}",
                a,
                b,
                b3.failures
            );
            let mut b4 = NumericBuilder::new(vec![]);
            air().eval_finite_subnormal(&mut b4, &cols);
            assert!(
                b4.ok(),
                "subnormal a={:08x} b={:08x}: {:?}",
                a,
                b,
                b4.failures
            );
        }
    }

    /// Joint forge: flip sub_rc0 and compensate result[0] by −256 (and
    /// result[1] by +1 so the carry cascade still holds). Before result limb
    /// RC this is a LogUp-only hole; with result RC the non-byte limb is
    /// rejected. NumericBuilder ignores interactions, so we also assert the
    /// forged limb is outside [0, 255] as documentation of the attack shape,
    /// and reject the incomplete (result[0]-only) forge via the cascade.
    #[test]
    fn joint_sub_rc0_result0_compensation_rejected() {
        use openvm_circuit::arch::BasicAdapterInterface;
        use openvm_circuit::arch::MinimalInstruction;
        use openvm_stark_backend::p3_field::PrimeField32;
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        // subnormal * 1.0 → subnormal result (exercises sub_rc*).
        let a = 0x00000001u32;
        let b = 1.0f32.to_bits();
        let (cols, _chip) = filler_cols(a, b);
        assert_eq!(
            cols.s_subnormal,
            F::ONE,
            "need a subnormal-result row for sub_rc0 forge"
        );
        assert_eq!(
            cols.sub_rc0,
            F::ZERO,
            "need sub_rc0=0 so the +1/−256 trade is available"
        );
        let honest0 = cols.result[0];
        let honest1 = cols.result[1];
        let mut buf = vec![F::ZERO; FmulCoreCols::<F>::width()];
        {
            let c: &mut FmulCoreCols<F> = buf.as_mut_slice().borrow_mut();
            *c = cols;
            // Incomplete forge (breaks cascade on result[1]).
            c.sub_rc0 = F::ONE;
            c.result[0] = honest0 - F::from_u32(256);
            let bad = c.result[0].as_canonical_u32();
            assert!(bad > 255, "expected non-byte canonical limb, got {bad}");
        }
        let mut bld = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), &mut bld, &buf, F::ZERO);
        assert!(
            !bld.ok(),
            "incomplete sub_rc0 forge must fail cascade (got ok; failures={:?})",
            bld.failures
        );
        // Complete forge: also bump result[1] so both carry eqs hold in-field.
        // Gate constraints pass; the soundness fix is result limb LogUp RC
        // (not visible to NumericBuilder). Document the non-byte attack shape.
        {
            let c: &mut FmulCoreCols<F> = buf.as_mut_slice().borrow_mut();
            c.result[1] = honest1 + F::ONE;
            assert!(
                c.result[0].as_canonical_u32() > 255,
                "complete forge must leave a non-byte result[0]"
            );
        }
        let mut bld2 = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), &mut bld2, &buf, F::ZERO);
        assert!(
            bld2.ok(),
            "complete joint forge holds gate constraints (RC is LogUp-only): {:?}",
            bld2.failures
        );
    }

    #[test]
    fn subnormal_fuzz_passes() {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt = IsLtSubAir::new(chip.bus(), 8);
        let air = FmulCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0);
        let mut rng = 0xB0BE_F7EE_0DEF_1234u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        let mut trials = 0;
        let mut sub_seen = false;
        let mut deep_seen = false;
        while trials < 60000 && (!sub_seen || !deep_seen) {
            let ea = (next() % 80) + 1;
            let eb = (next() % 80) + 1;
            let a = (ea << 23) | (next() & 0x7FFFFF);
            let b = (eb << 23) | (next() & 0x7FFFFF);
            let result = fmul_s(a, b);
            trials += 1;
            if (result >> 23) & 0xFF != 0 {
                continue;
            }
            let mut cols = FmulCoreCols::<F>::default();
            fill_fmul_witness(&mut cols, a, b, result, &chip, &lt);
            let mut builder = NumericBuilder::new(vec![]);
            air.eval_finite_subnormal(&mut builder, &cols);
            assert!(
                builder.ok(),
                "subnormal a={:08x} b={:08x}: {:?}",
                a,
                b,
                builder.failures
            );
            if cols.sub_capped == F::ONE {
                deep_seen = true;
            } else {
                sub_seen = true;
            }
        }
        let _ = trials;
        assert!(sub_seen, "never exercised a (non-deep) subnormal result");
        assert!(deep_seen, "never exercised a deep-underflow result");
    }

    #[test]
    fn full_eval_subnormal_inputs_pass() {
        // Subnormal operands are pre-normalized; every result class must be
        // provable (this was the unprovable class before the fix).
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        for &(a, b) in &[
            (0x00000001, 0x3F800000), // min subnormal * 1.0
            (0x00000001, 0x00000001), // min subnormal^2 -> +0
            (0x007FFFFF, 0x3F800000), // max subnormal * 1.0
            (0x00800000, 0x00400000), // min normal * subnormal
            (0x00400000, 0x40000000), // subnormal * 2.0 -> normal
            (0x80400000, 0x40000000), // negative subnormal * 2.0
            (0x00000001, 0x7F7FFFFF), // min subnormal * MAX
        ] {
            let (cols, _chip) = filler_cols(a, b);
            let air = air();
            let mut b0 = NumericBuilder::new(vec![]);
            // SAFETY: FmulCoreCols is #[repr(C)] and contains only F fields.
            let slice: &[F] = unsafe {
                std::slice::from_raw_parts(
                    &cols as *const FmulCoreCols<F> as *const F,
                    FmulCoreCols::<F>::width(),
                )
            };
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b0, slice, F::ZERO);
            assert!(
                b0.ok(),
                "fmul subnormal-input a={a:08x} b={b:08x}: {:?}",
                b0.failures
            );
        }
    }
}
