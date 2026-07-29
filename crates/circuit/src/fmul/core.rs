use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    utils::not,
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
    pub is_special: T,
    pub is_nan_result: T,
    pub is_inf_result: T,
    pub is_zero_result: T,

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
    pub s_normal: T, // finite inputs AND result exponent field in [1, 254]

    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FmulCoreCols<u8>)]
pub struct FmulCoreAir {
    pub range_tuple_bus: RangeTupleCheckerBus<2>,
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
        builder.assert_bool(cols.is_inf_result);
        builder.assert_bool(cols.is_zero_result);
        builder.assert_bool(cols.is_special);

        // Result sign = a_sign XOR b_sign
        let r_sign = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;
        builder.assert_eq(cols.result_sign, r_sign);

        // ── Phase 3: Significand construction ────────────────────────────
        // Significands are stored as columns (degree 1) to keep the carry chain
        // at degree 2. Verify: a_sig[2] = a_mant2 + is_normal * 128.
        // Gated on is_valid: on padding rows a/b_is_normal evaluate to 1 (all
        // classification columns are zero), so the implicit-bit term must not fire.
        builder.assert_eq(cols.a_sig[0], cols.a_mant0);
        builder.assert_eq(cols.a_sig[1], cols.a_mant1);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_sig[2], cols.a_mant2 + a_is_normal.clone() * f128);
        builder.assert_eq(cols.b_sig[0], cols.b_mant0);
        builder.assert_eq(cols.b_sig[1], cols.b_mant1);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.b_sig[2], cols.b_mant2 + b_is_normal.clone() * f128);

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

            self.range_tuple_bus
                .send(vec![cols.product[i].into(), carry[i].clone()])
                .eval(builder, cols.is_valid);
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

        // Sticky: slack_sum = norm_sig[0] + norm_sig[1] + norm2_lo
        let slack_sum = cols.norm_sig[0] + cols.norm_sig[1] + cols.norm2_lo;
        let slack_is_zero = one.clone() - slack_sum * cols.slack_sum_inv;
        builder.assert_bool(slack_is_zero.clone());
        let sticky = one.clone() - slack_is_zero;

        // LSB: bit 0 of norm_sig[3]
        // norm_sig[3] = lsb + norm3_high * 2
        let lsb = cols.norm_sig[3] - cols.norm3_high * f2;
        builder.assert_eq(cols.norm_sig[3], lsb.clone() + cols.norm3_high * f2);

        // round_up = guard AND (sticky OR lsb)
        let sticky_or_lsb = sticky.clone() + lsb.clone() - sticky * lsb;
        builder.assert_eq(cols.round_up, cols.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Phase 7: Finite normal-result output reconstruction ──────────
        // Ties result bytes / exponent to norm_sig + round_up via a carry chain
        // (rounding carry included). Gated on s_normal.
        self.eval_finite_output(builder, cols);

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
        let exp_normal = a_exp + b_exp - f127 + cols.is_high;

        // Carry / decomposition flags.
        builder.assert_bool(cols.c0);
        builder.assert_bool(cols.c1);
        builder.assert_bool(cols.c2);
        builder.assert_bool(cols.r_exp_lo);
        builder.assert_bool(cols.s_normal);

        // result_exp = r_exp_lo + 2 * r_exp_hi.
        builder.assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);

        // Sound "(r_exp_hi == 0)" and "(r_exp_hi == 127)" indicators via is-zero gadgets.
        let he0 = one.clone() - cols.r_exp_hi * cols.r_exp_hi_zero_inv;
        builder.assert_zero(cols.r_exp_hi * he0.clone());
        let delta127 = AB::Expr::from(f127) - cols.r_exp_hi;
        let he127 = one.clone() - delta127.clone() * cols.r_exp_hi_max_inv;
        builder.assert_zero(delta127 * he127.clone());

        // s_normal = s_finite AND result exp field not in {0, 255}.
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero;
        let exp_is_zero = (one.clone() - cols.r_exp_lo) * he0;
        let exp_is_max = cols.r_exp_lo * he127;
        builder.assert_eq(
            cols.s_normal,
            s_finite * (one.clone() - exp_is_zero - exp_is_max),
        );

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
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
}

impl<A> FmulFiller<A> {
    pub fn new(
        adapter: A,
        range_tuple_chip: SharedRangeTupleCheckerChip<2>,
        offset: usize,
    ) -> Self {
        Self {
            adapter,
            offset,
            range_tuple_chip,
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
            &self.range_tuple_chip,
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
    range_chip: &SharedRangeTupleCheckerChip<2>,
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

    // Product computation
    let a_sig = if a_is_normal {
        a_mant0 | (a_mant1 << 8) | ((a_mant2 | 0x80) << 16)
    } else {
        a_mant0 | (a_mant1 << 8) | (a_mant2 << 16)
    };
    let b_sig = if b_is_normal {
        b_mant0 | (b_mant1 << 8) | ((b_mant2 | 0x80) << 16)
    } else {
        b_mant0 | (b_mant1 << 8) | (b_mant2 << 16)
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
        range_chip.add_count(&[prod_limb, c]);
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
    let slack_sum = F::from_u32((slack & 0xFF) as u32)
        + F::from_u32(((slack >> 8) & 0xFF) as u32)
        + F::from_u32(norm2_lo);
    row.slack_sum_inv = if slack != 0 {
        slack_sum.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.norm3_high = F::from_u32(((norm >> 25) & 0x7F) as u32);
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

    let is_nan_res = a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf);
    let is_inf_res = !is_nan_res && (a_is_inf || b_is_inf || r_exp == 255);
    let is_zero_res = !is_nan_res && (a_is_zero || b_is_zero || (r_exp == 0 && r_mant == 0));
    row.is_nan_result = F::from_bool(is_nan_res);
    row.is_inf_result = F::from_bool(is_inf_res);
    row.is_zero_result = F::from_bool(is_zero_res);
    row.is_special = F::from_bool(is_nan_res || is_inf_res || is_zero_res);

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

    // Finite normal-result output reconstruction witnesses.
    let nb3 = ((norm >> 24) & 0xFF) as u32;
    let nb4 = ((norm >> 32) & 0xFF) as u32;
    let nb5 = ((norm >> 40) & 0xFF) as u32;
    let s0 = nb3 + round_up as u32;
    let c0 = s0 >> 8;
    let s1 = nb4 + c0;
    let c1 = s1 >> 8;
    // For a normal result the normalized product has its implicit bit at 47,
    // so nb5 >= 128 and (nb5 - 128) is the top 7 mantissa bits before rounding.
    let top7 = nb5.wrapping_sub(128);
    let c2 = top7.wrapping_add(c1) >> 7;
    row.c0 = F::from_u32(c0);
    row.c1 = F::from_u32(c1);
    row.c2 = F::from_u32(c2);
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
    let s_normal = s_finite && (1..=254).contains(&r_exp);
    row.s_normal = F::from_bool(s_normal);

    // The reconstruction is only meaningful for normal results; sanity-check
    // the exponent carry there so any witness drift surfaces immediately.
    debug_assert!(
        !s_normal || r_exp == a_exp + b_exp - 127 + is_high as u32 + c2,
        "fmul normal-result exponent reconstruction mismatch"
    );

    row.is_valid = F::ONE;

    let _ = (r_mant, p5_lo);
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
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus::new(0, [1 << 8, 8 << 8]),
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
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus::new(0, [1 << 8, 8 << 8]),
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
