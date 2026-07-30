use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    is_less_than::{IsLessThanIo, IsLtSubAir},
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

use crate::fp_shift::{run_shrn, run_shrn_round, ShrnCols, ShrnGadget, SHLIMB_BITS};

/// Which fused multiply-add variant an instruction performs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FmaddOp {
    /// (a * b) + c
    Madd,
    /// (a * b) - c
    Msub,
    /// -(a * b) + c
    Nmsub,
    /// -(a * b) - c
    Nmadd,
}

impl FmaddOp {
    /// (negate_product, negate_c)
    pub fn negations(self) -> (bool, bool) {
        match self {
            FmaddOp::Madd => (false, false),
            FmaddOp::Msub => (false, true),
            FmaddOp::Nmsub => (true, false),
            FmaddOp::Nmadd => (true, true),
        }
    }
}

pub(crate) fn opcode_to_fmadd_op(opcode: FpOpcode) -> Option<FmaddOp> {
    match opcode {
        FpOpcode::FMADD_S => Some(FmaddOp::Madd),
        FpOpcode::FMSUB_S => Some(FmaddOp::Msub),
        FpOpcode::FNMSUB_S => Some(FmaddOp::Nmsub),
        FpOpcode::FNMADD_S => Some(FmaddOp::Nmadd),
        _ => None,
    }
}

/// IEEE 754 / RISC-V fused multiply-add family, with a single rounding.
/// Per the RISC-V spec, the canonical NaN is returned when any input is NaN
/// or the operation is invalid (0 * inf, or inf + (-inf)).
#[inline(always)]
pub fn fmadd_result(a: u32, b: u32, c: u32, op: FmaddOp) -> u32 {
    let (fa, fb, fc) = (f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
    if fa.is_nan() || fb.is_nan() || fc.is_nan() {
        return CANONICAL_NAN;
    }
    let (neg_p, neg_c) = op.negations();
    let a_inf = fa.is_infinite();
    let b_inf = fb.is_infinite();
    // 0 * inf is an invalid operation
    if (a_inf && fb == 0.0) || (b_inf && fa == 0.0) {
        return CANONICAL_NAN;
    }
    // inf + (-inf) is an invalid operation
    if (a_inf || b_inf) && fc.is_infinite() {
        let prod_sign = (fa.is_sign_negative() != fb.is_sign_negative()) != neg_p;
        let c_sign = fc.is_sign_negative() != neg_c;
        if prod_sign != c_sign {
            return CANONICAL_NAN;
        }
    }
    let ea = if neg_p { -fa } else { fa };
    let ec = if neg_c { -fc } else { fc };
    ea.mul_add(fb, ec).to_bits()
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FmaCoreCols<T> {
    pub a: [T; 4],
    pub b: [T; 4],
    pub c: [T; 4],

    // === Decomposition of inputs ===
    pub a_sign: T,
    pub b_sign: T,
    pub c_sign: T,
    pub a_exp_lo: T,
    pub b_exp_lo: T,
    pub c_exp_lo: T,
    pub a_mant2: T,
    pub b_mant2: T,
    pub c_mant2: T,
    pub a_hi: T,
    pub b_hi: T,
    pub c_hi: T,

    // === Classification inverse witnesses and flags ===
    pub a_exp_zero_inv: T,
    pub b_exp_zero_inv: T,
    pub c_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub b_exp_max_inv: T,
    pub c_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub b_mant_sum_inv: T,
    pub c_mant_sum_inv: T,
    pub a_exp_is_zero: T,
    pub b_exp_is_zero: T,
    pub c_exp_is_zero: T,
    pub a_exp_is_max: T,
    pub b_exp_is_max: T,
    pub c_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub b_mant_is_zero: T,
    pub c_mant_is_zero: T,
    pub a_nan: T,
    pub b_nan: T,
    pub c_nan: T,
    pub a_inf: T,
    pub b_inf: T,
    pub c_inf: T,
    pub a_zero: T,
    pub b_zero: T,
    pub c_zero: T,

    // === Special-case selection ===
    pub any_nan: T,
    pub nan_sel: T,
    pub prod_nan: T,
    pub prod_inf: T,
    pub prod_sign_eff: T,
    pub c_sign_eff: T,
    pub sign_diff: T,
    pub inf_conflict: T,
    pub canon_sel: T,
    pub is_special: T,
    pub inf_res: T,
    pub r_inf_sign: T,

    pub madd_flag: T,
    pub msub_flag: T,
    pub nmsub_flag: T,
    pub nmadd_flag: T,

    // === Finite path: product + normalization ===
    pub a_normal: T,
    pub b_normal: T,
    pub c_normal: T,
    pub ab_normal: T,
    pub c_finite_ok: T,
    pub s_finite: T,
    pub a_sig: [T; 3],
    pub b_sig: [T; 3],
    pub c_sig: [T; 3],
    pub product: [T; 6],
    pub is_high: T,
    pub p5_lo: T,
    pub norm_prod: [T; 6],

    // === Phase 2: addend alignment ===
    pub ediff: T,
    pub c_is_larger: T,
    pub is_deep: T,
    pub align_lt_decomp: [T; 1],
    pub align: ShrnCols<T, 7>,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FmaCoreCols<u8>)]
pub struct FmaCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FmaCoreAir {
    fn width(&self) -> usize {
        FmaCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FmaCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FmaCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 4]; 3]>,
    I::Writes: From<[[AB::Expr; 4]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FmaCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        for flag in [
            cols.a_sign,
            cols.b_sign,
            cols.c_sign,
            cols.a_exp_lo,
            cols.b_exp_lo,
            cols.c_exp_lo,
            cols.a_exp_is_zero,
            cols.b_exp_is_zero,
            cols.c_exp_is_zero,
            cols.a_exp_is_max,
            cols.b_exp_is_max,
            cols.c_exp_is_max,
            cols.a_mant_is_zero,
            cols.b_mant_is_zero,
            cols.c_mant_is_zero,
            cols.a_nan,
            cols.b_nan,
            cols.c_nan,
            cols.a_inf,
            cols.b_inf,
            cols.c_inf,
            cols.a_zero,
            cols.b_zero,
            cols.any_nan,
            cols.nan_sel,
            cols.prod_nan,
            cols.prod_inf,
            cols.prod_sign_eff,
            cols.c_sign_eff,
            cols.sign_diff,
            cols.inf_conflict,
            cols.canon_sel,
            cols.is_special,
            cols.inf_res,
            cols.r_inf_sign,
            cols.madd_flag,
            cols.msub_flag,
            cols.nmsub_flag,
            cols.nmadd_flag,
        ] {
            builder.assert_bool(flag);
        }

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.madd_flag + cols.msub_flag + cols.nmsub_flag + cols.nmadd_flag);

        // ── Byte decomposition ───────────────────────────────────────────
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        builder.assert_eq(cols.b[2], cols.b_mant2 + cols.b_exp_lo * f128);
        builder.assert_eq(cols.b[3], cols.b_hi + cols.b_sign * f128);
        builder.assert_eq(cols.c[2], cols.c_mant2 + cols.c_exp_lo * f128);
        builder.assert_eq(cols.c[3], cols.c_hi + cols.c_sign * f128);

        // Range check all input bytes (bounds exponent/mantissa sums).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.b[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.c[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        // ── Classification (per operand: nan / inf / zero) ───────────────
        let mut classify = |exp_lo: AB::Var,
                            hi: AB::Var,
                            mant0: AB::Var,
                            mant1: AB::Var,
                            mant2: AB::Var,
                            exp_zero_inv: AB::Var,
                            exp_max_inv: AB::Var,
                            mant_sum_inv: AB::Var,
                            exp_is_zero: AB::Var,
                            exp_is_max: AB::Var,
                            mant_is_zero: AB::Var,
                            nan: AB::Var,
                            inf: AB::Var,
                            zero: Option<AB::Var>| {
            let exp = exp_lo + hi * f2;
            builder
                .when(cols.is_valid)
                .assert_eq(exp_is_zero, one.clone() - exp.clone() * exp_zero_inv);
            builder.assert_zero(exp.clone() * exp_is_zero);
            let exp_max_delta = AB::Expr::from(f255) - exp;
            builder.when(cols.is_valid).assert_eq(
                exp_is_max,
                one.clone() - exp_max_delta.clone() * exp_max_inv,
            );
            builder.assert_zero(exp_max_delta * exp_is_max);
            // exp cannot be both 0 and 255
            builder.assert_zero(exp_is_zero * exp_is_max);
            let mant_sum = mant0 + mant1 + mant2;
            builder
                .when(cols.is_valid)
                .assert_eq(mant_is_zero, one.clone() - mant_sum.clone() * mant_sum_inv);
            builder.assert_zero(mant_sum * mant_is_zero);
            builder.assert_eq(nan, exp_is_max * (one.clone() - mant_is_zero));
            builder.assert_eq(inf, exp_is_max * mant_is_zero);
            if let Some(zero) = zero {
                builder.assert_eq(zero, exp_is_zero * mant_is_zero);
            }
        };

        classify(
            cols.a_exp_lo,
            cols.a_hi,
            cols.a[0],
            cols.a[1],
            cols.a_mant2,
            cols.a_exp_zero_inv,
            cols.a_exp_max_inv,
            cols.a_mant_sum_inv,
            cols.a_exp_is_zero,
            cols.a_exp_is_max,
            cols.a_mant_is_zero,
            cols.a_nan,
            cols.a_inf,
            Some(cols.a_zero),
        );
        classify(
            cols.b_exp_lo,
            cols.b_hi,
            cols.b[0],
            cols.b[1],
            cols.b_mant2,
            cols.b_exp_zero_inv,
            cols.b_exp_max_inv,
            cols.b_mant_sum_inv,
            cols.b_exp_is_zero,
            cols.b_exp_is_max,
            cols.b_mant_is_zero,
            cols.b_nan,
            cols.b_inf,
            Some(cols.b_zero),
        );
        classify(
            cols.c_exp_lo,
            cols.c_hi,
            cols.c[0],
            cols.c[1],
            cols.c_mant2,
            cols.c_exp_zero_inv,
            cols.c_exp_max_inv,
            cols.c_mant_sum_inv,
            cols.c_exp_is_zero,
            cols.c_exp_is_max,
            cols.c_mant_is_zero,
            cols.c_nan,
            cols.c_inf,
            None, // c_zero is not needed
        );

        // ── Special-case selection ───────────────────────────────────────
        builder.assert_eq(
            cols.any_nan,
            one.clone()
                - (one.clone() - cols.a_nan)
                    * (one.clone() - cols.b_nan)
                    * (one.clone() - cols.c_nan),
        );
        builder.assert_eq(
            cols.prod_nan,
            cols.a_inf * cols.b_zero + cols.b_inf * cols.a_zero,
        );
        builder.assert_eq(cols.nan_sel, cols.any_nan + cols.prod_nan);
        builder.assert_eq(
            cols.prod_inf,
            cols.a_inf + cols.b_inf - cols.a_inf * cols.b_inf,
        );

        // Effective signs after opcode negations
        let neg_p = cols.nmsub_flag + cols.nmadd_flag;
        let neg_c = cols.msub_flag + cols.nmadd_flag;
        let prod_sign = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;
        builder.assert_eq(
            cols.prod_sign_eff,
            prod_sign.clone() + neg_p.clone() - prod_sign * neg_p * f2,
        );
        builder.assert_eq(
            cols.c_sign_eff,
            cols.c_sign + neg_c.clone() - cols.c_sign * neg_c * f2,
        );
        builder.assert_eq(
            cols.sign_diff,
            cols.prod_sign_eff + cols.c_sign_eff - cols.prod_sign_eff * cols.c_sign_eff * f2,
        );
        builder.assert_eq(
            cols.inf_conflict,
            cols.prod_inf * cols.c_inf * cols.sign_diff,
        );
        builder.assert_eq(cols.canon_sel, cols.nan_sel + cols.inf_conflict);
        builder.assert_eq(
            cols.is_special,
            cols.nan_sel
                + (one.clone() - cols.nan_sel)
                    * (cols.prod_inf + cols.c_inf - cols.prod_inf * cols.c_inf),
        );
        builder.assert_eq(
            cols.inf_res,
            cols.is_special * (one.clone() - cols.canon_sel),
        );
        builder.assert_eq(
            cols.r_inf_sign,
            cols.prod_inf * cols.prod_sign_eff
                + (one.clone() - cols.prod_inf) * cols.c_inf * cols.c_sign_eff,
        );

        // Canonical NaN result for NaN inputs and invalid operations.
        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &canon_byte) in canon.iter().enumerate() {
            builder
                .when(cols.canon_sel)
                .assert_eq(cols.result[i], AB::F::from_u32(canon_byte as u32));
        }
        // +-inf result for infinity cases.
        builder.when(cols.inf_res).assert_zero(cols.result[0]);
        builder.when(cols.inf_res).assert_zero(cols.result[1]);
        builder
            .when(cols.inf_res)
            .assert_eq(cols.result[2], AB::F::from_u32(128));
        builder.when(cols.inf_res).assert_eq(
            cols.result[3],
            AB::Expr::from(AB::F::from_u32(127)) + cols.r_inf_sign * AB::F::from_u32(128),
        );

        // ── Finite path: product + normalization ──────────────────────
        self.eval_finite_product(builder, cols);

        let expected_opcode = cols.madd_flag * AB::F::from_u32(FpOpcode::FMADD_S as u32)
            + cols.msub_flag * AB::F::from_u32(FpOpcode::FMSUB_S as u32)
            + cols.nmsub_flag * AB::F::from_u32(FpOpcode::FNMSUB_S as u32)
            + cols.nmadd_flag * AB::F::from_u32(FpOpcode::FNMADD_S as u32)
            + AB::F::from_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [
                cols.a.map(Into::into),
                cols.b.map(Into::into),
                cols.c.map(Into::into),
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

const SIG_LIMBS: usize = 3;
const PROD_LIMBS: usize = 6;
const LIMB_BITS: u32 = 8;

impl FmaCoreAir {
    /// Phase 1: Verify the product a_sig * b_sig = product (48-bit carry chain)
    /// and normalize to norm_prod with implicit bit at position 47.
    pub fn eval_finite_product<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmaCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f64: AB::F = AB::F::from_u32(64);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        // ── Normal indicators ──────────────────────────────────────────
        let a_ne = one.clone() - cols.a_exp_is_zero;
        let a_nmax = one.clone() - cols.a_exp_is_max;
        let b_ne = one.clone() - cols.b_exp_is_zero;
        let b_nmax = one.clone() - cols.b_exp_is_max;
        let c_ne = one.clone() - cols.c_exp_is_zero;
        let c_nmax = one.clone() - cols.c_exp_is_max;
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_normal, a_ne * a_nmax);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.b_normal, b_ne * b_nmax);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.c_normal, c_ne * c_nmax);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.c_zero, cols.c_exp_is_zero * cols.c_mant_is_zero);

        // s_finite via intermediate columns (degree ≤ 4).
        let not_special = one.clone() - cols.is_special;
        builder
            .when(cols.is_valid)
            .assert_eq(cols.ab_normal, cols.a_normal * cols.b_normal);
        builder.when(cols.is_valid).assert_eq(
            cols.c_finite_ok,
            cols.c_normal + cols.c_zero - cols.c_normal * cols.c_zero,
        );
        builder.when(cols.is_valid).assert_eq(
            cols.s_finite,
            not_special * cols.ab_normal * cols.c_finite_ok,
        );
        builder.assert_bool(cols.s_finite);
        let s = cols.s_finite;

        // ── Significand construction ───────────────────────────────────
        builder.when(s).assert_eq(cols.a_sig[0], cols.a[0]);
        builder.when(s).assert_eq(cols.a_sig[1], cols.a[1]);
        builder
            .when(s)
            .assert_eq(cols.a_sig[2], cols.a_mant2 + f128);
        builder.when(s).assert_eq(cols.b_sig[0], cols.b[0]);
        builder.when(s).assert_eq(cols.b_sig[1], cols.b[1]);
        builder
            .when(s)
            .assert_eq(cols.b_sig[2], cols.b_mant2 + f128);
        builder.when(s).assert_eq(cols.c_sig[0], cols.c[0]);
        builder.when(s).assert_eq(cols.c_sig[1], cols.c[1]);
        builder
            .when(s)
            .assert_eq(cols.c_sig[2], cols.c_mant2 + f128);
        for arr in [&cols.a_sig, &cols.b_sig, &cols.c_sig] {
            for &val in arr {
                self.range_bus
                    .range_check(val.into(), LIMB_BITS as usize)
                    .eval(builder, s.into());
            }
        }

        // ── Product carry chain: a_sig * b_sig = product (6 bytes) ─────
        let carry_divide = AB::F::from_u32(1 << LIMB_BITS).inverse();
        let mut carry = AB::Expr::ZERO;
        for i in 0..PROD_LIMBS {
            let mut expected = carry.clone();
            for k in 0..SIG_LIMBS {
                if k <= i && (i - k) < SIG_LIMBS {
                    expected += cols.a_sig[k] * cols.b_sig[i - k];
                }
            }
            carry = (expected - cols.product[i]) * carry_divide;
            self.range_bus
                .range_check(cols.product[i].into(), LIMB_BITS as usize)
                .eval(builder, s.into());
            self.range_bus
                .range_check(carry.clone(), 10)
                .eval(builder, s.into());
        }
        // Final carry must be zero.
        builder.when(s).assert_zero(carry);

        // ── Normalization: is_high ─────────────────────────────────────
        builder.assert_bool(cols.is_high);
        // product[5] bit 47 (is_high) or bit 46 (!is_high) must be set.
        builder
            .when(s)
            .when(cols.is_high)
            .assert_eq(cols.product[5], cols.p5_lo + f128);
        builder
            .when(s)
            .when(one.clone() - cols.is_high)
            .assert_eq(cols.product[5], cols.p5_lo + f64);
        self.range_bus
            .range_check(cols.p5_lo.into(), 7)
            .eval(builder, s.into());

        // norm_prod = is_high ? product : product << 1.
        let mut shift_carry = AB::Expr::ZERO;
        for i in 0..PROD_LIMBS {
            // is_high: norm_prod[i] = product[i].
            builder
                .when(s)
                .when(cols.is_high)
                .assert_eq(cols.norm_prod[i], cols.product[i]);
            // !is_high: norm_prod = product << 1 (carry chain).
            let shifted = cols.product[i] * f2 + shift_carry.clone();
            let new_carry = (shifted.clone() - cols.norm_prod[i]) * carry_divide;
            builder
                .when(s)
                .when(one.clone() - cols.is_high)
                .assert_eq(cols.norm_prod[i], shifted - new_carry.clone() * f256);
            shift_carry = new_carry;
        }
        builder
            .when(s)
            .when(one.clone() - cols.is_high)
            .assert_zero(shift_carry);

        // Range-check norm_prod bytes.
        for i in 0..PROD_LIMBS {
            self.range_bus
                .range_check(cols.norm_prod[i].into(), LIMB_BITS as usize)
                .eval(builder, s.into());
        }

        // ── Phase 2: Addend alignment ──────────────────────────────────
        self.eval_finite_align(builder, cols);
    }

    /// Phase 2: Align the smaller operand (c or product) into a 7-byte field
    /// using ShrnGadget<7>. Computes ediff = |prod_exp - c_exp|, determines
    /// which operand is larger, and shifts the smaller one right by ediff.
    pub fn eval_finite_align<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmaCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f55: AB::F = AB::F::from_u32(55);
        let f127: AB::F = AB::F::from_u32(127);
        let one = AB::Expr::ONE;
        let s = cols.s_finite;

        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;
        let c_exp = cols.c_exp_lo + cols.c_hi * f2;
        let prod_exp = a_exp + b_exp - AB::Expr::from(f127) + cols.is_high;

        // ── ediff = |prod_exp - c_exp| ─────────────────────────────────
        builder.assert_bool(cols.c_is_larger);
        let diff = prod_exp - c_exp;
        let ediff_expr =
            (one.clone() - cols.c_is_larger) * diff.clone() + cols.c_is_larger * (-diff);
        builder.when(s).assert_eq(cols.ediff, ediff_expr);
        self.range_bus
            .range_check(cols.ediff.into(), 9)
            .eval(builder, s.into());

        // ── Deep alignment: ediff >= 56 ────────────────────────────────
        builder.assert_bool(cols.is_deep);
        let lt_air = IsLtSubAir::new(self.range_bus, 9);
        let io = IsLessThanIo::new(
            AB::Expr::from(f55),
            cols.ediff.into(),
            cols.is_deep.into(),
            s.into(),
        );
        lt_air.eval(builder, (io, &cols.align_lt_decomp));

        // ── ShrnGadget<7>: shift the smaller operand right ─────────────
        // Input mux: c_is_larger=0 → c at [3..6]; c_is_larger=1 → product at [0..6].
        let cil = cols.c_is_larger;
        for i in 0..7 {
            let c_val = match i {
                0..=2 => AB::Expr::ZERO,
                3 => cols.c_sig[0].into(),
                4 => cols.c_sig[1].into(),
                5 => cols.c_sig[2].into(),
                _ => AB::Expr::ZERO,
            };
            let p_val = if i < 6 {
                cols.norm_prod[i].into()
            } else {
                AB::Expr::ZERO
            };
            let ncil = one.clone() - cil;
            builder
                .when(s)
                .assert_eq(cols.align.x[i], ncil * c_val + cil * p_val);
        }
        // Shift: clamp to 55 if deep.
        builder
            .when(s * (one.clone() - cols.is_deep))
            .assert_eq(cols.align.shift, cols.ediff);
        builder
            .when(s * cols.is_deep)
            .assert_eq(cols.align.shift, AB::Expr::from(f55));
        builder.assert_eq(cols.align.enable, s);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.align);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FmaCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
    pub c: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FmaExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FmaExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 3]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FmaCoreRecord),
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

        let [a, b, c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.b = b;
        core_record.c = c;
        core_record.opcode = local as u8;

        let op = opcode_to_fmadd_op(FpOpcode::from_usize(local)).unwrap();
        let result = fmadd_result(
            u32::from_le_bytes(a),
            u32::from_le_bytes(b),
            u32::from_le_bytes(c),
            op,
        );
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

pub struct FmaFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FmaFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FmaFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FmaCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FmaCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let c_bits = u32::from_le_bytes(record.c);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let op = opcode_to_fmadd_op(opcode).unwrap();
        let result = fmadd_result(a_bits, b_bits, c_bits, op);

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            core_row.c[i] = F::from_u32((c_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((b_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((c_bits >> (i * 8)) & 0xFF, 8);
        }

        let (neg_p, neg_c) = op.negations();

        let classify = |bits: u32,
                        sign: &mut F,
                        exp_lo: &mut F,
                        mant2: &mut F,
                        hi: &mut F,
                        exp_zero_inv: &mut F,
                        exp_max_inv: &mut F,
                        mant_sum_inv: &mut F,
                        exp_is_zero: &mut F,
                        exp_is_max: &mut F,
                        mant_is_zero: &mut F,
                        nan: &mut F,
                        inf: &mut F,
                        zero: &mut F| {
            let s = (bits >> 31) & 1;
            let exp = (bits >> 23) & 0xFF;
            let mant = bits & 0x7FFFFF;
            *sign = F::from_u32(s);
            *exp_lo = F::from_u32(exp & 1);
            *mant2 = F::from_u32((mant >> 16) & 0x7F);
            *hi = F::from_u32(exp >> 1);
            let exp_f = F::from_u32(exp);
            *exp_is_zero = F::from_bool(exp == 0);
            *exp_is_max = F::from_bool(exp == 255);
            *exp_zero_inv = if exp != 0 {
                exp_f.try_inverse().unwrap_or(F::ZERO)
            } else {
                F::ZERO
            };
            *exp_max_inv = if exp != 255 {
                (F::from_u32(255) - exp_f).try_inverse().unwrap_or(F::ZERO)
            } else {
                F::ZERO
            };
            let mant_sum = (mant & 0xFF) + ((mant >> 8) & 0xFF) + ((mant >> 16) & 0x7F);
            *mant_is_zero = F::from_bool(mant == 0);
            *mant_sum_inv = if mant != 0 {
                F::from_u32(mant_sum).try_inverse().unwrap_or(F::ZERO)
            } else {
                F::ZERO
            };
            *nan = F::from_bool(exp == 255 && mant != 0);
            *inf = F::from_bool(exp == 255 && mant == 0);
            *zero = F::from_bool(exp == 0 && mant == 0);
        };

        classify(
            a_bits,
            &mut core_row.a_sign,
            &mut core_row.a_exp_lo,
            &mut core_row.a_mant2,
            &mut core_row.a_hi,
            &mut core_row.a_exp_zero_inv,
            &mut core_row.a_exp_max_inv,
            &mut core_row.a_mant_sum_inv,
            &mut core_row.a_exp_is_zero,
            &mut core_row.a_exp_is_max,
            &mut core_row.a_mant_is_zero,
            &mut core_row.a_nan,
            &mut core_row.a_inf,
            &mut core_row.a_zero,
        );
        classify(
            b_bits,
            &mut core_row.b_sign,
            &mut core_row.b_exp_lo,
            &mut core_row.b_mant2,
            &mut core_row.b_hi,
            &mut core_row.b_exp_zero_inv,
            &mut core_row.b_exp_max_inv,
            &mut core_row.b_mant_sum_inv,
            &mut core_row.b_exp_is_zero,
            &mut core_row.b_exp_is_max,
            &mut core_row.b_mant_is_zero,
            &mut core_row.b_nan,
            &mut core_row.b_inf,
            &mut core_row.b_zero,
        );
        let mut dummy = F::ZERO;
        let _ = &mut dummy;
        classify(
            c_bits,
            &mut core_row.c_sign,
            &mut core_row.c_exp_lo,
            &mut core_row.c_mant2,
            &mut core_row.c_hi,
            &mut core_row.c_exp_zero_inv,
            &mut core_row.c_exp_max_inv,
            &mut core_row.c_mant_sum_inv,
            &mut core_row.c_exp_is_zero,
            &mut core_row.c_exp_is_max,
            &mut core_row.c_mant_is_zero,
            &mut core_row.c_nan,
            &mut core_row.c_inf,
            &mut core_row.c_zero,
        );

        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);
        let (c_sign, c_exp, _c_mant) = decompose(c_bits);
        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        let c_nan = c_exp == 255 && _c_mant != 0;
        let a_inf = a_exp == 255 && a_mant == 0;
        let b_inf = b_exp == 255 && b_mant == 0;
        let c_inf = c_exp == 255 && _c_mant == 0;
        let a_zero = a_exp == 0 && a_mant == 0;
        let b_zero = b_exp == 0 && b_mant == 0;

        let any_nan = a_nan || b_nan || c_nan;
        let prod_nan = (a_inf && b_zero) || (b_inf && a_zero);
        let nan_sel = any_nan || prod_nan;
        let prod_inf = a_inf || b_inf;
        let prod_sign_eff = (a_sign != b_sign) != neg_p;
        let c_sign_eff = (c_sign == 1) != neg_c;
        let sign_diff = prod_sign_eff != c_sign_eff;
        let inf_conflict = prod_inf && c_inf && sign_diff;
        let canon_sel = nan_sel || inf_conflict;
        let is_special = nan_sel || prod_inf || c_inf;
        let inf_res = is_special && !canon_sel;
        let r_inf_sign = if prod_inf {
            prod_sign_eff
        } else {
            c_inf && c_sign_eff
        };

        core_row.any_nan = F::from_bool(any_nan);
        core_row.prod_nan = F::from_bool(prod_nan);
        core_row.nan_sel = F::from_bool(nan_sel);
        core_row.prod_inf = F::from_bool(prod_inf);
        core_row.prod_sign_eff = F::from_bool(prod_sign_eff);
        core_row.c_sign_eff = F::from_bool(c_sign_eff);
        core_row.sign_diff = F::from_bool(sign_diff);
        core_row.inf_conflict = F::from_bool(inf_conflict);
        core_row.canon_sel = F::from_bool(canon_sel);
        core_row.is_special = F::from_bool(is_special);
        core_row.inf_res = F::from_bool(inf_res);
        core_row.r_inf_sign = F::from_bool(r_inf_sign);
        core_row.madd_flag = F::from_bool(op == FmaddOp::Madd);
        core_row.msub_flag = F::from_bool(op == FmaddOp::Msub);
        core_row.nmsub_flag = F::from_bool(op == FmaddOp::Nmsub);
        core_row.nmadd_flag = F::from_bool(op == FmaddOp::Nmadd);

        // ── Finite path ────────────────────────────────────────────────
        let a_normal = (1..=254).contains(&a_exp);
        let b_normal = (1..=254).contains(&b_exp);
        let c_normal = (1..=254).contains(&c_exp);
        let c_zero_v = c_exp == 0 && _c_mant == 0;
        let s_finite = !is_special && a_normal && b_normal && (c_normal || c_zero_v);
        core_row.a_normal = F::from_bool(a_normal);
        core_row.b_normal = F::from_bool(b_normal);
        core_row.c_normal = F::from_bool(c_normal);
        core_row.c_zero = F::from_bool(c_zero_v);
        core_row.ab_normal = F::from_bool(a_normal && b_normal);
        core_row.c_finite_ok = F::from_bool(c_normal || c_zero_v);
        core_row.s_finite = F::from_bool(s_finite);

        if s_finite {
            fill_finite_product(
                core_row,
                a_exp,
                b_exp,
                a_mant,
                b_mant,
                c_exp,
                _c_mant,
                &self.range_checker,
            );
        } else {
            zero_finite_columns(core_row);
        }

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

/// Zero all finite-path columns (for non-finite rows).
fn zero_finite_columns<F: PrimeField32>(row: &mut FmaCoreCols<F>) {
    row.s_finite = F::ZERO;
    for i in 0..3 {
        row.a_sig[i] = F::ZERO;
        row.b_sig[i] = F::ZERO;
        row.c_sig[i] = F::ZERO;
    }
    for i in 0..6 {
        row.product[i] = F::ZERO;
        row.norm_prod[i] = F::ZERO;
    }
    row.is_high = F::ZERO;
    row.p5_lo = F::ZERO;
    row.ediff = F::ZERO;
    row.c_is_larger = F::ZERO;
    row.is_deep = F::ZERO;
    row.align_lt_decomp = [F::ZERO];
    row.align = ShrnCols::<F, 7>::default();
}

/// Fill the product + normalization witness for the finite path.
#[allow(clippy::too_many_arguments)]
fn fill_finite_product<F: PrimeField32>(
    row: &mut FmaCoreCols<F>,
    a_exp: u32,
    b_exp: u32,
    a_mant: u32,
    b_mant: u32,
    c_exp: u32,
    c_mant: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let a_sig = a_mant | 0x800000;
    let b_sig = b_mant | 0x800000;

    // Significand bytes.
    let asig_bytes: [u32; 3] = [a_sig & 0xFF, (a_sig >> 8) & 0xFF, (a_sig >> 16) & 0xFF];
    let bsig_bytes: [u32; 3] = [b_sig & 0xFF, (b_sig >> 8) & 0xFF, (b_sig >> 16) & 0xFF];
    for i in 0..3 {
        row.a_sig[i] = F::from_u32(asig_bytes[i]);
        row.b_sig[i] = F::from_u32(bsig_bytes[i]);
        range_checker.add_count(asig_bytes[i], 8);
        range_checker.add_count(bsig_bytes[i], 8);
    }

    // c_sig is set later (Phase 2); for now just range-check a/b sig.

    // 48-bit product.
    let prod = (a_sig as u64) * (b_sig as u64);
    let prod_bytes: [u32; 6] = core::array::from_fn(|i| ((prod >> (i * 8)) & 0xFF) as u32);

    // Carry chain (for range check counts).
    let carry_div: u64 = 1 << 8;
    let mut carry: u64 = 0;
    for i in 0..6 {
        let mut expected = carry;
        for k in 0..3 {
            if k <= i && (i - k) < 3 {
                expected += asig_bytes[k] as u64 * bsig_bytes[i - k] as u64;
            }
        }
        carry = (expected - prod_bytes[i] as u64) / carry_div;
        row.product[i] = F::from_u32(prod_bytes[i]);
        range_checker.add_count(prod_bytes[i], 8);
        range_checker.add_count(carry as u32, 10);
    }
    debug_assert_eq!(carry, 0, "fma product final carry nonzero");

    // Normalization: is_high = bit 47 of product.
    let is_high = (prod >> 47) & 1 == 1;
    row.is_high = F::from_bool(is_high);
    let p5 = prod_bytes[5];
    let p5_lo = if is_high && p5 >= 128 {
        p5 - 128
    } else if !is_high && p5 >= 64 {
        p5 - 64
    } else {
        0
    };
    row.p5_lo = F::from_u32(p5_lo);
    range_checker.add_count(p5_lo, 7);

    // norm_prod = is_high ? product : product << 1.
    let norm = if is_high { prod } else { prod << 1 };
    let norm_bytes: [u32; 6] = core::array::from_fn(|i| ((norm >> (i * 8)) & 0xFF) as u32);
    for (i, &nb) in norm_bytes.iter().enumerate() {
        row.norm_prod[i] = F::from_u32(nb);
        range_checker.add_count(norm_bytes[i], 8);
    }

    // c_sig construction (for Phase 2 alignment).
    let c_sig_val = c_mant | 0x800000;
    let csig_bytes: [u32; 3] = [
        c_sig_val & 0xFF,
        (c_sig_val >> 8) & 0xFF,
        (c_sig_val >> 16) & 0xFF,
    ];
    for (i, &cb) in csig_bytes.iter().enumerate() {
        row.c_sig[i] = F::from_u32(cb);
        range_checker.add_count(csig_bytes[i], 8);
    }

    let _ = (a_exp, b_exp, c_exp);

    // ── Phase 2: Addend alignment ──────────────────────────────────
    let prod_exp_i = a_exp as i32 + b_exp as i32 - 127 + is_high as i32;
    let c_exp_i = c_exp as i32;
    let diff_i = prod_exp_i - c_exp_i;
    let c_is_larger = diff_i < 0;
    let ediff = diff_i.unsigned_abs();
    let is_deep = ediff >= 56;
    let cshift = if is_deep { 55u32 } else { ediff };

    row.ediff = F::from_u32(ediff);
    range_checker.add_count(ediff, 9);
    row.c_is_larger = F::from_bool(c_is_larger);
    row.is_deep = F::from_bool(is_deep);

    let lt_air = IsLtSubAir::new(range_checker.bus(), 9);
    lt_air.generate_subrow(
        (range_checker.as_ref(), 55u32, ediff),
        (&mut row.align_lt_decomp, &mut row.is_deep),
    );

    // ShrnGadget input: smaller operand placed in 7-byte field.
    let align_x: [u32; 7] = if c_is_larger {
        // Product is smaller: [norm_prod[0..5], 0].
        let norm = if is_high { prod } else { prod << 1 };
        core::array::from_fn(|i| {
            if i < 6 {
                ((norm >> (i * 8)) & 0xFF) as u32
            } else {
                0
            }
        })
    } else {
        // C is smaller: [0, 0, 0, c_sig[0], c_sig[1], c_sig[2], 0].
        [0, 0, 0, csig_bytes[0], csig_bytes[1], csig_bytes[2], 0]
    };

    row.align = make_shrn_cols7::<F>(align_x, cshift as usize, range_checker);
}

/// Build a `ShrnCols<7>` witness and register range-checker counts.
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
mod product_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn run(a: u32, b: u32, c: u32) -> NumericBuilder<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let mut row = FmaCoreCols::<F>::default();
        let result = fmadd_result(a, b, c, FmaddOp::Madd);

        let (a_sign, a_exp, a_mant) = decompose(a);
        let (b_sign, b_exp, b_mant) = decompose(b);
        let (c_sign, c_exp, c_mant) = decompose(c);

        row.is_valid = F::ONE;
        row.is_special = F::ZERO;
        for i in 0..4 {
            row.a[i] = F::from_u32((a >> (i * 8)) & 0xFF);
            row.b[i] = F::from_u32((b >> (i * 8)) & 0xFF);
            row.c[i] = F::from_u32((c >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        row.a_sign = F::from_u32(a_sign);
        row.b_sign = F::from_u32(b_sign);
        row.c_sign = F::from_u32(c_sign);
        row.a_exp_lo = F::from_u32(a_exp & 1);
        row.b_exp_lo = F::from_u32(b_exp & 1);
        row.c_exp_lo = F::from_u32(c_exp & 1);
        row.a_hi = F::from_u32(a_exp >> 1);
        row.b_hi = F::from_u32(b_exp >> 1);
        row.c_hi = F::from_u32(c_exp >> 1);
        row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        row.b_mant2 = F::from_u32((b_mant >> 16) & 0x7F);
        row.c_mant2 = F::from_u32((c_mant >> 16) & 0x7F);
        // Classification flags.
        row.a_exp_is_zero = F::from_bool(a_exp == 0);
        row.b_exp_is_zero = F::from_bool(b_exp == 0);
        row.c_exp_is_zero = F::from_bool(c_exp == 0);
        row.a_exp_is_max = F::from_bool(a_exp == 255);
        row.b_exp_is_max = F::from_bool(b_exp == 255);
        row.c_exp_is_max = F::from_bool(c_exp == 255);
        row.a_mant_is_zero = F::from_bool(a_mant == 0);
        row.b_mant_is_zero = F::from_bool(b_mant == 0);
        row.c_mant_is_zero = F::from_bool(c_mant == 0);

        let a_normal = (1..=254).contains(&a_exp);
        let b_normal = (1..=254).contains(&b_exp);
        let c_normal = (1..=254).contains(&c_exp);
        let c_zero_v = c_exp == 0 && c_mant == 0;
        let s_finite = a_normal && b_normal && (c_normal || c_zero_v);
        row.s_finite = F::from_bool(s_finite);
        row.a_normal = F::from_bool(a_normal);
        row.b_normal = F::from_bool(b_normal);
        row.c_normal = F::from_bool(c_normal);
        row.c_zero = F::from_bool(c_zero_v);
        row.ab_normal = F::from_bool(a_normal && b_normal);
        row.c_finite_ok = F::from_bool(c_normal || c_zero_v);

        if s_finite {
            fill_finite_product(&mut row, a_exp, b_exp, a_mant, b_mant, c_exp, c_mant, &chip);
        }

        let air = FmaCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut builder = NumericBuilder::new(vec![]);
        air.eval_finite_product(&mut builder, &row);
        air.eval_finite_align(&mut builder, &row);
        builder
    }

    #[test]
    fn product_numeric_passes() {
        for &(a, b, c) in &[
            (3.0f32.to_bits(), 4.0f32.to_bits(), 1.0f32.to_bits()),
            (1.5f32.to_bits(), 2.5f32.to_bits(), 0.5f32.to_bits()),
            (1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()),
            (
                f32::MAX.to_bits(),
                f32::MIN_POSITIVE.to_bits(),
                1.0f32.to_bits(),
            ),
            (0x3F800000u32, 0x3F800000, 0x3F800000), // 1*1+1
        ] {
            let builder = run(a, b, c);
            assert!(
                builder.ok(),
                "fma product failed: {:?}\n a={:08x} b={:08x} c={:08x}",
                builder.failures,
                a,
                b,
                c
            );
        }
        // Fuzz.
        let mut rng = 0xF00D_BEEFu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..500 {
            let a = ((next() % 254) + 1) << 23 | (next() & 0x7FFFFF);
            let b = ((next() % 254) + 1) << 23 | (next() & 0x7FFFFF);
            let c = ((next() % 254) + 1) << 23 | (next() & 0x7FFFFF);
            let builder = run(a, b, c);
            assert!(
                builder.ok(),
                "fma product fuzz failed: {:?}\n a={:08x} b={:08x} c={:08x}",
                builder.failures,
                a,
                b,
                c
            );
        }
    }
}
