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

use crate::fp_shift::{
    fill_op_norm, run_shln, run_shrn, run_shrn_round, ShlnCols, ShlnGadget, ShrnCols, ShrnGadget,
    SHLIMB_BITS,
};

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
    pub a_sub: T,
    pub b_sub: T,
    pub c_sub: T,
    pub ab_normal: T,
    pub c_finite_ok: T,
    pub s_finite: T,
    pub s_pzero: T,
    // Subnormal operand pre-normalization (leading-zeros left shift).
    pub a_lshift: T,
    pub b_lshift: T,
    pub c_lshift: T,
    pub a_shln: ShlnCols<T, 3>,
    pub b_shln: ShlnCols<T, 3>,
    pub c_shln: ShlnCols<T, 3>,
    pub a_y2_lo7: T,
    pub b_y2_lo7: T,
    pub c_y2_lo7: T,
    // c exponent used for alignment (effective; very negative for c = +-0).
    pub c_exp_eff: T,
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

    // === Phase 3: sum / difference ===
    pub s_add: T,
    pub s_cancel: T,
    pub raw_work: [T; 7],
    pub work_cb: [T; 6],
    pub add_overflow: T,
    pub work_lo: [T; 7],
    pub cancel_lshift: T,
    pub cancel_shln: ShlnCols<T, 7>,
    pub norm_work: [T; 7],
    pub raw_sum: T,
    pub raw_sum_inv: T,
    pub is_exact_zero: T,
    pub r_sign_eff: T,

    // === Phase 4: RNE rounding + output ===
    pub nw_guard: T,
    pub nw2_lo7: T,
    pub nw_lsb: T,
    pub nw3_hi7: T,
    pub nw_lsb_ok: T,
    pub below_guard: T,
    pub below_guard_nz: T,
    pub below_guard_inv: T,
    pub align_loss: T,
    pub aloss_nc: T,
    pub eff_sticky: T,
    pub round_up: T,
    pub m0: T,
    pub m1: T,
    pub rc0: T,
    pub rc1: T,
    pub rc2: T,
    pub rc2_rem: T,
    pub r_exp_lo: T,
    pub r_exp_hi: T,
    pub base_exp: T,
    pub is_ovf: T,
    pub is_sub: T,
    pub s_norm: T,
    pub s_sub: T,
    pub ovf_lt_decomp: [T; 1],
    pub sub_lt_decomp: [T; 1],

    // === Sub path: subnormal (denormalized) results ===
    pub sub_t: T,
    pub sub_deep: T,
    pub sub_deep_lt_decomp: [T; 1],
    pub sub_shrn: ShrnCols<T, 7>,
    pub sub_lsb: T,
    pub sub_hi7: T,
    pub sub_eff_sticky: T,
    pub sub_lsb_ok: T,
    pub sub_round_up: T,
    pub z0: T,
    pub z1: T,
    pub z2: T,
    pub zc0: T,
    pub zc1: T,

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
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // exp_lo / sign are free field choices (classification re-routing).
        for v in [
            cols.a_mant2,
            cols.b_mant2,
            cols.c_mant2,
            cols.a_hi,
            cols.b_hi,
            cols.c_hi,
        ] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

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

        // ── Normal / subnormal indicators ──────────────────────────────
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
        // Subnormal: exponent field 0 with nonzero mantissa.
        builder.assert_eq(
            cols.a_sub,
            cols.a_exp_is_zero * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(
            cols.b_sub,
            cols.b_exp_is_zero * (one.clone() - cols.b_mant_is_zero),
        );
        builder.assert_eq(
            cols.c_sub,
            cols.c_exp_is_zero * (one.clone() - cols.c_mant_is_zero),
        );
        for flag in [cols.a_sub, cols.b_sub, cols.c_sub] {
            builder.assert_bool(flag);
        }

        // Path selectors via intermediate columns (degree ≤ 4).
        let not_special = one.clone() - cols.is_special;
        // a_ok = a is finite nonzero (normal or subnormal), likewise b.
        builder.when(cols.is_valid).assert_eq(
            cols.ab_normal,
            (cols.a_normal + cols.a_sub) * (cols.b_normal + cols.b_sub),
        );
        builder
            .when(cols.is_valid)
            .assert_eq(cols.c_finite_ok, cols.c_normal + cols.c_sub + cols.c_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.s_finite,
            not_special.clone() * cols.ab_normal * cols.c_finite_ok,
        );
        builder.assert_bool(cols.s_finite);
        // s_pzero: the product is +-0 (a or b is zero), c finite.
        builder.when(cols.is_valid).assert_eq(
            cols.s_pzero,
            not_special * (cols.a_zero + cols.b_zero - cols.a_zero * cols.b_zero),
        );
        builder.assert_bool(cols.s_pzero);
        // Every valid row takes exactly one path.
        builder
            .when(cols.is_valid)
            .assert_one(cols.is_special + cols.s_pzero + cols.s_finite);
        let s = cols.s_finite;

        // Result for a +-0 product: c with its effective sign (or the
        // +-0 AND-rule when c is also +-0).
        let sp = cols.s_pzero;
        let cnz = one.clone() - cols.c_zero;
        builder
            .when(sp)
            .assert_eq(cols.result[0], cols.c[0] * cnz.clone());
        builder
            .when(sp)
            .assert_eq(cols.result[1], cols.c[1] * cnz.clone());
        builder
            .when(sp)
            .assert_eq(cols.result[2], cols.c[2] * cnz.clone());
        builder.when(sp).assert_eq(
            cols.result[3],
            cnz.clone() * (cols.c_hi + cols.c_sign_eff * f128)
                + cols.c_zero * cols.prod_sign_eff * cols.c_sign_eff * f128,
        );

        // ── Subnormal operand pre-normalization ────────────────────────
        // Shift the subnormal significand left so its MSB lands at bit 23
        // (the implicit-bit position); the exponent absorbs the shift, so
        // the downstream pipeline is unchanged.
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
            cols.a[0],
            cols.a[1],
            cols.a_mant2,
            &cols.a_shln,
            cols.a_y2_lo7,
        );
        eval_op_norm(
            cols.b_sub,
            cols.b_lshift,
            cols.b[0],
            cols.b[1],
            cols.b_mant2,
            &cols.b_shln,
            cols.b_y2_lo7,
        );
        eval_op_norm(
            cols.c_sub,
            cols.c_lshift,
            cols.c[0],
            cols.c[1],
            cols.c_mant2,
            &cols.c_shln,
            cols.c_y2_lo7,
        );

        // ── Significand construction ───────────────────────────────────
        // Normal: mantissa + implicit bit. Subnormal: normalized significand.
        let a_ns = one.clone() - cols.a_sub;
        builder.when(s).assert_eq(
            cols.a_sig[0],
            cols.a[0] * a_ns.clone() + cols.a_sub * cols.a_shln.y[0],
        );
        builder.when(s).assert_eq(
            cols.a_sig[1],
            cols.a[1] * a_ns.clone() + cols.a_sub * cols.a_shln.y[1],
        );
        builder.when(s).assert_eq(
            cols.a_sig[2],
            (cols.a_mant2 + f128) * a_ns + cols.a_sub * cols.a_shln.y[2],
        );
        let b_ns = one.clone() - cols.b_sub;
        builder.when(s).assert_eq(
            cols.b_sig[0],
            cols.b[0] * b_ns.clone() + cols.b_sub * cols.b_shln.y[0],
        );
        builder.when(s).assert_eq(
            cols.b_sig[1],
            cols.b[1] * b_ns.clone() + cols.b_sub * cols.b_shln.y[1],
        );
        builder.when(s).assert_eq(
            cols.b_sig[2],
            (cols.b_mant2 + f128) * b_ns + cols.b_sub * cols.b_shln.y[2],
        );
        let c_ns = one.clone() - cols.c_sub;
        builder.when(s).assert_eq(
            cols.c_sig[0],
            cols.c[0] * c_ns.clone() + cols.c_sub * cols.c_shln.y[0],
        );
        builder.when(s).assert_eq(
            cols.c_sig[1],
            cols.c[1] * c_ns.clone() + cols.c_sub * cols.c_shln.y[1],
        );
        // c = +-0 has no implicit bit (its significand is 0).
        builder.when(s).assert_eq(
            cols.c_sig[2],
            (cols.c_mant2 + (one.clone() - cols.c_zero) * f128) * c_ns
                + cols.c_sub * cols.c_shln.y[2],
        );
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
        let f56: AB::F = AB::F::from_u32(56);
        let f127: AB::F = AB::F::from_u32(127);
        let one = AB::Expr::ONE;
        let s = cols.s_finite;

        // Effective exponents: subnormal operands absorb their normalization
        // shift (value = sig * 2^(exp_eff - 127 - 23) in all cases).
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;
        let c_exp = cols.c_exp_lo + cols.c_hi * f2;
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let prod_exp = a_exp_eff + b_exp_eff - AB::Expr::from(f127) + cols.is_high;
        // c's alignment exponent: for c = +-0 the sum must be exactly the
        // product, so pretend c is far smaller (forces the deep path).
        builder.when(s).assert_eq(
            cols.c_exp_eff,
            (one.clone() - cols.c_zero)
                * ((one.clone() - cols.c_sub) * c_exp + cols.c_sub * (one.clone() - cols.c_lshift))
                + cols.c_zero * (prod_exp.clone() - AB::Expr::from(f56)),
        );

        // ── ediff = |prod_exp - c_exp_eff| ─────────────────────────────
        builder.assert_bool(cols.c_is_larger);
        let diff = prod_exp - cols.c_exp_eff;
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
        // ── Phase 3: sum / difference ──────────────────────────────────
        self.eval_finite_sumdiff(builder, cols);
    }

    /// Phase 3: Compute the sum (sign_diff=0) or difference (sign_diff=1)
    /// of the larger operand and the aligned smaller operand, then normalize.
    pub fn eval_finite_sumdiff<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmaCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;
        let s = cols.s_finite;
        let cil = cols.c_is_larger;

        // ── Path selectors ─────────────────────────────────────────────
        builder.when(s).assert_eq(
            cols.s_add,
            (one.clone() - cols.sign_diff) * (one.clone() - cols.is_deep),
        );
        builder
            .when(s)
            .assert_eq(cols.s_cancel, cols.sign_diff * (one.clone() - cols.is_deep));
        builder.assert_bool(cols.s_add);
        builder.assert_bool(cols.s_cancel);

        // ── Larger operand bytes (mux product vs c) ────────────────────
        // c_is_larger=0: larger = [norm_prod[0..5], 0].
        // c_is_larger=1: larger = [0,0,0, c_sig[0], c_sig[1], c_sig[2], 0].
        let large = |i: usize| -> AB::Expr {
            let p_val = if i < 6 {
                cols.norm_prod[i].into()
            } else {
                AB::Expr::ZERO
            };
            let c_val = match i {
                3 => cols.c_sig[0].into(),
                4 => cols.c_sig[1].into(),
                5 => cols.c_sig[2].into(),
                _ => AB::Expr::ZERO,
            };
            (one.clone() - cil) * p_val + cil * c_val
        };

        // ── Sum/difference carry/borrow chain (7 bytes) ─────────────────
        let s_add = cols.s_add;
        let s_cancel = cols.s_cancel;
        for i in 0..7 {
            let cb_in = if i == 0 {
                AB::Expr::ZERO
            } else {
                cols.work_cb[i - 1].into()
            };
            if i < 6 {
                let cb_out = cols.work_cb[i].into();
                builder.when(s_add).assert_eq(
                    cols.raw_work[i] + cb_out.clone() * f256,
                    large(i) + cols.align.y[i] + cb_in.clone(),
                );
                builder.when(s_cancel).assert_eq(
                    cols.raw_work[i],
                    large(i) - cols.align.y[i] - cb_in + cb_out * f256,
                );
            } else {
                // Byte 6: last byte, no carry/borrow out.
                builder
                    .when(s_add)
                    .assert_eq(cols.raw_work[i], large(i) + cols.align.y[i] + cb_in.clone());
                builder
                    .when(s_cancel)
                    .assert_eq(cols.raw_work[i], large(i) - cols.align.y[i] - cb_in);
            }
            self.range_bus
                .range_check(cols.raw_work[i].into(), LIMB_BITS as usize)
                .eval(builder, s.into());
            if i < 6 {
                builder.assert_bool(cols.work_cb[i]);
                self.range_bus
                    .range_check(cols.work_cb[i].into(), LIMB_BITS as usize)
                    .eval(builder, s.into());
            }
        }
        // Add overflow: add_overflow = (raw_work[6] > 0). Boolean; cancel must be 0.
        builder.assert_bool(cols.add_overflow);
        builder.when(s_cancel).assert_zero(cols.add_overflow);
        // When add_overflow=0, raw_work[6] must be 0 (no overflow to byte 6).
        builder
            .when(s_add)
            .assert_zero(cols.raw_work[6] * (one.clone() - cols.add_overflow));

        // ── Byte decomposition for add-overflow right-shift ────────────
        for i in 0..7 {
            builder.assert_bool(cols.work_lo[i]);
            // raw_work[i] = work_lo[i] + h7*2 where h7 is range-checked via norm constraint.
        }

        // ── Add normalization: if add_overflow, shift right by 1 ────────
        let inv2 = f2.inverse();
        for i in 0..7 {
            let lo_next = if i < 6 {
                cols.work_lo[i + 1].into()
            } else {
                AB::Expr::ZERO // byte 6 has no byte 7 to pull from
            };
            // When add_overflow: norm_work[i] = (raw-lo)*inv2 + lo_next*128.
            // When no overflow: norm_work[i] = raw_work[i].
            // Mux on add_overflow, gated on s_add.
            let shifted = (cols.raw_work[i] - cols.work_lo[i]) * inv2 + lo_next * f128;
            builder
                .when(s_add * cols.add_overflow)
                .assert_eq(cols.norm_work[i], shifted);
            builder
                .when(s_add * (one.clone() - cols.add_overflow))
                .assert_eq(cols.norm_work[i], cols.raw_work[i]);
            // Range-check (raw_work[i] - lo[i]) * inv2 to 7 bits (= work_hi7).
            let hi7 = (cols.raw_work[i] - cols.work_lo[i]) * inv2;
            self.range_bus
                .range_check(hi7, 7)
                .eval(builder, s_add.into());
        }

        // ── Cancel normalization via ShlnGadget<7> ──────────────────────
        for i in 0..7 {
            builder
                .when(s_cancel)
                .assert_eq(cols.cancel_shln.x[i], cols.raw_work[i]);
        }
        builder
            .when(s_cancel)
            .assert_eq(cols.cancel_shln.shift, cols.cancel_lshift);
        builder.assert_eq(cols.cancel_shln.enable, s_cancel);
        let shln = ShlnGadget::new(self.range_bus);
        shln.eval_shln(builder, &cols.cancel_shln);
        for i in 0..7 {
            builder
                .when(s_cancel)
                .assert_eq(cols.norm_work[i], cols.cancel_shln.y[i]);
            self.range_bus
                .range_check(cols.norm_work[i].into(), LIMB_BITS as usize)
                .eval(builder, s_cancel.into());
        }

        // ── Deep path: norm_work = larger operand ──────────────────────
        for i in 0..7 {
            builder
                .when(s * cols.is_deep)
                .assert_eq(cols.norm_work[i], large(i));
            self.range_bus
                .range_check(cols.norm_work[i].into(), LIMB_BITS as usize)
                .eval(builder, s * cols.is_deep);
        }

        // ── Exact-zero cancellation detection ──────────────────────────
        // raw_work == 0 happens only on the cancel path (add/deep include a
        // nonzero larger operand); it means the result is exactly +-0.
        let raw_sum_expr: AB::Expr = cols.raw_work.iter().map(|&v| v.into()).sum();
        builder.when(s).assert_eq(cols.raw_sum, raw_sum_expr);
        let ez_free = one.clone() - cols.raw_sum * cols.raw_sum_inv;
        builder
            .when(s)
            .assert_eq(cols.is_exact_zero, ez_free.clone());
        builder.assert_zero(cols.raw_sum * ez_free);
        builder.assert_bool(cols.is_exact_zero);
        // On exact-zero rows the normalization shift is meaningless; pin it
        // so base_exp stays the larger operand's exponent.
        builder
            .when(s * cols.is_exact_zero)
            .assert_zero(cols.cancel_lshift);

        // ── Result sign ────────────────────────────────────────────────
        // Add/deep: sign = prod_sign_eff.
        // Cancel: sign = prod_sign_eff if product >= c, else c_sign_eff.
        // When c_is_larger: result sign = c_sign_eff (c was larger in cancellation).
        // When !c_is_larger: result sign = prod_sign_eff.
        builder
            .when(s * (one.clone() - cols.sign_diff))
            .assert_eq(cols.r_sign_eff, cols.prod_sign_eff);
        builder
            .when(s_cancel * cil)
            .assert_eq(cols.r_sign_eff, cols.c_sign_eff);
        builder
            .when(s_cancel * (one.clone() - cil))
            .assert_eq(cols.r_sign_eff, cols.prod_sign_eff);
        builder
            .when(s * cols.is_deep * cil)
            .assert_eq(cols.r_sign_eff, cols.c_sign_eff);
        builder
            .when(s * cols.is_deep * (one.clone() - cil))
            .assert_eq(cols.r_sign_eff, cols.prod_sign_eff);

        // ── Phase 4: RNE rounding + output reconstruction ─────────────
        self.eval_finite_output(builder, cols);
    }

    /// Phase 4: Three output paths depending on the result exponent:
    /// normal (RNE rounding + implicit-bit reconstruction), overflow (+-inf),
    /// and subnormal (denormalizing right shift with single RNE rounding).
    /// Exact-zero cancellation outputs +0.
    pub fn eval_finite_output<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FmaCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f25: AB::F = AB::F::from_u32(25);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;
        let s = cols.s_finite;

        // ── Result class: normal / overflow / subnormal / exact-zero ────
        // base_exp (before rounding carry) = larger_exp + add_overflow -
        // cancel_lshift. base_exp in [-224, 383]; res_exp_b = base_exp + 256
        // is range-checked to 10 bits.
        let cil = cols.c_is_larger;
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let prod_exp = a_exp_eff + b_exp_eff - AB::Expr::from(f127) + cols.is_high;
        let larger_exp = (one.clone() - cil) * prod_exp + cil * cols.c_exp_eff;
        builder.when(s).assert_eq(
            cols.base_exp,
            larger_exp + cols.add_overflow - cols.cancel_lshift * cols.s_cancel,
        );
        let res_exp_b = cols.base_exp + AB::Expr::from(f256);
        self.range_bus
            .range_check(res_exp_b.clone(), 10)
            .eval(builder, s.into());
        builder.assert_bool(cols.is_ovf);
        builder.assert_bool(cols.is_sub);
        // is_ovf: base_exp >= 255 (255 + 256 = 511).
        let lt_air = IsLtSubAir::new(self.range_bus, 10);
        let ovf_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(510)),
            res_exp_b.clone(),
            cols.is_ovf.into(),
            s.into(),
        );
        lt_air.eval(builder, (ovf_io, &cols.ovf_lt_decomp));
        // is_sub: base_exp <= 0 (0 + 256 = 256 < 257).
        let sub_io = IsLessThanIo::new(
            res_exp_b,
            AB::Expr::from(AB::F::from_u32(257)),
            cols.is_sub.into(),
            s.into(),
        );
        lt_air.eval(builder, (sub_io, &cols.sub_lt_decomp));
        // is_ovf and is_sub are mutually exclusive.
        builder.assert_zero(cols.is_ovf * cols.is_sub);
        let ez = cols.is_exact_zero;
        builder.when(s).assert_eq(
            cols.s_norm,
            (one.clone() - cols.is_ovf) * (one.clone() - cols.is_sub) * (one.clone() - ez),
        );
        builder.assert_bool(cols.s_norm);
        builder.assert_eq(cols.s_sub, s * cols.is_sub * (one.clone() - ez));
        builder.assert_bool(cols.s_sub);
        let sn = cols.s_norm;
        let ss = cols.s_sub;

        // ── Exact-zero result: +0 (RNE) ─────────────────────────────────
        let sez = s * ez;
        for i in 0..4 {
            builder.when(sez.clone()).assert_zero(cols.result[i]);
        }

        // ── Guard / LSB extraction (normal path) ────────────────────────
        // Guard = bit 7 of norm_work[2]. LSB = bit 0 of norm_work[3].
        builder.assert_bool(cols.nw_guard);
        builder
            .when(sn)
            .assert_eq(cols.norm_work[2], cols.nw_guard * f128 + cols.nw2_lo7);
        self.range_bus
            .range_check(cols.nw2_lo7.into(), 7)
            .eval(builder, sn.into());
        builder.assert_bool(cols.nw_lsb);
        builder
            .when(sn)
            .assert_eq(cols.norm_work[3], cols.nw_lsb + cols.nw3_hi7 * f2);
        self.range_bus
            .range_check(cols.nw3_hi7.into(), 7)
            .eval(builder, sn.into());

        // Byte 6 of the normalized work value must be zero: together with
        // the implicit-bit check below this anchors the normalization shift
        // (a prover cannot over-shift and compensate via the exponent).
        builder.when(s).assert_zero(cols.norm_work[6]);

        // ── Sticky: bits below guard (shared witness) ───────────────────
        let below = cols.nw2_lo7 + cols.norm_work[1] + cols.norm_work[0];
        builder.when(sn).assert_eq(cols.below_guard, below);
        let bg_is_zero = one.clone() - cols.below_guard * cols.below_guard_inv;
        builder.assert_zero(cols.below_guard * bg_is_zero.clone());
        builder
            .when(sn)
            .assert_eq(cols.below_guard_nz, one.clone() - bg_is_zero);
        builder.assert_bool(cols.below_guard_nz);

        // Alignment loss indicator (shared by all rounding paths). The lost
        // fraction is align.guard*0.5 + align.sticky*(<0.5) working units.
        // On add/deep paths the lost bits add to the magnitude (a tie with
        // any loss rounds up); on the cancel path they are *subtracted*
        // (true value strictly less than raw << lshift, always by less than
        // half a ULP), so a tie rounds down instead of to-even.
        builder.when(s).assert_eq(
            cols.align_loss,
            cols.align.guard + cols.align.sticky - cols.align.guard * cols.align.sticky,
        );
        builder.assert_bool(cols.align_loss);
        builder.when(s).assert_eq(
            cols.aloss_nc,
            cols.align_loss * (one.clone() - cols.s_cancel),
        );
        builder.assert_bool(cols.aloss_nc);

        // Effective sticky (normal path): below OR align_loss-when-not-cancel.
        let sticky_expr = cols.below_guard_nz + cols.aloss_nc - cols.below_guard_nz * cols.aloss_nc;
        builder.when(sn).assert_eq(cols.eff_sticky, sticky_expr);
        builder.assert_bool(cols.eff_sticky);

        // lsb_ok: the lsb tie-break applies unless this is a cancellation
        // with subtractive loss (then a tie rounds down).
        builder.when(sn).assert_eq(
            cols.nw_lsb_ok,
            cols.nw_lsb * (one.clone() - cols.s_cancel * cols.align_loss),
        );
        builder.assert_bool(cols.nw_lsb_ok);

        // round_up = guard AND (eff_sticky OR lsb_ok).
        let sor = cols.eff_sticky + cols.nw_lsb_ok - cols.eff_sticky * cols.nw_lsb_ok;
        builder
            .when(sn)
            .assert_eq(cols.round_up, cols.nw_guard * sor);
        builder.assert_bool(cols.round_up);

        // ── Mantissa carry chain (normal path) ──────────────────────────
        builder.assert_bool(cols.rc0);
        builder.assert_bool(cols.rc1);
        builder.assert_bool(cols.rc2);
        builder
            .when(sn)
            .assert_eq(cols.m0 + cols.rc0 * f256, cols.norm_work[3] + cols.round_up);
        builder
            .when(sn)
            .assert_eq(cols.m1 + cols.rc1 * f256, cols.norm_work[4] + cols.rc0);
        self.range_bus
            .range_check(cols.m0.into(), 8)
            .eval(builder, sn.into());
        self.range_bus
            .range_check(cols.m1.into(), 8)
            .eval(builder, sn.into());
        // top7 = norm_work[5] - 128 (remove implicit bit).
        let top7 = cols.norm_work[5] - f128;
        builder
            .when(sn)
            .assert_eq(top7 + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, sn.into());

        // ── Result exponent (normal path; 255 encodes +-inf) ────────────
        builder
            .when(sn)
            .assert_eq(cols.r_exp_lo + cols.r_exp_hi * f2, cols.base_exp + cols.rc2);

        // ── Result bytes (normal path) ──────────────────────────────────
        builder.when(sn).assert_eq(cols.result[0], cols.m0);
        builder.when(sn).assert_eq(cols.result[1], cols.m1);
        builder
            .when(sn)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        builder
            .when(sn)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.r_sign_eff * f128);

        // ── Overflow: result = +-inf ────────────────────────────────────
        let so = s * cols.is_ovf;
        builder.when(so.clone()).assert_zero(cols.result[0]);
        builder.when(so.clone()).assert_zero(cols.result[1]);
        builder
            .when(so.clone())
            .assert_eq(cols.result[2], AB::F::from_u32(128));
        builder.when(so).assert_eq(
            cols.result[3],
            AB::Expr::from(f127) + cols.r_sign_eff * f128,
        );

        // ── Sub path: subnormal results ─────────────────────────────────
        // Denormalize: shift norm_work right by t = 25 - base_exp (>= 25),
        // clamped to 55 (deeper shifts always round to +-0). Single RNE
        // rounding from the gadget's guard/sticky; the round carry into bit
        // 23 yields the smallest normal naturally.
        builder
            .when(ss)
            .assert_eq(cols.sub_t, AB::Expr::from(f25) - cols.base_exp);
        self.range_bus
            .range_check(cols.sub_t.into(), 8)
            .eval(builder, ss.into());
        builder.assert_bool(cols.sub_deep);
        let lt8 = IsLtSubAir::new(self.range_bus, 8);
        let deep_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(55)),
            cols.sub_t.into(),
            cols.sub_deep.into(),
            ss.into(),
        );
        lt8.eval(builder, (deep_io, &cols.sub_deep_lt_decomp));
        builder.when(ss).assert_eq(
            cols.sub_shrn.shift,
            (one.clone() - cols.sub_deep) * cols.sub_t + cols.sub_deep * AB::F::from_u32(55),
        );
        for i in 0..7 {
            builder
                .when(ss)
                .assert_eq(cols.sub_shrn.x[i], cols.norm_work[i]);
        }
        builder.assert_eq(cols.sub_shrn.enable, ss);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.sub_shrn);

        // Rounding: guard/sticky from the gadget, lsb from y[0].
        builder.assert_bool(cols.sub_lsb);
        builder
            .when(ss)
            .assert_eq(cols.sub_shrn.y[0], cols.sub_lsb + cols.sub_hi7 * f2);
        self.range_bus
            .range_check(cols.sub_hi7.into(), 7)
            .eval(builder, ss.into());
        let sub_sticky_expr =
            cols.sub_shrn.sticky + cols.aloss_nc - cols.sub_shrn.sticky * cols.aloss_nc;
        builder
            .when(ss)
            .assert_eq(cols.sub_eff_sticky, sub_sticky_expr);
        builder.assert_bool(cols.sub_eff_sticky);
        builder.when(ss).assert_eq(
            cols.sub_lsb_ok,
            cols.sub_lsb * (one.clone() - cols.s_cancel * cols.align_loss),
        );
        builder.assert_bool(cols.sub_lsb_ok);
        let sub_sor = cols.sub_eff_sticky + cols.sub_lsb_ok - cols.sub_eff_sticky * cols.sub_lsb_ok;
        builder
            .when(ss)
            .assert_eq(cols.sub_round_up, cols.sub_shrn.guard * sub_sor);
        builder.assert_bool(cols.sub_round_up);

        // z = y + round_up (<= 2^23; bit 23 of z is the exp field LSB).
        builder.assert_bool(cols.zc0);
        builder.assert_bool(cols.zc1);
        builder.when(ss).assert_eq(
            cols.z0 + cols.zc0 * f256,
            cols.sub_shrn.y[0] + cols.sub_round_up,
        );
        builder
            .when(ss)
            .assert_eq(cols.z1 + cols.zc1 * f256, cols.sub_shrn.y[1] + cols.zc0);
        builder
            .when(ss)
            .assert_eq(cols.z2, cols.sub_shrn.y[2] + cols.zc1);
        for z in [cols.z0, cols.z1, cols.z2] {
            self.range_bus
                .range_check(z.into(), 8)
                .eval(builder, ss.into());
        }
        builder.when(ss).assert_eq(cols.result[0], cols.z0);
        builder.when(ss).assert_eq(cols.result[1], cols.z1);
        builder.when(ss).assert_eq(cols.result[2], cols.z2);
        builder
            .when(ss)
            .assert_eq(cols.result[3], cols.r_sign_eff * f128);
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
        fill_fma_witness(
            core_row,
            a_bits,
            b_bits,
            c_bits,
            op,
            fmadd_result(a_bits, b_bits, c_bits, op),
            &self.range_checker,
        );
    }
}

/// Fill a complete FMADD family core witness row (production filler + corruption tests).
fn fill_fma_witness<F: PrimeField32>(
    core_row: &mut FmaCoreCols<F>,
    a_bits: u32,
    b_bits: u32,
    c_bits: u32,
    op: FmaddOp,
    result: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    for i in 0..4 {
        core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
        core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
        core_row.c[i] = F::from_u32((c_bits >> (i * 8)) & 0xFF);
        core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
        range_checker.add_count((b_bits >> (i * 8)) & 0xFF, 8);
        range_checker.add_count((c_bits >> (i * 8)) & 0xFF, 8);
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
        range_checker.add_count((mant >> 16) & 0x7F, 7);
        range_checker.add_count(exp >> 1, 7);
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
    let (c_sign, c_exp, c_mant) = decompose(c_bits);
    let a_nan = a_exp == 255 && a_mant != 0;
    let b_nan = b_exp == 255 && b_mant != 0;
    let c_nan = c_exp == 255 && c_mant != 0;
    let a_inf = a_exp == 255 && a_mant == 0;
    let b_inf = b_exp == 255 && b_mant == 0;
    let c_inf = c_exp == 255 && c_mant == 0;
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

    let a_normal = (1..=254).contains(&a_exp);
    let b_normal = (1..=254).contains(&b_exp);
    let c_normal = (1..=254).contains(&c_exp);
    let a_sub = a_exp == 0 && a_mant != 0;
    let b_sub = b_exp == 0 && b_mant != 0;
    let c_sub = c_exp == 0 && c_mant != 0;
    let c_zero_v = c_exp == 0 && c_mant == 0;
    let a_ok = a_normal || a_sub;
    let b_ok = b_normal || b_sub;
    let c_fok = c_normal || c_sub || c_zero_v;
    let s_finite = !is_special && a_ok && b_ok && c_fok;
    let s_pzero = !is_special && (a_zero || b_zero);
    debug_assert!(is_special || s_pzero || s_finite);
    core_row.a_normal = F::from_bool(a_normal);
    core_row.b_normal = F::from_bool(b_normal);
    core_row.c_normal = F::from_bool(c_normal);
    core_row.a_sub = F::from_bool(a_sub);
    core_row.b_sub = F::from_bool(b_sub);
    core_row.c_sub = F::from_bool(c_sub);
    core_row.c_zero = F::from_bool(c_zero_v);
    core_row.ab_normal = F::from_bool(a_ok && b_ok);
    core_row.c_finite_ok = F::from_bool(c_fok);
    core_row.s_finite = F::from_bool(s_finite);
    core_row.s_pzero = F::from_bool(s_pzero);

    if s_finite {
        fill_finite_product(
            core_row,
            a_exp,
            b_exp,
            a_mant,
            b_mant,
            c_exp,
            c_mant,
            prod_sign_eff,
            c_sign_eff,
            sign_diff,
            result,
            range_checker,
        );
    } else {
        zero_finite_columns(core_row);
        fill_op_norm(
            a_exp,
            a_mant,
            &mut core_row.a_lshift,
            &mut core_row.a_shln,
            &mut core_row.a_y2_lo7,
            range_checker,
        );
        fill_op_norm(
            b_exp,
            b_mant,
            &mut core_row.b_lshift,
            &mut core_row.b_shln,
            &mut core_row.b_y2_lo7,
            range_checker,
        );
        fill_op_norm(
            c_exp,
            c_mant,
            &mut core_row.c_lshift,
            &mut core_row.c_shln,
            &mut core_row.c_y2_lo7,
            range_checker,
        );
    }

    core_row.is_valid = F::ONE;
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
    row.a_lshift = F::ZERO;
    row.b_lshift = F::ZERO;
    row.c_lshift = F::ZERO;
    row.a_shln = ShlnCols::<F, 3>::default();
    row.b_shln = ShlnCols::<F, 3>::default();
    row.c_shln = ShlnCols::<F, 3>::default();
    row.a_y2_lo7 = F::ZERO;
    row.b_y2_lo7 = F::ZERO;
    row.c_y2_lo7 = F::ZERO;
    row.c_exp_eff = F::ZERO;
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
    row.s_add = F::ZERO;
    row.s_cancel = F::ZERO;
    for i in 0..7 {
        row.raw_work[i] = F::ZERO;
        row.norm_work[i] = F::ZERO;
        row.work_lo[i] = F::ZERO;
    }
    for i in 0..6 {
        row.work_cb[i] = F::ZERO;
    }
    row.add_overflow = F::ZERO;
    row.cancel_lshift = F::ZERO;
    row.cancel_shln = ShlnCols::<F, 7>::default();
    row.raw_sum = F::ZERO;
    row.raw_sum_inv = F::ZERO;
    row.is_exact_zero = F::ZERO;
    row.r_sign_eff = F::ZERO;
    row.nw_guard = F::ZERO;
    row.nw2_lo7 = F::ZERO;
    row.nw_lsb = F::ZERO;
    row.nw3_hi7 = F::ZERO;
    row.nw_lsb_ok = F::ZERO;
    row.below_guard = F::ZERO;
    row.below_guard_nz = F::ZERO;
    row.below_guard_inv = F::ZERO;
    row.align_loss = F::ZERO;
    row.aloss_nc = F::ZERO;
    row.eff_sticky = F::ZERO;
    row.round_up = F::ZERO;
    row.m0 = F::ZERO;
    row.m1 = F::ZERO;
    row.rc0 = F::ZERO;
    row.rc1 = F::ZERO;
    row.rc2 = F::ZERO;
    row.rc2_rem = F::ZERO;
    row.r_exp_lo = F::ZERO;
    row.r_exp_hi = F::ZERO;
    row.base_exp = F::ZERO;
    row.is_ovf = F::ZERO;
    row.is_sub = F::ZERO;
    row.s_norm = F::ZERO;
    row.s_sub = F::ZERO;
    row.ovf_lt_decomp = [F::ZERO];
    row.sub_lt_decomp = [F::ZERO];
    row.sub_t = F::ZERO;
    row.sub_deep = F::ZERO;
    row.sub_deep_lt_decomp = [F::ZERO];
    row.sub_shrn = ShrnCols::<F, 7>::default();
    row.sub_lsb = F::ZERO;
    row.sub_hi7 = F::ZERO;
    row.sub_eff_sticky = F::ZERO;
    row.sub_lsb_ok = F::ZERO;
    row.sub_round_up = F::ZERO;
    row.z0 = F::ZERO;
    row.z1 = F::ZERO;
    row.z2 = F::ZERO;
    row.zc0 = F::ZERO;
    row.zc1 = F::ZERO;
}

/// Fill the product + alignment + sum/diff witness for the finite path.
#[allow(clippy::too_many_arguments)]
fn fill_finite_product<F: PrimeField32>(
    row: &mut FmaCoreCols<F>,
    a_exp: u32,
    b_exp: u32,
    a_mant: u32,
    b_mant: u32,
    c_exp: u32,
    c_mant: u32,
    prod_sign_eff: bool,
    c_sign_eff: bool,
    sign_diff: bool,
    result: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    // ── Subnormal operand pre-normalization ──────────────────────────
    let (_a_sub, a_exp_eff, a_sig) = fill_op_norm(
        a_exp,
        a_mant,
        &mut row.a_lshift,
        &mut row.a_shln,
        &mut row.a_y2_lo7,
        range_checker,
    );
    let (_b_sub, b_exp_eff, b_sig) = fill_op_norm(
        b_exp,
        b_mant,
        &mut row.b_lshift,
        &mut row.b_shln,
        &mut row.b_y2_lo7,
        range_checker,
    );
    let (_c_sub, c_exp_sub_eff, c_sig_norm) = fill_op_norm(
        c_exp,
        c_mant,
        &mut row.c_lshift,
        &mut row.c_shln,
        &mut row.c_y2_lo7,
        range_checker,
    );

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

    // c_sig construction (for Phase 2 alignment). c = +-0 has no implicit bit.
    let c_zero_v = c_exp == 0 && c_mant == 0;
    let c_sig_val = if c_zero_v { 0 } else { c_sig_norm };
    let csig_bytes: [u32; 3] = [
        c_sig_val & 0xFF,
        (c_sig_val >> 8) & 0xFF,
        (c_sig_val >> 16) & 0xFF,
    ];
    for (i, &cb) in csig_bytes.iter().enumerate() {
        row.c_sig[i] = F::from_u32(cb);
        range_checker.add_count(csig_bytes[i], 8);
    }

    // ── Phase 2: Addend alignment ──────────────────────────────────
    let prod_exp_i = a_exp_eff + b_exp_eff - 127 + is_high as i32;
    // c = +-0 is treated as far smaller than the product (deep path).
    let c_exp_eff_i = if c_zero_v {
        prod_exp_i - 56
    } else {
        c_exp_sub_eff
    };
    let diff_i = prod_exp_i - c_exp_eff_i;
    // At equal exponents the larger operand is decided by significand
    // magnitude (the AIR enforces the choice via the borrow chain).
    let c_placed = (c_sig_val as u64) << 24;
    let c_is_larger = diff_i < 0 || (diff_i == 0 && c_placed > norm);
    let ediff = diff_i.unsigned_abs();
    debug_assert!(ediff < 512, "fma ediff out of range: {ediff}");
    let is_deep = ediff >= 56;
    let cshift = if is_deep { 55u32 } else { ediff };

    row.ediff = F::from_u32(ediff);
    range_checker.add_count(ediff, 9);
    row.c_is_larger = F::from_bool(c_is_larger);
    row.is_deep = F::from_bool(is_deep);
    row.c_exp_eff = if c_exp_eff_i < 0 {
        -F::from_u32((-c_exp_eff_i) as u32)
    } else {
        F::from_u32(c_exp_eff_i as u32)
    };

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

    // ── Phase 3: sum / difference ──────────────────────────────────
    let s_add = !sign_diff && !is_deep;
    let s_cancel = sign_diff && !is_deep;
    row.s_add = F::from_bool(s_add);
    row.s_cancel = F::from_bool(s_cancel);

    // Larger operand bytes.
    let large_bytes: [u32; 7] = if c_is_larger {
        [0, 0, 0, csig_bytes[0], csig_bytes[1], csig_bytes[2], 0]
    } else {
        let norm_arr: [u32; 6] = core::array::from_fn(|i| ((norm >> (i * 8)) & 0xFF) as u32);
        [
            norm_arr[0],
            norm_arr[1],
            norm_arr[2],
            norm_arr[3],
            norm_arr[4],
            norm_arr[5],
            0,
        ]
    };
    // Aligned smaller operand bytes.
    let align_y: [u32; 7] = core::array::from_fn(|i| row.align.y[i].as_canonical_u32());

    // Sum or difference.
    let raw_bytes: [u32; 7];
    let cb_bytes: [u32; 6];
    if s_add {
        let mut carry: u32 = 0;
        let mut rb = [0u32; 7];
        let mut cb = [0u32; 6];
        for i in 0..7 {
            let val = large_bytes[i] + align_y[i] + carry;
            rb[i] = val & 0xFF;
            carry = val >> 8;
            if i < 6 {
                cb[i] = carry;
            }
        }
        raw_bytes = rb;
        cb_bytes = cb;
    } else if s_cancel {
        let mut borrow: i64 = 0;
        let mut rb = [0u32; 7];
        let mut cb = [0u32; 6];
        for i in 0..7 {
            let d = large_bytes[i] as i64 - align_y[i] as i64 - borrow;
            if d < 0 {
                rb[i] = (d + 256) as u32;
                borrow = 1;
            } else {
                rb[i] = d as u32;
                borrow = 0;
            }
            if i < 6 {
                cb[i] = borrow as u32;
            }
        }
        debug_assert_eq!(borrow, 0, "fma cancel: difference is negative");
        raw_bytes = rb;
        cb_bytes = cb;
    } else {
        // Deep: raw_work = larger (passthrough).
        raw_bytes = large_bytes;
        cb_bytes = [0; 6];
    }

    // add_overflow = (byte 6 of raw_work is nonzero): sum exceeded 48 bits.
    let add_overflow_val: u32 = if s_add { (raw_bytes[6] > 0) as u32 } else { 0 };

    for (i, &rb) in raw_bytes.iter().enumerate() {
        row.raw_work[i] = F::from_u32(rb);
        range_checker.add_count(rb, 8);
        row.work_lo[i] = F::from_u32(rb & 1);
    }
    for (i, &cb) in cb_bytes.iter().enumerate() {
        row.work_cb[i] = F::from_u32(cb);
        range_checker.add_count(cb, 8);
    }
    row.add_overflow = F::from_u32(add_overflow_val);

    // Exact-zero cancellation (raw_work all zero).
    let raw_sum: u32 = raw_bytes.iter().sum();
    let is_exact_zero = raw_sum == 0;
    row.raw_sum = F::from_u32(raw_sum);
    row.is_exact_zero = F::from_bool(is_exact_zero);
    row.raw_sum_inv = if is_exact_zero {
        F::ZERO
    } else {
        F::from_u32(raw_sum).try_inverse().unwrap_or(F::ZERO)
    };

    // Initialize cancel columns to default (overwritten if cancel path).
    row.cancel_lshift = F::ZERO;
    row.cancel_shln = ShlnCols::<F, 7>::default();

    // Normalization.
    if s_add && add_overflow_val == 1 {
        // Right-shift by 1. AIR sends hi7 range check (count=s_add).
        for i in 0..7 {
            let lo_next = if i < 6 {
                raw_bytes[i + 1] & 1
            } else {
                0 // byte 6 has no byte 7
            };
            let nw = (raw_bytes[i] >> 1) | (lo_next << 7);
            row.norm_work[i] = F::from_u32(nw);
            range_checker.add_count(raw_bytes[i] >> 1, 7);
        }
    } else if s_cancel {
        // Left-shift by leading zeros to normalize (MSB to bit 47).
        let raw_val: u64 = (0..7).fold(0u64, |acc, i| acc | ((raw_bytes[i] as u64) << (i * 8)));
        let lshift = if raw_val == 0 {
            0
        } else {
            ((raw_val.leading_zeros() - 16) as usize).min(55)
        };
        row.cancel_lshift = F::from_u32(lshift as u32);
        let shifted = raw_val << lshift;
        let norm_arr: [u32; 7] = core::array::from_fn(|i| ((shifted >> (i * 8)) & 0xFF) as u32);
        // Build ShlnCols.
        row.cancel_shln = ShlnCols::<F, 7>::default();
        for (i, (&rb, &nb)) in raw_bytes.iter().zip(norm_arr.iter()).enumerate() {
            row.cancel_shln.x[i] = F::from_u32(rb);
            row.norm_work[i] = F::from_u32(nb);
            range_checker.add_count(nb, 8);
        }
        let (y_shl, q_shl, r_shl) = run_shln(&raw_bytes, lshift);
        debug_assert_eq!(norm_arr, core::array::from_fn(|i| y_shl[i]));
        let mut lsm = [F::ZERO; 7];
        let mut bsm = [F::ZERO; SHLIMB_BITS];
        lsm[q_shl] = F::ONE;
        bsm[r_shl] = F::ONE;
        let cv = |v: u32| {
            if r_shl == 0 {
                0
            } else {
                v >> (SHLIMB_BITS - r_shl)
            }
        };
        row.cancel_shln.shift = F::from_u32(lshift as u32);
        row.cancel_shln.enable = F::ONE;
        row.cancel_shln.limb_shift_marker = lsm;
        row.cancel_shln.bit_shift_marker = bsm;
        row.cancel_shln.bit_mult = F::from_u32(1u32 << r_shl);
        row.cancel_shln.bit_shift_carry = core::array::from_fn(|i| F::from_u32(cv(raw_bytes[i])));
        row.cancel_shln.y = core::array::from_fn(|i| F::from_u32(y_shl[i]));
        for i in 0..7 {
            range_checker.add_count(cv(raw_bytes[i]), r_shl);
            range_checker.add_count(y_shl[i], 8);
        }
    } else {
        // Deep or add-no-overflow: norm_work = raw_work.
        for (i, &rb) in raw_bytes.iter().enumerate() {
            row.norm_work[i] = F::from_u32(rb);
            if is_deep {
                range_checker.add_count(rb, 8);
            }
        }
        if s_add {
            for &rb in &raw_bytes {
                range_checker.add_count(rb >> 1, 7);
            }
        }
    }

    // Result sign: add path takes the product sign; cancellation/deep take
    // the larger operand's sign (matches the AIR gating).
    row.r_sign_eff = if (s_cancel || is_deep) && c_is_larger {
        F::from_bool(c_sign_eff)
    } else {
        F::from_bool(prod_sign_eff)
    };

    // ── Phase 4: output paths ──────────────────────────────────────
    let norm_bytes: [u32; 7] = core::array::from_fn(|i| row.norm_work[i].as_canonical_u32());

    // Alignment loss indicator (shared by all rounding paths).
    let align_guard = row.align.guard.as_canonical_u32() != 0;
    let align_sticky = row.align.sticky.as_canonical_u32() != 0;
    let align_loss = align_guard || align_sticky;
    row.align_loss = F::from_bool(align_loss);
    row.aloss_nc = F::from_bool(align_loss && !s_cancel);

    // Result class: base_exp is the exponent before the rounding carry.
    let larger_exp_i = if c_is_larger { c_exp_eff_i } else { prod_exp_i };
    let base_exp_i = larger_exp_i + add_overflow_val as i32
        - if s_cancel {
            row.cancel_lshift.as_canonical_u32() as i32
        } else {
            0
        };
    row.base_exp = if base_exp_i < 0 {
        -F::from_u32((-base_exp_i) as u32)
    } else {
        F::from_u32(base_exp_i as u32)
    };
    let res_exp_b = base_exp_i + 256;
    debug_assert!(
        (0..1024).contains(&res_exp_b),
        "fma res_exp_b out of range: {res_exp_b}"
    );
    range_checker.add_count(res_exp_b as u32, 10);

    let is_ovf = base_exp_i >= 255;
    let is_sub = base_exp_i <= 0;
    let s_norm = !is_ovf && !is_sub && !is_exact_zero;
    let s_sub = is_sub && !is_exact_zero;
    let lt_air = IsLtSubAir::new(range_checker.bus(), 10);
    lt_air.generate_subrow(
        (range_checker.as_ref(), 510u32, res_exp_b as u32),
        (&mut row.ovf_lt_decomp, &mut row.is_ovf),
    );
    lt_air.generate_subrow(
        (range_checker.as_ref(), res_exp_b as u32, 257u32),
        (&mut row.sub_lt_decomp, &mut row.is_sub),
    );
    debug_assert_eq!(row.is_ovf, F::from_bool(is_ovf));
    debug_assert_eq!(row.is_sub, F::from_bool(is_sub));
    row.s_norm = F::from_bool(s_norm);
    row.s_sub = F::from_bool(s_sub);

    // ── Normal path: RNE rounding + implicit-bit reconstruction ─────
    // Guard = bit 7 of norm_work[2]. LSB = bit 0 of norm_work[3].
    let guard = (norm_bytes[2] >> 7) & 1;
    let nw2_lo7 = norm_bytes[2] & 0x7F;
    let lsb = norm_bytes[3] & 1;
    row.nw_guard = F::from_u32(guard);
    row.nw2_lo7 = F::from_u32(nw2_lo7);
    row.nw_lsb = F::from_u32(lsb);
    row.nw3_hi7 = F::from_u32(norm_bytes[3] >> 1);
    if s_norm {
        range_checker.add_count(nw2_lo7, 7);
        range_checker.add_count(norm_bytes[3] >> 1, 7);
    }

    // Sticky: bits below guard.
    let below = nw2_lo7 + norm_bytes[1] + norm_bytes[0];
    row.below_guard = F::from_u32(below);
    let bg_nz = below != 0;
    row.below_guard_nz = F::from_bool(bg_nz);
    row.below_guard_inv = if bg_nz {
        F::from_u32(below).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // Effective sticky: on add/deep paths the alignment-lost bits count as
    // sticky; on the cancel path they are subtractive.
    let eff_sticky = bg_nz || (align_loss && !s_cancel);
    row.eff_sticky = F::from_bool(eff_sticky);
    let lsb_ok = lsb == 1 && !(s_cancel && align_loss);
    row.nw_lsb_ok = F::from_bool(lsb_ok);

    // round_up = guard AND (eff_sticky OR lsb_ok).
    let round_up = guard == 1 && (eff_sticky || lsb_ok);
    row.round_up = F::from_bool(round_up);

    // Mantissa carry chain.
    let s0 = norm_bytes[3] + round_up as u32;
    let m0 = s0 & 0xFF;
    let rc0 = s0 >> 8;
    let s1 = norm_bytes[4] + rc0;
    let m1 = s1 & 0xFF;
    let rc1 = s1 >> 8;
    let top7 = norm_bytes[5].saturating_sub(128);
    let top7c = top7 + rc1;
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    row.m0 = F::from_u32(m0);
    row.m1 = F::from_u32(m1);
    row.rc0 = F::from_u32(rc0);
    row.rc1 = F::from_u32(rc1);
    row.rc2 = F::from_u32(rc2);
    row.rc2_rem = F::from_u32(rc2_rem);
    if s_norm {
        range_checker.add_count(m0, 8);
        range_checker.add_count(m1, 8);
        range_checker.add_count(rc2_rem, 7);
    }

    if s_norm {
        let res_exp_i = base_exp_i + rc2 as i32;
        row.r_exp_lo = F::from_u32((res_exp_i as u32) & 1);
        row.r_exp_hi = F::from_u32((res_exp_i as u32) >> 1);
        // Sanity against the native result.
        debug_assert_eq!(
            (result >> 23) & 0xFF,
            res_exp_i as u32,
            "fma exponent mismatch: a*b+c result {result:08x}"
        );
        debug_assert_eq!(result & 0xFF, m0, "fma mantissa byte 0 mismatch");
        debug_assert_eq!((result >> 8) & 0xFF, m1, "fma mantissa byte 1 mismatch");
        debug_assert_eq!(
            (result >> 16) & 0x7F,
            rc2_rem,
            "fma mantissa byte 2 mismatch"
        );
    } else if is_ovf {
        debug_assert_eq!(
            result & 0x7FFF_FFFF,
            0x7F80_0000,
            "fma overflow result not inf: {result:08x}"
        );
        row.r_exp_lo = F::ZERO;
        row.r_exp_hi = F::ZERO;
    } else {
        row.r_exp_lo = F::ZERO;
        row.r_exp_hi = F::ZERO;
    }

    // ── Sub path: subnormal results ──────────────────────────────────
    if s_sub {
        let t = 25 - base_exp_i;
        debug_assert!((25..256).contains(&t), "fma sub_t out of range: {t}");
        let sub_deep = t > 55;
        let shift = if sub_deep { 55usize } else { t as usize };
        row.sub_t = F::from_u32(t as u32);
        range_checker.add_count(t as u32, 8);
        let lt8 = IsLtSubAir::new(range_checker.bus(), 8);
        lt8.generate_subrow(
            (range_checker.as_ref(), 55u32, t as u32),
            (&mut row.sub_deep_lt_decomp, &mut row.sub_deep),
        );
        debug_assert_eq!(row.sub_deep, F::from_bool(sub_deep));
        row.sub_shrn = make_shrn_cols7::<F>(norm_bytes, shift, range_checker);

        // Rounding from the gadget's guard/sticky + y[0] lsb.
        let y: [u32; 7] = core::array::from_fn(|i| row.sub_shrn.y[i].as_canonical_u32());
        let sub_lsb = y[0] & 1;
        row.sub_lsb = F::from_u32(sub_lsb);
        row.sub_hi7 = F::from_u32(y[0] >> 1);
        range_checker.add_count(y[0] >> 1, 7);
        let sub_eff = row.sub_shrn.sticky.as_canonical_u32() != 0 || (align_loss && !s_cancel);
        row.sub_eff_sticky = F::from_bool(sub_eff);
        let sub_lsb_ok = sub_lsb == 1 && !(s_cancel && align_loss);
        row.sub_lsb_ok = F::from_bool(sub_lsb_ok);
        let sub_round_up = row.sub_shrn.guard.as_canonical_u32() == 1 && (sub_eff || sub_lsb_ok);
        row.sub_round_up = F::from_bool(sub_round_up);

        // z = y + round_up (bit 23 of z is the exponent field LSB).
        let zc0 = (y[0] + sub_round_up as u32) >> 8;
        let z0 = (y[0] + sub_round_up as u32) & 0xFF;
        let zc1 = (y[1] + zc0) >> 8;
        let z1 = (y[1] + zc0) & 0xFF;
        let z2 = y[2] + zc1;
        let z = z0 | (z1 << 8) | (z2 << 16);
        debug_assert!(z <= 0x80_0000, "fma sub z overflow: {z}");
        row.z0 = F::from_u32(z0);
        row.z1 = F::from_u32(z1);
        row.z2 = F::from_u32(z2);
        row.zc0 = F::from_u32(zc0);
        row.zc1 = F::from_u32(zc1);
        range_checker.add_count(z0, 8);
        range_checker.add_count(z1, 8);
        range_checker.add_count(z2, 8);

        // Sanity against the native result.
        debug_assert_eq!(
            result & 0x7FFF_FFFF,
            z,
            "fma subnormal result mismatch: {result:08x} vs z={z:x}"
        );
        debug_assert_eq!(
            result >> 31,
            row.r_sign_eff.as_canonical_u32(),
            "fma subnormal sign mismatch"
        );
    }

    if is_exact_zero {
        debug_assert_eq!(result, 0, "fma exact cancellation must be +0");
    }

    let _ = (a_exp, b_exp, c_exp);
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

    fn run_op(a: u32, b: u32, c: u32, op: FmaddOp) -> NumericBuilder<F> {
        eval_row(&build_row(a, b, c, op))
    }

    fn eval_row(row: &FmaCoreCols<F>) -> NumericBuilder<F> {
        let air = FmaCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut builder = NumericBuilder::new(vec![]);
        air.eval_finite_product(&mut builder, row);
        builder
    }

    pub(crate) fn build_row(a: u32, b: u32, c: u32, op: FmaddOp) -> FmaCoreCols<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let mut row = FmaCoreCols::<F>::default();
        let result = fmadd_result(a, b, c, op);

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
        let a_sub = a_exp == 0 && a_mant != 0;
        let b_sub = b_exp == 0 && b_mant != 0;
        let c_sub = c_exp == 0 && c_mant != 0;
        let c_zero_v = c_exp == 0 && c_mant == 0;
        let a_zero = a_exp == 0 && a_mant == 0;
        let b_zero = b_exp == 0 && b_mant == 0;
        let s_finite =
            (a_normal || a_sub) && (b_normal || b_sub) && (c_normal || c_sub || c_zero_v);
        let s_pzero = a_zero || b_zero;
        row.s_finite = F::from_bool(s_finite);
        row.s_pzero = F::from_bool(s_pzero);
        row.a_normal = F::from_bool(a_normal);
        row.b_normal = F::from_bool(b_normal);
        row.c_normal = F::from_bool(c_normal);
        row.a_sub = F::from_bool(a_sub);
        row.b_sub = F::from_bool(b_sub);
        row.c_sub = F::from_bool(c_sub);
        row.c_zero = F::from_bool(c_zero_v);
        row.a_zero = F::from_bool(a_zero);
        row.b_zero = F::from_bool(b_zero);
        row.ab_normal = F::from_bool((a_normal || a_sub) && (b_normal || b_sub));
        row.c_finite_ok = F::from_bool(c_normal || c_sub || c_zero_v);

        let (neg_p, neg_c) = op.negations();
        let prod_sign_eff = (a_sign != b_sign) != neg_p;
        let c_sign_eff = (c_sign == 1) != neg_c;
        let sign_diff = prod_sign_eff != c_sign_eff;
        row.prod_sign_eff = F::from_bool(prod_sign_eff);
        row.c_sign_eff = F::from_bool(c_sign_eff);
        row.sign_diff = F::from_bool(sign_diff);
        row.madd_flag = F::from_bool(op == FmaddOp::Madd);
        row.msub_flag = F::from_bool(op == FmaddOp::Msub);
        row.nmsub_flag = F::from_bool(op == FmaddOp::Nmsub);
        row.nmadd_flag = F::from_bool(op == FmaddOp::Nmadd);

        if s_finite {
            fill_finite_product(
                &mut row,
                a_exp,
                b_exp,
                a_mant,
                b_mant,
                c_exp,
                c_mant,
                prod_sign_eff,
                c_sign_eff,
                sign_diff,
                result,
                &chip,
            );
        } else {
            fill_op_norm(
                a_exp,
                a_mant,
                &mut row.a_lshift,
                &mut row.a_shln,
                &mut row.a_y2_lo7,
                &chip,
            );
            fill_op_norm(
                b_exp,
                b_mant,
                &mut row.b_lshift,
                &mut row.b_shln,
                &mut row.b_y2_lo7,
                &chip,
            );
            fill_op_norm(
                c_exp,
                c_mant,
                &mut row.c_lshift,
                &mut row.c_shln,
                &mut row.c_y2_lo7,
                &chip,
            );
        }

        row
    }

    const OPS: [FmaddOp; 4] = [FmaddOp::Madd, FmaddOp::Msub, FmaddOp::Nmsub, FmaddOp::Nmadd];

    fn assert_ok(a: u32, b: u32, c: u32, op: FmaddOp) {
        let builder = run_op(a, b, c, op);
        assert!(
            builder.ok(),
            "fma {op:?} failed: {:?}\n a={a:08x} b={b:08x} c={c:08x}",
            builder.failures
        );
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
            // ediff=0 cancellation, c significand larger than product.
            (0x3F800000, 0x3F800000, 0xBFC00000), // 1*1 + (-1.5) = -0.5
            // ediff=0 cancellation, product larger.
            (0x40400000, 0x3F800000, 0xC0000000), // 3*1 + (-2) = 1
            // Exact cancellation -> +0.
            (0x3F800000, 0x40000000, 0xC0000000), // 1*2 + (-2) = +0
            (0x3FC00000, 0x3F800000, 0xBFC00000), // 1.5 - 1.5 = +0
            // c = +-0 with a small product exponent (phantom implicit bit).
            (0x25800000, 0x25800000, 0x00000000), // 2^-52 * 2^-52 + 0
            (0x25800000, 0x25800000, 0x80000000), // ... + (-0)
            // Zero operand: result is c (with effective sign).
            (0x00000000, 0x3F800000, 0x3F800000), // +0 * 1 + 1
            (0x80000000, 0x3F800000, 0x3F800000), // -0 * 1 + 1
            (0x00000000, 0x3F800000, 0x00000000), // +0 * 1 + +0
            (0x80000000, 0x3F800000, 0x80000000), // -0 * 1 + -0 = -0
            (0x00000000, 0x3F800000, 0x80000000), // +0 * 1 + -0 = +0
            (0x00000000, 0x3F800000, 0x00400000), // +0 * 1 + subnormal c
            // Overflow -> inf.
            (0x7F7FFFFF, 0x40000000, 0x00000000), // MAX * 2 + 0
            (0x7F7FFFFF, 0x3F800000, 0x7F7FFFFF), // MAX * 1 + MAX (add overflow)
            (0x7F7FFFFF, 0x3F800000, 0x73000000), // MAX + 2^103: tie -> rc2 carry
            // Cancellation with alignment sticky loss (fused single rounding).
            (0x3F801000, 0x3F801000, 0xBF800000), // (1+2^-12)^2 - 1
            // Deep alignment (ediff >= 56): result is the larger operand.
            (0x3F800000, 0x3F800000, 0x0C800000), // 1 + tiny
            (0x3F800000, 0x3F800000, 0x8C800000), // 1 - tiny
            (0x0C800000, 0x0C800000, 0x3F800000), // tiny + 1
            (0x0C800000, 0x0C800000, 0xBF800000), // tiny - 1
            // Subnormal operands and results.
            (0x00800000, 0x00800000, 0x00800000), // min_norm^2 + min_norm
            (0x00000001, 0x3F800000, 0x3F800000), // min_sub * 1 + 1
            (0x00000001, 0x3F800000, 0x80000001), // min_sub * 1 - min_sub = +0
            (0x00400000, 0x3F800000, 0x3F800000), // sub * 1 + 1 (deep)
            (0x007FFFFF, 0x007FFFFF, 0x00000000), // max_sub^2 + 0
            (0x00800000, 0x3F000000, 0x00000001), // min_norm * 0.5 + min_sub
            (0x3F800000, 0x3F800000, 0x80800000), // 1 - min_norm (subnormal?)
            (0x33800000, 0x33800000, 0x00000000), // 2^-24^2 + 0 (deep underflow)
        ] {
            for &op in &OPS {
                assert_ok(a, b, c, op);
            }
        }
        // Fuzz with random signs and opcodes, including subnormal/zero
        // operands and subnormal results — no deferred buckets remain.
        let mut rng = 0xF00D_BEEFu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for i in 0..3000 {
            let gen = |next: &mut dyn FnMut() -> u32| {
                if i % 4 == 0 {
                    // Include subnormals/zeros (but not inf/NaN: finite path only).
                    ((next() % 255) << 23) | (next() & 0x7FFFFF) | ((next() & 1) << 31)
                } else {
                    ((next() % 254) + 1) << 23 | (next() & 0x7FFFFF) | ((next() & 1) << 31)
                }
            };
            let a = gen(&mut next);
            let b = gen(&mut next);
            let c = gen(&mut next);
            let op = OPS[(i + next() as usize) % 4];
            assert_ok(a, b, c, op);
        }
        // Fuzz biased to tiny exponents for subnormal-result coverage.
        for i in 0..1000 {
            let a = ((next() % 40) + 1) << 23 | (next() & 0x7FFFFF) | ((next() & 1) << 31);
            let b = ((next() % 40) + 1) << 23 | (next() & 0x7FFFFF) | ((next() & 1) << 31);
            let c = ((next() % 40) + 1) << 23 | (next() & 0x7FFFFF) | ((next() & 1) << 31);
            let op = OPS[(i + next() as usize) % 4];
            assert_ok(a, b, c, op);
        }
    }

    /// Construct a cancellation tie: product aligned right by 1 with a lost
    /// half-ULP (sticky), landing exactly on a rounding tie. RNE must round
    /// DOWN (the lost bits are subtractive), not to-even.
    #[test]
    fn cancel_tie_rounds_down() {
        let mut rng = 0xCA4C_E15Eu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        let mut found = 0;
        for _ in 0..100_000 {
            let a_sig = 0x800000 | (next() & 0x7FFFFF);
            let b_sig = 0x800000 | (next() & 0x7FFFFF);
            let prod = (a_sig as u64) * (b_sig as u64);
            let is_high = (prod >> 47) & 1 == 1;
            let norm = if is_high { prod } else { prod << 1 };
            let low24 = norm & 0xFF_FFFF;
            let m = norm >> 24;
            // Want norm - Y = M*2^24 + 2^23 (exact tie, M odd) with
            // Y = c_sig >> 1 in [2^22, 2^23) and c_sig odd (sticky loss).
            if !(0xC0_0000..0x100_0000).contains(&low24) || m % 2 == 0 {
                continue;
            }
            let y = low24 - 0x80_0000;
            let c_sig = 2 * y + 1;
            debug_assert!((0x80_0000..0x100_0000).contains(&c_sig));
            // Exponents: prod_exp - c_exp = 25, c_exp in [1, 254].
            let a_exp = 1 + (next() % 100);
            let b_exp = 1 + (next() % 100);
            let prod_exp = a_exp as i32 + b_exp as i32 - 127 + is_high as i32;
            let c_exp = prod_exp - 25;
            if !(1..=254).contains(&c_exp) {
                continue;
            }
            let a = (a_exp << 23) | (a_sig & 0x7FFFFF);
            let b = (b_exp << 23) | (b_sig & 0x7FFFFF);
            let c = 0x8000_0000 | ((c_exp as u32) << 23) | ((c_sig as u32) & 0x7FFFFF);
            let r = fmadd_result(a, b, c, FmaddOp::Madd);
            if (r >> 23) & 0xFF == 0 {
                continue;
            }
            // The tie must round down: result mantissa == M, not M + 1.
            assert_eq!(
                r & 0x7F_FFFF,
                (m as u32) & 0x7F_FFFF,
                "expected tie to round down: a={a:08x} b={b:08x} c={c:08x} r={r:08x}"
            );
            assert_ok(a, b, c, FmaddOp::Madd);
            found += 1;
            if found >= 3 {
                break;
            }
        }
        assert!(found >= 3, "failed to construct tie cases");
    }

    #[test]
    fn negative_tests() {
        // Corrupting the result must fail.
        let mut row = build_row(
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            1.0f32.to_bits(),
            FmaddOp::Madd,
        );
        assert!(eval_row(&row).ok());
        row.result[0] += F::ONE;
        assert!(!eval_row(&row).ok(), "corrupted result accepted");

        // Flipping nw_lsb must fail (it feeds the RNE tie-break).
        let mut row = build_row(0x3F000001, 0x3F800000, 0x3E800000, FmaddOp::Madd);
        assert!(eval_row(&row).ok());
        row.nw_lsb = F::ONE - row.nw_lsb;
        assert!(!eval_row(&row).ok(), "flipped nw_lsb accepted");

        // Flipping round_up must fail.
        let mut row = build_row(
            1.5f32.to_bits(),
            2.5f32.to_bits(),
            0.5f32.to_bits(),
            FmaddOp::Madd,
        );
        row.round_up = F::ONE - row.round_up;
        assert!(!eval_row(&row).ok(), "flipped round_up accepted");

        // norm_work[6] != 0 must fail (anchors the normalization shift).
        let mut row = build_row(
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            1.0f32.to_bits(),
            FmaddOp::Madd,
        );
        row.norm_work[6] = F::ONE;
        assert!(!eval_row(&row).ok(), "norm_work[6]=1 accepted");

        // Claiming a normal result is overflow must fail.
        let mut row = build_row(
            3.0f32.to_bits(),
            4.0f32.to_bits(),
            1.0f32.to_bits(),
            FmaddOp::Madd,
        );
        row.is_ovf = F::ONE;
        assert!(!eval_row(&row).ok(), "fraudulent is_ovf accepted");
    }
}

#[cfg(test)]
mod corruption_sweep_tests {
    use super::*;
    use crate::test_utils::assert_columns_constrained;
    use openvm_circuit::arch::BasicAdapterInterface;
    use openvm_circuit::arch::MinimalInstruction;
    use openvm_circuit_primitives::var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    };
    use openvm_circuit_primitives::StructReflectionHelper;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;
    type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 3, 1, 4, 4>;

    fn rc() -> SharedVariableRangeCheckerChip {
        Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )))
    }

    fn air() -> FmaCoreAir {
        FmaCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET)
    }

    fn filler_cols(a: u32, b: u32, c: u32, op: FmaddOp) -> FmaCoreCols<F> {
        let chip = rc();
        let mut cols = FmaCoreCols::<F>::default();
        fill_fma_witness(&mut cols, a, b, c, op, fmadd_result(a, b, c, op), &chip);
        cols
    }

    /// Result bytes, routing selectors, and critical aux must reject +1.
    #[test]
    fn corruption_sweep_pins_result_and_selectors() {
        let names = FmaCoreCols::<F>::struct_reflection().unwrap_or_default();
        let eval_full = |b: &mut crate::test_utils::NumericBuilder<F>, cols: &[F]| {
            VmCoreAir::<crate::test_utils::NumericBuilder<F>, TestI>::eval(
                &air(),
                b,
                cols,
                F::ZERO,
            );
        };
        let classes: &[(u32, u32, u32, FmaddOp, &[&str])] = &[
            (
                3.0f32.to_bits(),
                4.0f32.to_bits(),
                1.0f32.to_bits(),
                FmaddOp::Madd,
                &["is_high", "round_up", "is_deep", "s_add", "madd_flag"][..],
            ),
            (
                0x3F800000,
                0x3F800000,
                0xBFC00000, // 1*1 + (-1.5) cancel
                FmaddOp::Madd,
                &["s_cancel", "madd_flag"][..],
            ),
            (
                0x7F7FFFFF,
                0x40000000,
                0x00000000, // MAX * 2 + 0 -> inf
                FmaddOp::Madd,
                &[][..],
            ),
            (
                0x7F800000,
                0x3F800000,
                0x3F800000, // inf * 1 + 1
                FmaddOp::Madd,
                &[][..],
            ),
            (
                0x7FC00000,
                0x3F800000,
                0x3F800000, // NaN
                FmaddOp::Madd,
                &[][..],
            ),
            (
                0x00000001,
                0x3F800000,
                0x3F800000, // min sub * 1 + 1
                FmaddOp::Madd,
                &[][..],
            ),
            (
                3.0f32.to_bits(),
                4.0f32.to_bits(),
                1.0f32.to_bits(),
                FmaddOp::Nmadd,
                &["nmadd_flag"][..],
            ),
        ];
        for &(a, b, c, op, selectors) in classes {
            let cols = filler_cols(a, b, c, op);
            let mut buf = vec![F::ZERO; FmaCoreCols::<F>::width()];
            *buf.as_mut_slice().borrow_mut() = cols;
            let mut bld = crate::test_utils::NumericBuilder::new(vec![]);
            VmCoreAir::<crate::test_utils::NumericBuilder<F>, TestI>::eval(
                &air(),
                &mut bld,
                &buf,
                F::ZERO,
            );
            assert!(
                bld.ok(),
                "honest fma {op:?} a={a:08x} b={b:08x} c={c:08x}: {:?}",
                bld.failures
            );
            let mut pinned = vec!["result__0", "result__1", "result__2", "result__3"];
            pinned.extend_from_slice(selectors);
            assert_columns_constrained(
                &buf,
                eval_full,
                &names,
                &pinned,
                &format!("fma {op:?} a={a:08x} b={b:08x} c={c:08x}"),
            );
        }
    }
}
