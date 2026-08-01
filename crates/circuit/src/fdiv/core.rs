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
    fill_op_norm, run_shrn, run_shrn_round, ShlnCols, ShlnGadget, ShrnCols, ShrnGadget,
};

/// IEEE 754 / RISC-V FDIV.S: correctly-rounded (RNE) division.
/// Per the RISC-V spec, the canonical NaN is returned when any input is NaN,
/// or the operation is invalid (0 / 0, inf / inf).
#[inline(always)]
pub fn fdiv_s(a: u32, b: u32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    if fa.is_nan() || fb.is_nan() {
        return CANONICAL_NAN;
    }
    if fa.is_infinite() && fb.is_infinite() {
        return CANONICAL_NAN; // inf / inf
    }
    if fa == 0.0 && fb == 0.0 {
        return CANONICAL_NAN; // 0 / 0
    }
    let r = fa / fb;
    debug_assert!(!r.is_nan(), "all NaN cases handled above");
    r.to_bits()
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FdivCoreCols<T> {
    pub a: [T; 4],
    pub b: [T; 4],

    // === Decomposition of inputs ===
    pub a_sign: T,
    pub b_sign: T,
    pub a_exp_lo: T,
    pub b_exp_lo: T,
    pub a_mant2: T,
    pub b_mant2: T,
    pub a_hi: T,
    pub b_hi: T,

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
    pub a_nan: T,
    pub b_nan: T,
    pub a_inf: T,
    pub b_inf: T,
    pub a_zero: T,
    pub b_zero: T,
    pub any_nan: T,

    // === Special-case selection ===
    pub canon_sel: T, // canonical NaN: any_nan | (a_inf && b_inf) | (a_zero && b_zero)
    pub inf_res: T,   // +-inf: (a_inf && !b_inf) || (b_zero && !a_inf), excluding NaN
    pub zero_res: T,  // +-0: b_inf && !a_inf, excluding NaN
    pub is_special: T,
    pub r_sign: T, // a_sign ^ b_sign

    // === Finite division (long-division verification) ===
    pub a_sig: [T; 3], // a significand (mant | implicit)
    pub b_sig: [T; 3], // b significand (divisor)
    pub a_sub: T,      // a is subnormal (exp 0, mantissa nonzero)
    pub b_sub: T,      // b is subnormal
    pub a_lshift: T,   // leading-zeros normalization shift for a
    pub b_lshift: T,   // ... and b
    pub a_shln: ShlnCols<T, 3>,
    pub b_shln: ShlnCols<T, 3>,
    pub a_y2_lo7: T,
    pub b_y2_lo7: T,
    pub a_sig_hi: [T; 3],  // bit 7 of each a_sig byte (for a_sig << 25 target)
    pub q: [T; 4],         // quotient bytes (a_sig*2^25 / b_sig)
    pub r: [T; 3],         // remainder bytes (< b_sig)
    pub div_carry: [T; 6], // carry chain for q*b_sig + r = a_sig<<25
    pub lt_borrow: [T; 3], // borrow chain for r < b_sig verification
    pub diff: [T; 3],      // b_sig - r - 1 byte decomposition (proves r < b_sig)
    pub q_lo2: [T; 4],     // q[i] & 3 (for is_high mantissa extraction)
    pub q_hi6: [T; 4],     // q[i] >> 2 (for is_high mantissa extraction)
    pub lo_bit0: [T; 4],   // q_lo2[i] & 1 (for !is_high extraction)
    pub lo_bit1: [T; 4],   // q_lo2[i] >> 1 (for !is_high extraction)
    pub is_high: T,        // bool: q >= 2^25 (a_sig >= b_sig)
    pub qh6_lo: T,         // q_hi6[0] & 1 (is_high mantissa LSB)
    pub qh6_rest: T,       // q_hi6[0] >> 1
    pub guard: T,
    pub lsb: T,
    pub sticky: T,
    pub r_nz: T, // bool: r != 0
    pub r_nz_inv: T,
    pub round_up: T,
    pub rc0: T,
    pub rc1: T,
    pub rc2: T,
    pub rc2_rem: T,
    pub r_exp_lo: T,
    pub r_exp_hi: T,
    pub result_exp: T,
    pub a_normal: T,
    pub b_normal: T, // both-normal indicators (degree 1)
    pub s_finite: T,
    pub s_normal: T,
    pub s_overflow: T,
    pub s_subnormal: T,
    pub route_ovf_decomp: [T; 1],
    pub route_sub_decomp: [T; 1],
    pub sub_capped: T,
    pub sub_lt_decomp: [T; 1],
    pub sub_shift: T,
    pub sub_shrn: ShrnCols<T, 7>,
    pub sub_y0_high7: T,
    pub sub_lsb: T,
    pub sub_sticky_eff: T,
    pub sub_round_up: T,
    pub sub_rc0: T,
    pub sub_rc1: T,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FdivCoreCols<u8>)]
pub struct FdivCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FdivCoreAir {
    fn width(&self) -> usize {
        FdivCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FdivCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FdivCoreAir
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
        let cols: &FdivCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        for flag in [
            cols.a_sign,
            cols.b_sign,
            cols.a_exp_lo,
            cols.b_exp_lo,
            cols.a_exp_is_zero,
            cols.b_exp_is_zero,
            cols.a_exp_is_max,
            cols.b_exp_is_max,
            cols.a_mant_is_zero,
            cols.b_mant_is_zero,
            cols.a_nan,
            cols.b_nan,
            cols.a_inf,
            cols.b_inf,
            cols.a_zero,
            cols.b_zero,
            cols.any_nan,
            cols.canon_sel,
            cols.inf_res,
            cols.zero_res,
            cols.is_special,
            cols.r_sign,
        ] {
            builder.assert_bool(flag);
        }

        // ── Byte decomposition ───────────────────────────────────────────
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        builder.assert_eq(cols.b[2], cols.b_mant2 + cols.b_exp_lo * f128);
        builder.assert_eq(cols.b[3], cols.b_hi + cols.b_sign * f128);
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // exp_lo / sign are free field choices (classification re-routing).
        for v in [cols.a_mant2, cols.b_mant2, cols.a_hi, cols.b_hi] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

        // Range check input bytes (bounds exponent/mantissa sums).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.b[i].into(), 8)
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
                            zero: AB::Var| {
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
            builder.assert_eq(zero, exp_is_zero * mant_is_zero);
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
            cols.a_zero,
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
            cols.b_zero,
        );

        // ── Special-case selection ───────────────────────────────────────
        builder.assert_eq(
            cols.any_nan,
            cols.a_nan + cols.b_nan - cols.a_nan * cols.b_nan,
        );
        builder.assert_eq(
            cols.canon_sel,
            cols.any_nan + cols.a_inf * cols.b_inf + cols.a_zero * cols.b_zero,
        );
        // +-inf result: a infinite (b finite or zero), or finite-nonzero / zero.
        // (a_inf and a_zero are mutually exclusive, so (1-a_inf)(1-a_zero)
        //  = 1 - a_inf - a_zero.)
        builder.assert_eq(
            cols.inf_res,
            (one.clone() - cols.any_nan)
                * (cols.a_inf * (one.clone() - cols.b_inf)
                    + cols.b_zero * (one.clone() - cols.a_inf - cols.a_zero)),
        );
        // +-0 result: finite / inf, or +-0 / finite-nonzero.
        builder.assert_eq(
            cols.zero_res,
            (one.clone() - cols.any_nan)
                * (cols.b_inf * (one.clone() - cols.a_inf)
                    + cols.a_zero * (one.clone() - cols.b_zero) * (one.clone() - cols.b_inf)),
        );
        builder.assert_eq(
            cols.is_special,
            cols.canon_sel + cols.inf_res + cols.zero_res,
        );
        builder.assert_eq(
            cols.r_sign,
            cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2,
        );

        // Canonical NaN for NaN inputs and invalid operations.
        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &canon_byte) in canon.iter().enumerate() {
            builder
                .when(cols.canon_sel)
                .assert_eq(cols.result[i], AB::F::from_u32(canon_byte as u32));
        }
        // +-inf result.
        builder.when(cols.inf_res).assert_zero(cols.result[0]);
        builder.when(cols.inf_res).assert_zero(cols.result[1]);
        builder
            .when(cols.inf_res)
            .assert_eq(cols.result[2], AB::F::from_u32(128));
        builder.when(cols.inf_res).assert_eq(
            cols.result[3],
            AB::Expr::from(AB::F::from_u32(127)) + cols.r_sign * AB::F::from_u32(128),
        );
        // +-0 result.
        builder.when(cols.zero_res).assert_zero(cols.result[0]);
        builder.when(cols.zero_res).assert_zero(cols.result[1]);
        builder.when(cols.zero_res).assert_zero(cols.result[2]);
        builder
            .when(cols.zero_res)
            .assert_eq(cols.result[3], cols.r_sign * AB::F::from_u32(128));

        // ── Finite result: long-division verification ───────────────────
        self.eval_finite_div(builder, cols);
        self.eval_finite_subnormal(builder, cols);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FDIV_S);

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

impl FdivCoreAir {
    /// Constrain the finite division result via long-division verification.
    ///
    /// Verifies `q * b_sig + r = a_sig << 25` (product carry chain), `0 <= r < b_sig`
    /// (borrow chain), then extracts the correctly-rounded mantissa from q and r.
    /// Handles normal, overflow, and subnormal/underflow results; subnormal
    /// operands are pre-normalized before the long division.
    pub fn eval_finite_div<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FdivCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f4: AB::F = AB::F::from_u32(4);
        let f64: AB::F = AB::F::from_u32(64);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;

        // ── Sound selector: finite AND both operands nonzero finite ─────
        // (subnormal operands are pre-normalized below; b=0 is special-cased).
        let not_special = one.clone() - cols.is_special;
        let a_normal_expr = (one.clone() - cols.a_exp_is_zero) * (one.clone() - cols.a_exp_is_max);
        let b_normal_expr = (one.clone() - cols.b_exp_is_zero) * (one.clone() - cols.b_exp_is_max);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_normal, a_normal_expr);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.b_normal, b_normal_expr);
        // Subnormal: exponent field 0 with nonzero mantissa.
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
        builder.when(cols.is_valid).assert_eq(
            cols.s_finite,
            not_special.clone() * (cols.a_normal + cols.a_sub) * (cols.b_normal + cols.b_sub),
        );
        builder.assert_bool(cols.s_finite);
        let s = cols.s_finite;

        // ── Subnormal operand pre-normalization ─────────────────────────
        // Shift the subnormal significand left so its MSB lands at bit 23
        // (the implicit-bit position); the exponent absorbs the shift.
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

        // ── Significand construction ────────────────────────────────────
        // Normal: mantissa + implicit bit. Subnormal: normalized significand.
        let a_ns = one.clone() - cols.a_sub;
        let b_ns = one.clone() - cols.b_sub;
        builder.when(s).assert_eq(
            cols.a_sig[0],
            a_ns.clone() * cols.a[0] + cols.a_sub * cols.a_shln.y[0],
        );
        builder.when(s).assert_eq(
            cols.a_sig[1],
            a_ns.clone() * cols.a[1] + cols.a_sub * cols.a_shln.y[1],
        );
        builder.when(s).assert_eq(
            cols.a_sig[2],
            a_ns * (cols.a_mant2 + f128) + cols.a_sub * cols.a_shln.y[2],
        );
        builder.when(s).assert_eq(
            cols.b_sig[0],
            b_ns.clone() * cols.b[0] + cols.b_sub * cols.b_shln.y[0],
        );
        builder.when(s).assert_eq(
            cols.b_sig[1],
            b_ns.clone() * cols.b[1] + cols.b_sub * cols.b_shln.y[1],
        );
        builder.when(s).assert_eq(
            cols.b_sig[2],
            b_ns * (cols.b_mant2 + f128) + cols.b_sub * cols.b_shln.y[2],
        );

        // a_sig bit-7 decomposition (for a_sig << 25 target).
        for i in 0..3 {
            builder.assert_bool(cols.a_sig_hi[i]);
            let lo7 = cols.a_sig[i] - cols.a_sig_hi[i] * f128;
            builder
                .when(s)
                .assert_eq(cols.a_sig[i], lo7 + cols.a_sig_hi[i] * f128);
            self.range_bus
                .range_check(cols.a_sig[i].into(), 8)
                .eval(builder, s.into());
        }
        for i in 0..3 {
            self.range_bus
                .range_check(cols.b_sig[i].into(), 8)
                .eval(builder, s.into());
        }
        for i in 0..4 {
            self.range_bus
                .range_check(cols.q[i].into(), 8)
                .eval(builder, s.into());
        }
        for i in 0..3 {
            self.range_bus
                .range_check(cols.r[i].into(), 8)
                .eval(builder, s.into());
        }

        // ── Product carry chain: q * b_sig + r = a_sig << 25 ────────────
        // target bytes for a_sig << 25:
        //   target[0..2] = 0
        //   target[3] = lo7_a0 * 2
        //   target[4] = a_sig_hi[0] + lo7_a1 * 2
        //   target[5] = a_sig_hi[1] + lo7_a2 * 2
        //   target[6] = a_sig_hi[2]
        let lo7_a = |i: usize| cols.a_sig[i] - cols.a_sig_hi[i] * f128;
        let target = |i: usize| -> AB::Expr {
            match i {
                0..=2 => AB::Expr::ZERO,
                3 => lo7_a(0) * f2,
                4 => cols.a_sig_hi[0].into() + lo7_a(1) * f2,
                5 => cols.a_sig_hi[1].into() + lo7_a(2) * f2,
                6 => cols.a_sig_hi[2].into(),
                _ => unreachable!(),
            }
        };
        let carry_divide = AB::F::from_u32(256).inverse();
        for i in 0..7 {
            // Convolution: sum of q[k] * b_sig[j] for k+j = i.
            let mut conv = AB::Expr::ZERO;
            for k in 0..4usize {
                let j = i as isize - k as isize;
                if j >= 0 && (j as usize) < 3 {
                    conv += cols.q[k] * cols.b_sig[j as usize];
                }
            }
            let r_byte = if i < 3 {
                cols.r[i].into()
            } else {
                AB::Expr::ZERO
            };
            let carry_in = if i == 0 {
                AB::Expr::ZERO
            } else {
                cols.div_carry[i - 1].into()
            };
            let expected = carry_in + conv + r_byte;
            // expected = target[i] + carry_out * 256
            let carry_out = (expected.clone() - target(i)) * carry_divide;
            if i < 6 {
                builder.when(s).assert_eq(cols.div_carry[i], carry_out);
                self.range_bus
                    .range_check(cols.div_carry[i].into(), 10)
                    .eval(builder, s.into());
            } else {
                // Final carry must be 0 (a_sig << 25 fits in 49 bits = 7 bytes).
                builder.when(s).assert_zero(carry_out);
            }
        }

        // ── Remainder range check: r < b_sig ────────────────────────────
        // Verify b_sig - r - 1 >= 0 via a borrow chain with diff bytes.
        for i in 0..3 {
            builder.assert_bool(cols.lt_borrow[i]);
            self.range_bus
                .range_check(cols.diff[i].into(), 8)
                .eval(builder, s.into());
        }
        // byte 0: b_sig[0] - r[0] - 1 + borrow_out*256 = diff[0]
        builder.when(s).assert_eq(
            cols.b_sig[0] - cols.r[0] - AB::Expr::ONE + cols.lt_borrow[0] * f256,
            cols.diff[0].into(),
        );
        // bytes 1..2: b_sig[i] - r[i] - borrow_in + borrow_out*256 = diff[i]
        for i in 1..3 {
            builder.when(s).assert_eq(
                cols.b_sig[i] - cols.r[i] - cols.lt_borrow[i - 1] + cols.lt_borrow[i] * f256,
                cols.diff[i].into(),
            );
        }
        // Final borrow must be 0 (b_sig > r).
        builder.when(s).assert_zero(cols.lt_borrow[2]);

        // ── Quotient decomposition for mantissa extraction ──────────────
        // q[i] = q_lo2[i] + q_hi6[i] * 4; q_lo2[i] = lo_bit0[i] + lo_bit1[i] * 2.
        for i in 0..4 {
            builder
                .when(s)
                .assert_eq(cols.q[i], cols.q_lo2[i] + cols.q_hi6[i] * f4);
            self.range_bus
                .range_check(cols.q_lo2[i].into(), 2)
                .eval(builder, s.into());
            self.range_bus
                .range_check(cols.q_hi6[i].into(), 6)
                .eval(builder, s.into());
            builder.assert_bool(cols.lo_bit0[i]);
            builder.assert_bool(cols.lo_bit1[i]);
            builder
                .when(s)
                .assert_eq(cols.q_lo2[i], cols.lo_bit0[i] + cols.lo_bit1[i] * f2);
        }

        // is_high = (q >= 2^25) = q_hi6[3] >= 2 = lo_bit1[3] (bit 1 of q3).
        // Actually q3 = q_lo2[3] + q_hi6[3]*4. Bit 25 of q = bit 1 of q3 = lo_bit1[3]... wait.
        // q3 byte has bits [31..24] of q. Bit 25 = bit 1 of q3.
        // q3 = lo_bit0[3] + lo_bit1[3]*2 + q_hi6[3]*4.
        // Bit 1 of q3 = lo_bit1[3]. Bit 0 of q3 = lo_bit0[3]. Bits 2..7 = q_hi6[3].
        // is_high = bit 25 set = lo_bit1[3] (since q3 occupies bits 24..31).
        // Wait: bit 24 = q3 bit 0 = lo_bit0[3]. bit 25 = q3 bit 1 = lo_bit1[3].
        // For is_high, bit 25 is the leading bit. So is_high = lo_bit1[3].
        // But q3 < 64 (since q < 2^26), so q_hi6[3] < 16. And lo_bit1[3] is bit 1 of q3.
        // For q >= 2^25: q3 >= 2, i.e., bit 1 of q3 is set OR q_hi6[3] >= 1.
        // Actually: q >= 2^25 means bit 25 is set. Bit 25 = q3 bit 1 = lo_bit1[3].
        // So is_high = lo_bit1[3]. ✓
        builder.assert_bool(cols.is_high);
        builder.when(s).assert_eq(cols.is_high, cols.lo_bit1[3]);

        // ── RNE rounding + mantissa extraction ──────────────────────────
        // q_hi6[0] bit decomposition (for is_high mantissa LSB).
        builder.assert_bool(cols.qh6_lo);
        builder
            .when(s)
            .assert_eq(cols.q_hi6[0], cols.qh6_lo + cols.qh6_rest * f2);
        self.range_bus
            .range_check(cols.qh6_rest.into(), 5)
            .eval(builder, s.into());

        // r_nz = (r != 0).
        let r_sum = cols.r[0] + cols.r[1] + cols.r[2];
        let r_is_zero = one.clone() - r_sum.clone() * cols.r_nz_inv;
        builder.assert_zero(r_sum * r_is_zero.clone());
        builder
            .when(s)
            .assert_eq(cols.r_nz, one.clone() - r_is_zero);
        builder.assert_bool(cols.r_nz);

        // Guard: is_high → lo_bit1[0]; !is_high → lo_bit0[0].
        builder
            .when(s * cols.is_high)
            .assert_eq(cols.guard, cols.lo_bit1[0]);
        builder
            .when(s * (one.clone() - cols.is_high))
            .assert_eq(cols.guard, cols.lo_bit0[0]);
        builder.assert_bool(cols.guard);

        // Lsb: is_high → qh6_lo; !is_high → lo_bit1[0].
        builder
            .when(s * cols.is_high)
            .assert_eq(cols.lsb, cols.qh6_lo);
        builder
            .when(s * (one.clone() - cols.is_high))
            .assert_eq(cols.lsb, cols.lo_bit1[0]);
        builder.assert_bool(cols.lsb);

        // Sticky: is_high → lo_bit0[0] OR r_nz; !is_high → r_nz. Stored as column.
        let sticky_ih = cols.lo_bit0[0] + cols.r_nz - cols.lo_bit0[0] * cols.r_nz;
        builder
            .when(s * cols.is_high)
            .assert_eq(cols.sticky, sticky_ih);
        builder
            .when(s * (one.clone() - cols.is_high))
            .assert_eq(cols.sticky, cols.r_nz);
        builder.assert_bool(cols.sticky);

        // round_up = guard AND (sticky OR lsb).
        let sticky_or_lsb = cols.sticky + cols.lsb - cols.sticky * cols.lsb;
        builder
            .when(s)
            .assert_eq(cols.round_up, cols.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Mantissa extraction + carry chain ───────────────────────────
        // is_high: mant0 = q_hi6[0] + q_lo2[1]*64, mant1 = q_hi6[1] + q_lo2[2]*64,
        //          top7 = q_hi6[2] + lo_bit0[3]*64.
        // !is_high: mant0 = lo_bit1[0] + q_hi6[0]*2 + lo_bit0[1]*128,
        //           mant1 = lo_bit1[1] + q_hi6[1]*2 + lo_bit0[2]*128,
        //           top7 = lo_bit1[2] + q_hi6[2]*2.
        let ih = cols.is_high;
        let nih = one.clone() - ih;
        let f64e = AB::Expr::from(f64);

        let mant0 = ih * (cols.q_hi6[0] + cols.q_lo2[1] * f64e.clone())
            + nih.clone() * (cols.lo_bit1[0] + cols.q_hi6[0] * f2 + cols.lo_bit0[1] * f128);
        let mant1 = ih * (cols.q_hi6[1] + cols.q_lo2[2] * f64e.clone())
            + nih.clone() * (cols.lo_bit1[1] + cols.q_hi6[1] * f2 + cols.lo_bit0[2] * f128);
        let top7 = ih * (cols.q_hi6[2] + cols.lo_bit0[3] * f64e)
            + nih * (cols.lo_bit1[2] + cols.q_hi6[2] * f2);

        // Carry chain: mant0 + round_up → rc0; mant1 + rc0 → rc1; top7 + rc1 → rc2/rem.
        builder.assert_bool(cols.rc0);
        builder.assert_bool(cols.rc1);
        builder.assert_bool(cols.rc2);
        builder.when(cols.s_normal).assert_eq(
            cols.result[0] + cols.rc0 * f256,
            mant0.clone() + cols.round_up,
        );
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[1] + cols.rc1 * f256, mant1.clone() + cols.rc0);
        builder
            .when(s)
            .assert_eq(top7.clone() + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s.into());
        // Result exponent: a_exp_eff - b_exp_eff + 126 + is_high + rc2, where
        // the effective exponents absorb the subnormal normalization shift.
        // The pre-rounding exponent base_exp (without rc2) routes the result
        // class via less-than gadgets, so the class can never be prover-chosen.
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp.clone());
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp.clone());
        let base_exp = a_exp_eff - b_exp_eff + AB::Expr::from(AB::F::from_u32(126)) + cols.is_high;
        let result_exp_expr = base_exp.clone() + cols.rc2;
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);
        builder
            .when(cols.s_normal)
            .assert_eq(cols.r_exp_lo + cols.r_exp_hi * f2, result_exp_expr);

        // Result sign = a_sign ^ b_sign (r_sign). Only for normal results.
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.r_sign * f128);

        // ── Result-class routing (sound, via base_exp) ───────────────────
        // base_exp in [-127, 380]; base_exp_b = base_exp + 256 fits 10 bits.
        // s_overflow: base_exp + rc2 >= 255  <=>  510 < base_exp_b + rc2.
        // s_subnormal: base_exp <= 0  <=>  base_exp_b < 257.
        let base_exp_b = base_exp + AB::Expr::from(f256);
        self.range_bus
            .range_check(base_exp_b.clone(), 10)
            .eval(builder, s.into());
        let lt10 = IsLtSubAir::new(self.range_bus, 10);
        let ovf_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(510)),
            base_exp_b.clone() + cols.rc2,
            cols.s_overflow.into(),
            s.into(),
        );
        lt10.eval(builder, (ovf_io, &cols.route_ovf_decomp));
        let sub_io = IsLessThanIo::new(
            base_exp_b,
            AB::Expr::from(AB::F::from_u32(257)),
            cols.s_subnormal.into(),
            s.into(),
        );
        lt10.eval(builder, (sub_io, &cols.route_sub_decomp));
        builder.assert_bool(cols.s_overflow);
        builder.assert_bool(cols.s_subnormal);
        builder.assert_eq(
            cols.s_normal,
            s * (one.clone() - cols.s_overflow) * (one.clone() - cols.s_subnormal),
        );
        builder.assert_bool(cols.s_normal);

        // Normal result: tie result bytes (redundant with above for s_normal ⊆ s).
        builder
            .when(cols.s_normal)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        // Overflow result: +-inf.
        builder.when(cols.s_overflow).assert_zero(cols.result[0]);
        builder.when(cols.s_overflow).assert_zero(cols.result[1]);
        builder
            .when(cols.s_overflow)
            .assert_eq(cols.result[2], f128);
        builder.when(cols.s_overflow).assert_eq(
            cols.result[3],
            AB::Expr::from(AB::F::from_u32(127)) + cols.r_sign * f128,
        );
    }

    /// Constrain the subnormal / underflow result (result exponent field == 0).
    ///
    /// The quotient q (26-bit) is extended to 7 bytes and right-shifted by
    /// `cshift = min(b_exp - a_exp - 124, 55)` via `ShrnGadget<7>`. The low
    /// 23 bits of the shifted result become the subnormal mantissa, with RNE
    /// rounding from the gadget's guard/sticky + the mantissa LSB. If rounding
    /// carries into bit 23, the result becomes the smallest normal (exp field 1).
    pub fn eval_finite_subnormal<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FdivCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f55: AB::F = AB::F::from_u32(55);
        let f124: AB::F = AB::F::from_u32(124);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;

        // s_subnormal is routed soundly in eval_finite_div (base_exp <= 0).
        let s = cols.s_subnormal;

        // Shift = b_exp_eff - a_exp_eff - 124 (>= 2 on subnormal rows), using
        // the effective (subnormal-adjusted) exponents.
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp);
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp);
        let shift = b_exp_eff - a_exp_eff - AB::Expr::from(f124);

        // sub_capped = (shift >= 56) = (55 < shift).
        let lt_air = IsLtSubAir::new(self.range_bus, 8);
        let io = IsLessThanIo::new(
            AB::Expr::from(f55),
            shift.clone(),
            cols.sub_capped.into(),
            s.into(),
        );
        lt_air.eval(builder, (io, &cols.sub_lt_decomp));
        builder.assert_bool(cols.sub_capped);

        // cshift = capped ? 55 : shift.
        builder
            .when(s * (one.clone() - cols.sub_capped))
            .assert_eq(cols.sub_shift, shift.clone());
        builder
            .when(s * cols.sub_capped)
            .assert_eq(cols.sub_shift, AB::Expr::from(f55));
        self.range_bus
            .range_check(cols.sub_shift.into(), 6)
            .eval(builder, s.into());

        // ShrnGadget<7>: x = q extended to 7 bytes, y = x >> cshift.
        for i in 0..4 {
            builder.when(s).assert_eq(cols.sub_shrn.x[i], cols.q[i]);
        }
        for i in 4..7 {
            builder.when(s).assert_zero(cols.sub_shrn.x[i]);
        }
        builder
            .when(s)
            .assert_eq(cols.sub_shrn.shift, cols.sub_shift);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.sub_shrn);
        builder.assert_eq(cols.sub_shrn.enable, s);

        // Since shift >= 3 and q < 2^26, y < 2^23, so bytes [3..7] are zero.
        for i in 3..7 {
            builder.when(s).assert_zero(cols.sub_shrn.y[i]);
        }

        // RNE rounding. The sticky must include the division remainder: any
        // nonzero r means there are magnitude bits below the LSB of q.
        builder
            .when(s)
            .assert_eq(cols.sub_shrn.y[0], cols.sub_y0_high7 * f2 + cols.sub_lsb);
        builder.assert_bool(cols.sub_lsb);
        self.range_bus
            .range_check(cols.sub_y0_high7.into(), 7)
            .eval(builder, s.into());
        builder.assert_bool(cols.sub_shrn.guard);
        builder.assert_bool(cols.sub_shrn.sticky);
        builder.when(s).assert_eq(
            cols.sub_sticky_eff,
            cols.sub_shrn.sticky + cols.r_nz - cols.sub_shrn.sticky * cols.r_nz,
        );
        builder.assert_bool(cols.sub_sticky_eff);
        let sticky_or_lsb = cols.sub_sticky_eff + cols.sub_lsb - cols.sub_sticky_eff * cols.sub_lsb;
        builder
            .when(s)
            .assert_eq(cols.sub_round_up, cols.sub_shrn.guard * sticky_or_lsb);
        builder.assert_bool(cols.sub_round_up);

        // Output reconstruction.
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
        builder
            .when(s)
            .assert_eq(cols.result[2], cols.sub_shrn.y[2] + cols.sub_rc1);
        builder
            .when(s)
            .assert_eq(cols.result[3], cols.r_sign * f128);
        builder.when(s).assert_eq(cols.r_exp_hi, AB::Expr::ZERO);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FdivCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FdivExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FdivExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FdivCoreRecord),
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
        debug_assert_eq!(FpOpcode::from_usize(local), FpOpcode::FDIV_S);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.b = b;

        let result = fdiv_s(u32::from_le_bytes(a), u32::from_le_bytes(b));
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

pub struct FdivFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FdivFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FdivFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FdivCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FdivCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let result = fdiv_s(a_bits, b_bits);

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((b_bits >> (i * 8)) & 0xFF, 8);
        }

        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);

        let classify = |exp: u32,
                        mant: u32,
                        exp_zero_inv: &mut F,
                        exp_max_inv: &mut F,
                        mant_sum_inv: &mut F,
                        exp_is_zero: &mut F,
                        exp_is_max: &mut F,
                        mant_is_zero: &mut F| {
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
        };

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

        classify(
            a_exp,
            a_mant,
            &mut core_row.a_exp_zero_inv,
            &mut core_row.a_exp_max_inv,
            &mut core_row.a_mant_sum_inv,
            &mut core_row.a_exp_is_zero,
            &mut core_row.a_exp_is_max,
            &mut core_row.a_mant_is_zero,
        );
        classify(
            b_exp,
            b_mant,
            &mut core_row.b_exp_zero_inv,
            &mut core_row.b_exp_max_inv,
            &mut core_row.b_mant_sum_inv,
            &mut core_row.b_exp_is_zero,
            &mut core_row.b_exp_is_max,
            &mut core_row.b_mant_is_zero,
        );

        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        let a_inf = a_exp == 255 && a_mant == 0;
        let b_inf = b_exp == 255 && b_mant == 0;
        let a_zero = a_exp == 0 && a_mant == 0;
        let b_zero = b_exp == 0 && b_mant == 0;
        let any_nan = a_nan || b_nan;
        let canon_sel = any_nan || (a_inf && b_inf) || (a_zero && b_zero);
        let inf_res = !any_nan && ((a_inf && !b_inf) || (b_zero && !a_inf && !a_zero));
        let zero_res = !any_nan && ((b_inf && !a_inf) || (a_zero && !b_zero && !b_inf));

        core_row.a_nan = F::from_bool(a_nan);
        core_row.b_nan = F::from_bool(b_nan);
        core_row.a_inf = F::from_bool(a_inf);
        core_row.b_inf = F::from_bool(b_inf);
        core_row.a_zero = F::from_bool(a_zero);
        core_row.b_zero = F::from_bool(b_zero);
        core_row.any_nan = F::from_bool(any_nan);
        core_row.canon_sel = F::from_bool(canon_sel);
        core_row.inf_res = F::from_bool(inf_res);
        core_row.zero_res = F::from_bool(zero_res);
        core_row.is_special = F::from_bool(canon_sel || inf_res || zero_res);
        core_row.r_sign = F::from_bool(a_sign != b_sign);

        // ── Finite division witness ──────────────────────────────────────
        let a_normal = (1..=254).contains(&a_exp);
        let b_normal = (1..=254).contains(&b_exp);
        let a_sub = a_exp == 0 && a_mant != 0;
        let b_sub = b_exp == 0 && b_mant != 0;
        let s_finite =
            !canon_sel && !inf_res && !zero_res && (a_normal || a_sub) && (b_normal || b_sub);
        core_row.s_finite = F::from_bool(s_finite);
        core_row.a_normal = F::from_bool(a_normal);
        core_row.b_normal = F::from_bool(b_normal);
        core_row.a_sub = F::from_bool(a_sub);
        core_row.b_sub = F::from_bool(b_sub);

        // Set r_exp_lo/r_exp_hi from the result for ALL rows (fill_finite_div
        // will override with identical values for finite rows).
        let r_exp_field = (result >> 23) & 0xFF;
        core_row.r_exp_lo = F::from_u32(r_exp_field & 1);
        core_row.r_exp_hi = F::from_u32(r_exp_field >> 1);

        if s_finite {
            fill_finite_div(
                core_row,
                a_exp,
                b_exp,
                a_mant,
                b_mant,
                result,
                &self.range_checker,
            );
        } else {
            zero_finite_columns(core_row);
            // Operand normalization gadgets stay enabled on subnormal inputs
            // regardless of the path — always fill them.
            fill_op_norm(
                a_exp,
                a_mant,
                &mut core_row.a_lshift,
                &mut core_row.a_shln,
                &mut core_row.a_y2_lo7,
                &self.range_checker,
            );
            fill_op_norm(
                b_exp,
                b_mant,
                &mut core_row.b_lshift,
                &mut core_row.b_shln,
                &mut core_row.b_y2_lo7,
                &self.range_checker,
            );
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
fn zero_finite_columns<F: PrimeField32>(row: &mut FdivCoreCols<F>) {
    row.a_lshift = F::ZERO;
    row.b_lshift = F::ZERO;
    row.a_shln = ShlnCols::<F, 3>::default();
    row.b_shln = ShlnCols::<F, 3>::default();
    row.a_y2_lo7 = F::ZERO;
    row.b_y2_lo7 = F::ZERO;
    for i in 0..3 {
        row.a_sig[i] = F::ZERO;
        row.b_sig[i] = F::ZERO;
        row.a_sig_hi[i] = F::ZERO;
        row.r[i] = F::ZERO;
        row.lt_borrow[i] = F::ZERO;
        row.diff[i] = F::ZERO;
    }
    for i in 0..4 {
        row.q[i] = F::ZERO;
        row.q_lo2[i] = F::ZERO;
        row.q_hi6[i] = F::ZERO;
        row.lo_bit0[i] = F::ZERO;
        row.lo_bit1[i] = F::ZERO;
    }
    for i in 0..6 {
        row.div_carry[i] = F::ZERO;
    }
    row.qh6_lo = F::ZERO;
    row.qh6_rest = F::ZERO;
    row.is_high = F::ZERO;
    row.guard = F::ZERO;
    row.lsb = F::ZERO;
    row.sticky = F::ZERO;
    row.r_nz = F::ZERO;
    row.r_nz_inv = F::ZERO;
    row.round_up = F::ZERO;
    row.rc0 = F::ZERO;
    row.rc1 = F::ZERO;
    row.rc2 = F::ZERO;
    row.rc2_rem = F::ZERO;
    row.result_exp = F::ZERO;
    row.s_normal = F::ZERO;
    row.s_overflow = F::ZERO;
    row.s_subnormal = F::ZERO;
    row.route_ovf_decomp = [F::ZERO];
    row.route_sub_decomp = [F::ZERO];
    row.sub_capped = F::ZERO;
    row.sub_lt_decomp = [F::ZERO];
    row.sub_shift = F::ZERO;
    row.sub_shrn = ShrnCols::<F, 7>::default();
    row.sub_y0_high7 = F::ZERO;
    row.sub_lsb = F::ZERO;
    row.sub_sticky_eff = F::ZERO;
    row.sub_round_up = F::ZERO;
    row.sub_rc0 = F::ZERO;
    row.sub_rc1 = F::ZERO;
}

/// Fill the finite-division witness columns.
fn fill_finite_div<F: PrimeField32>(
    row: &mut FdivCoreCols<F>,
    a_exp: u32,
    b_exp: u32,
    a_mant: u32,
    b_mant: u32,
    result: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    // Pre-normalize subnormal operands (exponent absorbs the shift).
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

    // Significand bytes.
    let asig_bytes: [u32; 3] = [a_sig & 0xFF, (a_sig >> 8) & 0xFF, (a_sig >> 16) & 0xFF];
    let bsig_bytes: [u32; 3] = [b_sig & 0xFF, (b_sig >> 8) & 0xFF, (b_sig >> 16) & 0xFF];
    for i in 0..3 {
        row.a_sig[i] = F::from_u32(asig_bytes[i]);
        row.b_sig[i] = F::from_u32(bsig_bytes[i]);
        row.a_sig_hi[i] = F::from_u32(asig_bytes[i] >> 7);
        range_checker.add_count(asig_bytes[i], 8);
        range_checker.add_count(bsig_bytes[i], 8);
    }

    // Long division: q = a_sig * 2^25 / b_sig, r = remainder.
    let a_ext = (a_sig as u64) << 25;
    let q_val = a_ext / (b_sig as u64);
    let r_val = a_ext % (b_sig as u64);
    debug_assert!(q_val < (1 << 26), "fdiv quotient overflow");
    debug_assert!(r_val < b_sig as u64, "fdiv remainder >= divisor");

    let q_bytes: [u32; 4] = core::array::from_fn(|i| ((q_val >> (i * 8)) & 0xFF) as u32);
    let r_bytes: [u32; 3] = core::array::from_fn(|i| ((r_val >> (i * 8)) & 0xFF) as u32);

    for (i, &qb) in q_bytes.iter().enumerate() {
        row.q[i] = F::from_u32(qb);
        range_checker.add_count(qb, 8);
    }
    for (i, &rb) in r_bytes.iter().enumerate() {
        row.r[i] = F::from_u32(rb);
        range_checker.add_count(rb, 8);
    }

    // Target bytes: a_sig << 25.
    let lo7 = |i: usize| asig_bytes[i] & 0x7F;
    let target: [u32; 7] = [
        0,
        0,
        0,
        lo7(0) * 2,
        (asig_bytes[0] >> 7) + lo7(1) * 2,
        (asig_bytes[1] >> 7) + lo7(2) * 2,
        asig_bytes[2] >> 7,
    ];

    // Carry chain: verify q * b_sig + r = target.
    let mut carry_in: u64 = 0;
    for i in 0..7 {
        let mut conv: u64 = 0;
        for k in 0..4usize {
            if k <= i && (i - k) < 3 {
                conv += (q_bytes[k] as u64) * (bsig_bytes[i - k] as u64);
            }
        }
        let r_byte: u64 = if i < 3 { r_bytes[i] as u64 } else { 0 };
        let expected = carry_in + conv + r_byte;
        let carry_out = (expected - target[i] as u64) / 256;
        debug_assert_eq!(
            expected - target[i] as u64,
            carry_out * 256,
            "fdiv carry mismatch at byte {}",
            i
        );
        if i < 6 {
            row.div_carry[i] = F::from_u32(carry_out as u32);
            range_checker.add_count(carry_out as u32, 10);
        } else {
            debug_assert_eq!(carry_out, 0, "fdiv final carry nonzero");
        }
        carry_in = carry_out;
    }

    // Remainder range check: r < b_sig.
    let b_minus_r_minus1: i64 = (b_sig as i64) - (r_val as i64) - 1;
    debug_assert!(b_minus_r_minus1 >= 0, "fdiv remainder >= divisor");
    let mut borrow: i64 = 1; // subtract 1 from byte 0
    for i in 0..3 {
        let d = (bsig_bytes[i] as i64) - (r_bytes[i] as i64) - borrow;
        let (byte_val, new_borrow) = if d < 0 { (d + 256, 1) } else { (d, 0) };
        row.lt_borrow[i] = F::from_u32(new_borrow as u32);
        row.diff[i] = F::from_u32(byte_val as u32);
        range_checker.add_count(byte_val as u32, 8);
        borrow = new_borrow;
    }
    debug_assert_eq!(borrow, 0, "fdiv r >= b_sig");

    // Quotient byte decomposition.
    for (i, &qb) in q_bytes.iter().enumerate() {
        let lo2 = qb & 0x3;
        let hi6 = qb >> 2;
        row.q_lo2[i] = F::from_u32(lo2);
        row.q_hi6[i] = F::from_u32(hi6);
        row.lo_bit0[i] = F::from_u32(lo2 & 1);
        row.lo_bit1[i] = F::from_u32(lo2 >> 1);
        range_checker.add_count(lo2, 2);
        range_checker.add_count(hi6, 6);
    }

    // is_high: bit 25 of q = bit 1 of q_bytes[3] = lo_bit1[3].
    let is_high = (q_bytes[3] >> 1) & 1 == 1;
    row.is_high = F::from_bool(is_high);

    // q_hi6[0] decomposition for is_high lsb.
    let qh6_lo_val = q_bytes[0] >> 2 & 1;
    let qh6_rest_val = (q_bytes[0] >> 2) >> 1;
    row.qh6_lo = F::from_u32(qh6_lo_val);
    row.qh6_rest = F::from_u32(qh6_rest_val);
    range_checker.add_count(qh6_rest_val, 5);

    // r_nz: use byte sum (matches the AIR's r_sum = r[0]+r[1]+r[2]).
    let r_sum_val = r_bytes[0] + r_bytes[1] + r_bytes[2];
    let r_nz = r_sum_val != 0;
    row.r_nz = F::from_bool(r_nz);
    row.r_nz_inv = if r_nz {
        F::from_u32(r_sum_val).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // Rounding metadata.
    let (guard, lsb) = if is_high {
        ((q_bytes[0] >> 1) & 1, (q_bytes[0] >> 2) & 1)
    } else {
        (q_bytes[0] & 1, (q_bytes[0] >> 1) & 1)
    };
    let sticky = if is_high {
        (q_bytes[0] & 1 != 0) || r_nz
    } else {
        r_nz
    };
    let round_up = guard == 1 && (sticky || lsb == 1);
    row.guard = F::from_u32(guard);
    row.lsb = F::from_u32(lsb);
    row.sticky = F::from_bool(sticky);
    row.round_up = F::from_bool(round_up);

    // Mantissa extraction + carry chain.
    let (mant0, mant1, top7) = if is_high {
        (
            q_bytes[0] >> 2 | ((q_bytes[1] & 3) << 6),
            q_bytes[1] >> 2 | ((q_bytes[2] & 3) << 6),
            q_bytes[2] >> 2 | ((q_bytes[3] & 1) << 6),
        )
    } else {
        (
            q_bytes[0] >> 1 | ((q_bytes[1] & 1) << 7),
            q_bytes[1] >> 1 | ((q_bytes[2] & 1) << 7),
            q_bytes[2] >> 1,
        )
    };
    let s0 = mant0 + round_up as u32;
    let rc0 = s0 >> 8;
    let s1 = mant1 + rc0;
    let rc1 = s1 >> 8;
    let top7c = top7 + rc1;
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    row.rc0 = F::from_u32(rc0);
    row.rc1 = F::from_u32(rc1);
    row.rc2 = F::from_u32(rc2);
    row.rc2_rem = F::from_u32(rc2_rem);
    range_checker.add_count(rc2_rem, 7);

    // Result exponent: for normal results, a_exp - b_exp + 126 + is_high + rc2.
    // For overflow (native result is inf), use the native exp field (255).
    // For subnormal/underflow (native result exp field == 0), use 0.
    let base_exp = (a_exp_eff as i64) - (b_exp_eff as i64) + 126 + is_high as i64;
    let computed_exp = base_exp + rc2 as i64;
    let r_exp_field = (result >> 23) & 0xFF;
    let r_exp = if r_exp_field == 255 {
        255
    } else if r_exp_field == 0 {
        0
    } else {
        computed_exp as u32
    };
    let r_exp_lo = r_exp & 1;
    let r_exp_hi = r_exp >> 1;
    row.r_exp_lo = F::from_u32(r_exp_lo);
    row.r_exp_hi = F::from_u32(r_exp_hi);
    row.result_exp = F::from_u32(r_exp);

    // Result-class routing (sound, via base_exp + rc2 / base_exp).
    let is_overflow = base_exp + rc2 as i64 >= 255;
    let s_subnormal = base_exp <= 0;
    row.s_normal = F::from_bool(!is_overflow && !s_subnormal);
    row.s_overflow = F::from_bool(is_overflow);
    row.s_subnormal = F::from_bool(s_subnormal);
    let base_exp_b = (base_exp + 256) as u32;
    debug_assert!(base_exp_b < 1024, "fdiv base_exp_b out of range");
    range_checker.add_count(base_exp_b, 10);
    let lt10 = IsLtSubAir::new(range_checker.bus(), 10);
    lt10.generate_subrow(
        (range_checker.as_ref(), 510u32, base_exp_b + rc2),
        (&mut row.route_ovf_decomp, &mut row.s_overflow),
    );
    lt10.generate_subrow(
        (range_checker.as_ref(), base_exp_b, 257u32),
        (&mut row.route_sub_decomp, &mut row.s_subnormal),
    );
    debug_assert_eq!(row.s_overflow, F::from_bool(is_overflow));
    debug_assert_eq!(row.s_subnormal, F::from_bool(s_subnormal));

    // Sanity: computed exponent must match native result for normal results.
    debug_assert!(
        r_exp_field == 255 || r_exp_field == 0 || r_exp == computed_exp as u32,
        "fdiv exp mismatch: a_exp={} b_exp={} is_high={} rc2={} computed={} field={}",
        a_exp,
        b_exp,
        is_high,
        rc2,
        computed_exp,
        r_exp_field
    );

    // ── Subnormal / underflow witness ───────────────────────────────
    if s_subnormal {
        let true_shift = b_exp_eff - a_exp_eff - 124;
        debug_assert!(true_shift >= 2, "fdiv subnormal shift < 2: {}", true_shift);
        let sub_capped = true_shift >= 56;
        let lt_air = IsLtSubAir::new(range_checker.bus(), 8);
        lt_air.generate_subrow(
            (range_checker.as_ref(), 55u32, true_shift as u32),
            (&mut row.sub_lt_decomp, &mut row.sub_capped),
        );
        debug_assert_eq!(
            row.sub_capped.as_canonical_u32(),
            sub_capped as u32,
            "fdiv sub_capped mismatch"
        );

        let cshift = if sub_capped { 55u32 } else { true_shift as u32 };
        row.sub_shift = F::from_u32(cshift);
        range_checker.add_count(cshift, 6);

        // Extend q to 7 bytes and build ShrnCols.
        let x: [u32; 7] = core::array::from_fn(|i| if i < 4 { q_bytes[i] } else { 0 });
        row.sub_shrn = make_shrn_cols7::<F>(x, cshift as usize, range_checker);

        let y0 = row.sub_shrn.y[0].as_canonical_u32();
        let y1 = row.sub_shrn.y[1].as_canonical_u32();
        let y2 = row.sub_shrn.y[2].as_canonical_u32();
        row.sub_y0_high7 = F::from_u32(y0 >> 1);
        row.sub_lsb = F::from_u32(y0 & 1);
        range_checker.add_count(y0 >> 1, 7);

        let guard = row.sub_shrn.guard.as_canonical_u32();
        let sticky = row.sub_shrn.sticky.as_canonical_u32() != 0 || r_val != 0;
        row.sub_sticky_eff = F::from_bool(sticky);
        let round_up = guard == 1 && (sticky || (y0 & 1) == 1);
        row.sub_round_up = F::from_bool(round_up);

        let s0 = y0 + round_up as u32;
        let rc0 = s0 >> 8;
        let s1 = y1 + rc0;
        let rc1 = s1 >> 8;
        row.sub_rc0 = F::from_u32(rc0);
        row.sub_rc1 = F::from_u32(rc1);

        debug_assert_eq!(s0 & 0xFF, result & 0xFF, "fdiv subnormal byte0");
        debug_assert_eq!(s1 & 0xFF, (result >> 8) & 0xFF, "fdiv subnormal byte1");
        debug_assert_eq!(
            (y2 + rc1) & 0xFF,
            (result >> 16) & 0xFF,
            "fdiv subnormal byte2"
        );
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
}

/// Build a `ShrnCols<7>` witness and register range-checker counts.
fn make_shrn_cols7<F: PrimeField32>(
    x: [u32; 7],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShrnCols<F, 7> {
    const N: usize = 7;
    let shift = shift.min(N * crate::fp_shift::SHLIMB_BITS - 1);
    let (y, q, r) = run_shrn(&x, shift);
    let (guard, sticky) = run_shrn_round(&x, shift);
    let mut limb_shift_marker = [F::ZERO; N];
    let mut bit_shift_marker = [F::ZERO; crate::fp_shift::SHLIMB_BITS];
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
mod finite_numeric_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn run(a: u32, b: u32) -> NumericBuilder<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let mut row = FdivCoreCols::<F>::default();
        let result = fdiv_s(a, b);
        // Reuse the filler's witness computation by calling fill_finite_div directly
        // for normal inputs.
        let (a_sign, a_exp, a_mant) = decompose(a);
        let (b_sign, b_exp, b_mant) = decompose(b);
        let a_normal = (1..=254).contains(&a_exp);
        let b_normal = (1..=254).contains(&b_exp);
        let a_sub = a_exp == 0 && a_mant != 0;
        let b_sub = b_exp == 0 && b_mant != 0;
        let any_nan = (a_exp == 255 && a_mant != 0) || (b_exp == 255 && b_mant != 0);
        let a_inf = a_exp == 255 && a_mant == 0;
        let b_inf = b_exp == 255 && b_mant == 0;
        let a_zero = a_exp == 0 && a_mant == 0;
        let b_zero = b_exp == 0 && b_mant == 0;
        let canon_sel = any_nan || (a_inf && b_inf) || (a_zero && b_zero);
        let inf_res = !any_nan && ((a_inf && !b_inf) || (b_zero && !a_inf && !a_zero));
        let zero_res = !any_nan && ((b_inf && !a_inf) || (a_zero && !b_zero && !b_inf));
        let s_finite =
            !canon_sel && !inf_res && !zero_res && (a_normal || a_sub) && (b_normal || b_sub);
        row.s_finite = F::from_bool(s_finite);
        row.a_normal = F::from_bool(a_normal);
        row.b_normal = F::from_bool(b_normal);
        row.a_sub = F::from_bool(a_sub);
        row.b_sub = F::from_bool(b_sub);
        row.a_exp_is_zero = F::from_bool(a_exp == 0);
        row.b_exp_is_zero = F::from_bool(b_exp == 0);
        row.a_exp_is_max = F::from_bool(a_exp == 255);
        row.b_exp_is_max = F::from_bool(b_exp == 255);
        row.a_mant_is_zero = F::from_bool(a_mant == 0);
        row.b_mant_is_zero = F::from_bool(b_mant == 0);
        row.is_valid = F::ONE;
        row.r_sign = F::from_bool(a_sign != b_sign);
        row.is_special = F::from_bool(canon_sel || inf_res || zero_res);
        for i in 0..4 {
            row.a[i] = F::from_u32((a >> (i * 8)) & 0xFF);
            row.b[i] = F::from_u32((b >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        row.a_sign = F::from_u32(a_sign);
        row.a_exp_lo = F::from_u32(a_exp & 1);
        row.a_hi = F::from_u32(a_exp >> 1);
        row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        row.b_sign = F::from_u32(b_sign);
        row.b_exp_lo = F::from_u32(b_exp & 1);
        row.b_hi = F::from_u32(b_exp >> 1);
        row.b_mant2 = F::from_u32((b_mant >> 16) & 0x7F);
        if s_finite {
            fill_finite_div(&mut row, a_exp, b_exp, a_mant, b_mant, result, &chip);
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
        }
        let air = FdivCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0);
        let mut builder = NumericBuilder::new(vec![]);
        air.eval_finite_div(&mut builder, &row);
        air.eval_finite_subnormal(&mut builder, &row);
        builder
    }

    #[test]
    fn fdiv_finite_numeric_passes() {
        // Test specific known values first.
        for &(a, b) in &[
            (3.0f32.to_bits(), 2.0f32.to_bits()), // 1.5
            (7.0f32.to_bits(), 3.0f32.to_bits()), // 2.333...
            (1.0f32.to_bits(), 3.0f32.to_bits()), // 0.333...
            (5.0f32.to_bits(), 7.0f32.to_bits()),
            (1.5f32.to_bits(), 1.5f32.to_bits()),     // 1.0
            (f32::MAX.to_bits(), f32::MIN.to_bits()), // overflow -> -inf
            (0x3F800000u32, 0x7F7FFFFF),              // 1/MAX -> subnormal
            (0x00800000u32, 0x7F7FFFFF),              // MIN_NORMAL/MAX -> deep underflow
            (0x00000001, 0x3F800000),                 // min subnormal / 1
            (0x3F800000, 0x00000001),                 // 1 / min subnormal -> inf
            (0x00000001, 0x00000001),                 // min subnormal / min subnormal
            (0x007FFFFF, 0x40000000),                 // max subnormal * 2
            (0x00800000, 0x00400000),                 // MIN_NORMAL / subnormal
        ] {
            let builder = run(a, b);
            assert!(
                builder.ok(),
                "fdiv failed: {:?}\n a={:08x} b={:08x} result={:08x}",
                builder.failures,
                a,
                b,
                fdiv_s(a, b)
            );
        }
        // Fuzz.
        let mut rng = 0xB0B0_000Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..2000 {
            let a_exp = (next() % 254) + 1;
            let b_exp = (next() % 254) + 1;
            let a = ((next() & 1) << 31) | (a_exp << 23) | (next() & 0x7FFFFF);
            let b = ((next() & 1) << 31) | (b_exp << 23) | (next() & 0x7FFFFF);
            let builder = run(a, b);
            if !builder.ok() {
                let r = fdiv_s(a, b);
                let r_exp = (r >> 23) & 0xFF;
                panic!(
                    "fdiv finite failed: {:?}\n a={:08x} b={:08x} result={:08x} exp_field={}",
                    builder.failures, a, b, r, r_exp
                );
            }
        }
        // Fuzz including subnormal operands.
        for _ in 0..2000 {
            let a_exp = next() % 255;
            let b_exp = next() % 255;
            let a = ((next() & 1) << 31) | (a_exp << 23) | (next() & 0x7FFFFF);
            let b = ((next() & 1) << 31) | (b_exp << 23) | (next() & 0x7FFFFF);
            if (a_exp == 0 && (a & 0x7FFFFF) == 0) || (b_exp == 0 && (b & 0x7FFFFF) == 0) {
                continue; // zero operands are special-cased
            }
            let builder = run(a, b);
            assert!(
                builder.ok(),
                "fdiv subnormal-operand fuzz failed: {:?}\n a={:08x} b={:08x}",
                builder.failures,
                a,
                b
            );
        }
    }
}
