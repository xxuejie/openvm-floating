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

/// IEEE 754 / RISC-V FSQRT.S: correctly-rounded (RNE) square root.
/// For all nonnegative inputs (including -0.0 and +inf) the result is the
/// unique IEEE-correctly-rounded value — identical to x86's `sqrtss`.
/// Per the RISC-V spec, the canonical NaN is returned for NaN inputs and
/// negative nonzero inputs (invalid operation).
#[inline(always)]
pub fn fsqrt_s(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        return CANONICAL_NAN;
    }
    if f < 0.0 {
        return CANONICAL_NAN; // negative (nonzero) is invalid
    }
    // +0.0, -0.0, +inf, or positive finite: native sqrt is the unique
    // correctly-rounded result (sqrt(-0.0) = -0.0, sqrt(+inf) = +inf).
    f.sqrt().to_bits()
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FsqrtCoreCols<T> {
    pub a: [T; 4],

    // === Decomposition of input a ===
    pub a_sign: T,
    pub a_exp_lo: T,
    pub a_mant2: T,
    pub a_hi: T,

    // === Classification inverse witnesses and flags ===
    pub a_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub a_exp_is_zero: T,
    pub a_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub a_nan: T,
    pub a_inf: T,
    pub a_zero: T,

    // === Special-case selection ===
    pub canon_sel: T, // a_nan | (a_sign && !a_zero): NaN input or negative nonzero
    pub inf_res: T,   // +inf
    pub zero_res: T,  // +-0: result = a
    pub is_special: T,

    // === Finite square root (normal positive inputs) ===
    pub s_finite: T,
    pub a_normal: T,
    pub a_subnormal: T,
    pub eff_exp_odd: T,
    pub eff_exp_hi: T,
    pub norm_shift: T,
    pub ns_lo: T,
    pub ns_half: T,
    pub a_sig2_lo7: T, // low 7 bits of the normalized subnormal significand top byte
    pub norm_shln: ShlnCols<T, 4>,
    pub a_sig: [T; 3],    // significand bytes (mant | implicit)
    pub a_sig_hi: [T; 3], // bit 7 of each a_sig byte (for odd-exp left shift)
    pub q: [T; 5],        // floor(sqrt(n)) bytes
    pub q_hi: [T; 5],     // bit 7 of each q byte (for 2q computation)
    pub q4_lo4: T,        // low 4 bits of q[4]
    pub is_hi: T,         // bool: q[4] >= 16 (msb of q is bit 36)
    pub rem: [T; 6],      // remainder = n - q² bytes
    pub rem_nz: T,        // bool: rem != 0
    pub rem_nz_inv: T,
    pub sq_carry: [T; 9],          // carry chain for q² + rem = n
    pub twoq: [T; 6],              // 2q bytes (for remainder bound)
    pub lt_borrow: [T; 6],         // borrow chain for 2q >= rem
    pub diff: [T; 6],              // 2q - rem diff bytes
    pub sqrt_shrn: ShrnCols<T, 7>, // mantissa extraction via right shift
    pub sqrt_shift: T,             // 12 + is_hi
    pub y0_high7: T,               // y[0] >> 1 (for lsb extraction)
    pub eff_sticky: T,
    pub round_up: T,
    pub rc0: T,
    pub rc1: T,
    pub rc2: T,
    pub rc2_rem: T,
    pub r_exp_lo: T,
    pub r_exp_hi: T,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FsqrtCoreCols<u8>)]
pub struct FsqrtCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FsqrtCoreAir {
    fn width(&self) -> usize {
        FsqrtCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FsqrtCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FsqrtCoreAir
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
        let cols: &FsqrtCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        for flag in [
            cols.a_sign,
            cols.a_exp_lo,
            cols.a_exp_is_zero,
            cols.a_exp_is_max,
            cols.a_mant_is_zero,
            cols.a_nan,
            cols.a_inf,
            cols.a_zero,
            cols.canon_sel,
            cols.inf_res,
            cols.zero_res,
            cols.is_special,
        ] {
            builder.assert_bool(flag);
        }

        // ── Byte decomposition ───────────────────────────────────────────
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // exp_lo / sign are free field choices (classification re-routing).
        for v in [cols.a_mant2, cols.a_hi] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

        // Range check input bytes (bounds exponent/mantissa sums) and the
        // result bytes (the memory side does not byte-check written data).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.result[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        // ── Classification ───────────────────────────────────────────────
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_zero,
            one.clone() - a_exp.clone() * cols.a_exp_zero_inv,
        );
        builder.assert_zero(a_exp.clone() * cols.a_exp_is_zero);
        let a_exp_max_delta = AB::Expr::from(f255) - a_exp;
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_max,
            one.clone() - a_exp_max_delta.clone() * cols.a_exp_max_inv,
        );
        builder.assert_zero(a_exp_max_delta * cols.a_exp_is_max);
        // exp cannot be both 0 and 255
        builder.assert_zero(cols.a_exp_is_zero * cols.a_exp_is_max);
        let a_mant_sum = cols.a[0] + cols.a[1] + cols.a_mant2;
        builder.when(cols.is_valid).assert_eq(
            cols.a_mant_is_zero,
            one.clone() - a_mant_sum.clone() * cols.a_mant_sum_inv,
        );
        builder.assert_zero(a_mant_sum * cols.a_mant_is_zero);
        builder.assert_eq(
            cols.a_nan,
            cols.a_exp_is_max * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(cols.a_inf, cols.a_exp_is_max * cols.a_mant_is_zero);
        builder.assert_eq(cols.a_zero, cols.a_exp_is_zero * cols.a_mant_is_zero);

        // ── Special-case selection ───────────────────────────────────────
        // Canonical NaN: NaN input, or negative nonzero (incl. -inf and
        // negative subnormals).
        builder.assert_eq(
            cols.canon_sel,
            cols.a_nan + cols.a_sign * (one.clone() - cols.a_zero),
        );
        builder.assert_eq(cols.inf_res, cols.a_inf * (one.clone() - cols.a_sign));
        builder.assert_eq(cols.zero_res, cols.a_zero);
        builder.assert_eq(
            cols.is_special,
            cols.canon_sel + cols.inf_res + cols.zero_res,
        );

        // Canonical NaN for NaN inputs and invalid (negative) operations.
        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &canon_byte) in canon.iter().enumerate() {
            builder
                .when(cols.canon_sel)
                .assert_eq(cols.result[i], AB::F::from_u32(canon_byte as u32));
        }
        // +inf result.
        builder.when(cols.inf_res).assert_zero(cols.result[0]);
        builder.when(cols.inf_res).assert_zero(cols.result[1]);
        builder
            .when(cols.inf_res)
            .assert_eq(cols.result[2], AB::F::from_u32(128));
        builder
            .when(cols.inf_res)
            .assert_eq(cols.result[3], AB::F::from_u32(127));
        // +-0 result: sqrt copies the input (sqrt(-0.0) = -0.0).
        for i in 0..4 {
            builder
                .when(cols.zero_res)
                .assert_eq(cols.result[i], cols.a[i]);
        }

        // ── Finite result: integer square root verification ─────────────
        self.eval_finite_sqrt(builder, cols);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FSQRT_S);

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

impl FsqrtCoreAir {
    /// Constrain the finite square-root result via integer isqrt verification.
    ///
    /// Verifies `q² + rem = n` (10-byte carry chain) and `rem ≤ 2q` (6-byte
    /// borrow chain), where `n = sig << (48 + exp_odd)`. Then extracts the
    /// 24-bit result significand from q via ShrnGadget (shift 12 + is_hi)
    /// with RNE rounding. The result exponent is `a_hi + 63 + is_hi + rc2`.
    ///
    /// Subnormal inputs are pre-normalized via ShlnGadget (exponent absorbs the
    /// shift); the sqrt of a positive finite input is always normal.
    pub fn eval_finite_sqrt<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FsqrtCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f16: AB::F = AB::F::from_u32(16);
        let f63: AB::F = AB::F::from_u32(63);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(256);
        let one = AB::Expr::ONE;

        // ── Selector: positive finite AND (normal OR subnormal) ─────────
        let not_special = one.clone() - cols.is_special;
        let a_normal_expr = (one.clone() - cols.a_exp_is_zero) * (one.clone() - cols.a_exp_is_max);
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_normal, a_normal_expr);
        builder.when(cols.is_valid).assert_eq(
            cols.a_subnormal,
            cols.a_exp_is_zero * (one.clone() - cols.a_mant_is_zero) * (one.clone() - cols.a_sign),
        );
        builder.when(cols.is_valid).assert_eq(
            cols.s_finite,
            not_special * (cols.a_normal + cols.a_subnormal) * (one.clone() - cols.a_sign),
        );
        builder.assert_bool(cols.s_finite);
        let s = cols.s_finite;
        let is_sub = cols.a_subnormal;

        // ── Subnormal normalization via ShlnGadget<4> ──────────────────
        builder.when(s).assert_eq(cols.norm_shln.x[0], cols.a[0]);
        builder.when(s).assert_eq(cols.norm_shln.x[1], cols.a[1]);
        builder.when(s).assert_eq(cols.norm_shln.x[2], cols.a_mant2);
        builder.when(s).assert_zero(cols.norm_shln.x[3]);
        builder
            .when(s)
            .assert_eq(cols.norm_shln.shift, cols.norm_shift);
        // norm_shift decomposes as ns_lo + 2*ns_half (parity + half-shift),
        // which the effective exponent below is built from.
        builder.assert_bool(cols.ns_lo);
        builder
            .when(s)
            .assert_eq(cols.norm_shift, cols.ns_lo + cols.ns_half * f2);
        self.range_bus
            .range_check(cols.ns_half.into(), 4)
            .eval(builder, is_sub.into());
        builder.assert_eq(cols.norm_shln.enable, is_sub);
        let shln = ShlnGadget::new(self.range_bus);
        shln.eval_shln(builder, &cols.norm_shln);

        // ── Significand construction ────────────────────────────────────
        // Normal: a_sig = mant | implicit. Subnormal: a_sig = norm_shln.y[0..3].
        builder
            .when(s * (one.clone() - is_sub))
            .assert_eq(cols.a_sig[0], cols.a[0]);
        builder
            .when(s * (one.clone() - is_sub))
            .assert_eq(cols.a_sig[1], cols.a[1]);
        builder
            .when(s * (one.clone() - is_sub))
            .assert_eq(cols.a_sig[2], cols.a_mant2 + f128);
        builder
            .when(s * is_sub)
            .assert_eq(cols.a_sig[0], cols.norm_shln.y[0]);
        builder
            .when(s * is_sub)
            .assert_eq(cols.a_sig[1], cols.norm_shln.y[1]);
        builder
            .when(s * is_sub)
            .assert_eq(cols.a_sig[2], cols.norm_shln.y[2]);
        builder.when(s * is_sub).assert_zero(cols.norm_shln.y[3]);
        // The normalized significand's MSB must land at bit 23 (anchors the
        // normalization shift; otherwise the exponent reconstruction breaks).
        builder
            .when(s * is_sub)
            .assert_eq(cols.a_sig[2], cols.a_sig2_lo7 + f128);
        self.range_bus
            .range_check(cols.a_sig2_lo7.into(), 7)
            .eval(builder, s * is_sub);
        // Pin a_sig_hi as bit 7: a_sig = lo7 + hi*128 with lo7 ∈ [0, 127].
        for i in 0..3 {
            builder.assert_bool(cols.a_sig_hi[i]);
            self.range_bus
                .range_check(cols.a_sig[i].into(), 8)
                .eval(builder, s.into());
            self.range_bus
                .range_check(cols.a_sig[i] - cols.a_sig_hi[i] * f128, 7)
                .eval(builder, s.into());
        }

        // ── Effective exp_odd and exp_hi ────────────────────────────────
        builder.when(s).assert_eq(
            cols.eff_exp_odd,
            (one.clone() - is_sub) * cols.a_exp_lo + is_sub * (one.clone() - cols.ns_lo),
        );
        builder.when(s).assert_eq(
            cols.eff_exp_hi,
            (one.clone() - is_sub) * cols.a_hi - is_sub * cols.ns_half,
        );

        // ── n = sig << (48 + eff_exp_odd) ───────────────────────────────
        let lo7 = |i: usize| cols.a_sig[i] - cols.a_sig_hi[i] * f128;
        let exp_odd = cols.eff_exp_odd;

        // ── q range checks ──────────────────────────────────────────────
        for i in 0..5 {
            self.range_bus
                .range_check(cols.q[i].into(), 8)
                .eval(builder, s.into());
        }
        // q[4] < 32 (since q < 2^37). Decompose: q[4] = q4_lo4 + is_hi*16.
        builder.assert_bool(cols.is_hi);
        builder
            .when(s)
            .assert_eq(cols.q[4], cols.q4_lo4 + cols.is_hi * f16);
        self.range_bus
            .range_check(cols.q4_lo4.into(), 4)
            .eval(builder, s.into());

        // Pin q_hi as bit 7 of each q byte: q = lo7 + hi*128 with lo7 ∈ [0, 127].
        // q_hi[4] must be 0 (q[4] < 32 from the q4_lo4 / is_hi split above).
        for i in 0..5 {
            builder.assert_bool(cols.q_hi[i]);
            self.range_bus
                .range_check(cols.q[i] - cols.q_hi[i] * f128, 7)
                .eval(builder, s.into());
        }
        builder.when(s).assert_zero(cols.q_hi[4]);

        for i in 0..6 {
            self.range_bus
                .range_check(cols.rem[i].into(), 8)
                .eval(builder, s.into());
        }

        // ── Carry chain: q² + rem = n (10 bytes) ───────────────────────
        let carry_div = AB::F::from_u32(256).inverse();
        for i in 0..10 {
            let mut conv = AB::Expr::ZERO;
            for k in 0..5usize {
                let j = i as isize - k as isize;
                if j >= 0 && (j as usize) < 5 {
                    conv += cols.q[k] * cols.q[j as usize];
                }
            }
            let rem_byte = if i < 6 {
                cols.rem[i].into()
            } else {
                AB::Expr::ZERO
            };
            let carry_in = if i == 0 {
                AB::Expr::ZERO
            } else {
                cols.sq_carry[i - 1].into()
            };
            let expected = carry_in + conv + rem_byte;
            let nih = one.clone() - exp_odd;
            let n_bi = match i {
                0..=5 => AB::Expr::ZERO,
                6 => nih * cols.a_sig[0] + exp_odd * lo7(0) * f2,
                7 => nih * cols.a_sig[1] + exp_odd * (lo7(1) * f2 + cols.a_sig_hi[0]),
                8 => nih * cols.a_sig[2] + exp_odd * (lo7(2) * f2 + cols.a_sig_hi[1]),
                9 => exp_odd * cols.a_sig_hi[2],
                _ => unreachable!(),
            };
            let carry_out = (expected - n_bi) * carry_div;
            if i < 9 {
                builder.when(s).assert_eq(cols.sq_carry[i], carry_out);
                self.range_bus
                    .range_check(cols.sq_carry[i].into(), 12)
                    .eval(builder, s.into());
            } else {
                builder.when(s).assert_zero(carry_out);
            }
        }

        // ── Remainder bound: rem ≤ 2q (6-byte borrow chain) ────────────
        // twoq bytes: twoq[0] = q[0]*2 - q_hi[0]*256, twoq[i] = q[i]*2 + q_hi[i-1] - q_hi[i]*256.
        for i in 0..6 {
            let twoq_expr = if i == 0 {
                cols.q[0] * f2 - cols.q_hi[0] * f256
            } else if i < 5 {
                cols.q[i] * f2 + cols.q_hi[i - 1] - cols.q_hi[i] * f256
            } else {
                // twoq[5] = q_hi[4] = 0
                cols.q_hi[4].into()
            };
            builder.when(s).assert_eq(cols.twoq[i], twoq_expr);
            self.range_bus
                .range_check(cols.twoq[i].into(), 8)
                .eval(builder, s.into());
            builder.assert_bool(cols.lt_borrow[i]);
        }
        // byte 0: twoq[0] - rem[0] + borrow*256 = diff[0]
        builder.when(s).assert_eq(
            cols.twoq[0] - cols.rem[0] + cols.lt_borrow[0] * f256,
            cols.diff[0],
        );
        for i in 1..6 {
            builder.when(s).assert_eq(
                cols.twoq[i] - cols.rem[i] - cols.lt_borrow[i - 1] + cols.lt_borrow[i] * f256,
                cols.diff[i],
            );
        }
        builder.when(s).assert_zero(cols.lt_borrow[5]);
        for i in 0..6 {
            self.range_bus
                .range_check(cols.diff[i].into(), 8)
                .eval(builder, s.into());
        }

        // ── rem_nz ──────────────────────────────────────────────────────
        let rem_sum =
            cols.rem[0] + cols.rem[1] + cols.rem[2] + cols.rem[3] + cols.rem[4] + cols.rem[5];
        let rem_is_zero = one.clone() - rem_sum.clone() * cols.rem_nz_inv;
        builder.assert_zero(rem_sum * rem_is_zero.clone());
        builder
            .when(s)
            .assert_eq(cols.rem_nz, one.clone() - rem_is_zero);
        builder.assert_bool(cols.rem_nz);

        // ── Mantissa extraction via ShrnGadget<7> ───────────────────────
        // x = q extended to 7 bytes. shift = 12 + is_hi.
        for i in 0..5 {
            builder.when(s).assert_eq(cols.sqrt_shrn.x[i], cols.q[i]);
        }
        for i in 5..7 {
            builder.when(s).assert_zero(cols.sqrt_shrn.x[i]);
        }
        builder.when(s).assert_eq(
            cols.sqrt_shift,
            AB::Expr::from(AB::F::from_u32(12)) + cols.is_hi * one.clone(),
        );
        // f48/4 = 12
        builder
            .when(s)
            .assert_eq(cols.sqrt_shrn.shift, cols.sqrt_shift);
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.sqrt_shrn);
        builder.assert_eq(cols.sqrt_shrn.enable, s);
        // y bytes 3..7 must be zero (result fits in 3 bytes).
        for i in 3..7 {
            builder.when(s).assert_zero(cols.sqrt_shrn.y[i]);
        }

        // ── RNE rounding ────────────────────────────────────────────────
        // Decompose y[0] = y0_high7 * 2 + lsb_y0 (lsb_y0 ∈ {0,1}).
        let lsb_y0 = cols.sqrt_shrn.y[0] - cols.y0_high7 * f2;
        self.range_bus
            .range_check(cols.y0_high7.into(), 7)
            .eval(builder, s.into());
        builder.when(s).assert_bool(lsb_y0.clone());

        // Effective sticky: gadget sticky OR rem_nz.
        let eff_sticky_expr =
            cols.sqrt_shrn.sticky + cols.rem_nz - cols.sqrt_shrn.sticky * cols.rem_nz;
        builder.when(s).assert_eq(cols.eff_sticky, eff_sticky_expr);
        builder.assert_bool(cols.eff_sticky);
        builder.assert_bool(cols.sqrt_shrn.guard);

        // round_up = guard AND (eff_sticky OR lsb_y0).
        let sor = cols.eff_sticky + lsb_y0.clone() - cols.eff_sticky * lsb_y0;
        builder
            .when(s)
            .assert_eq(cols.round_up, cols.sqrt_shrn.guard * sor);
        builder.assert_bool(cols.round_up);

        // ── Output reconstruction ───────────────────────────────────────
        // y[2] has implicit bit in bit 7. Extract top7 = y[2] - 128.
        // Carry chain: y[0] + round_up → rc0; y[1] + rc0 → rc1; (y[2]-128) + rc1 → rc2/rem.
        builder.assert_bool(cols.rc0);
        builder.assert_bool(cols.rc1);
        builder.assert_bool(cols.rc2);
        builder.when(s).assert_eq(
            cols.result[0] + cols.rc0 * f256,
            cols.sqrt_shrn.y[0] + cols.round_up,
        );
        builder.when(s).assert_eq(
            cols.result[1] + cols.rc1 * f256,
            cols.sqrt_shrn.y[1] + cols.rc0,
        );
        let top7 = cols.sqrt_shrn.y[2] - f128;
        // Pin implicit bit in y[2]: RC top7 to 7 bits (same class as fadd/fma
        // cancel normalize — y[2]=127 with rc1=1 would otherwise satisfy top7c).
        builder
            .when(s)
            .assert_eq(top7.clone() + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus.range_check(top7, 7).eval(builder, s.into());
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s.into());

        // Result exponent: a_hi + 63 + is_hi + rc2.
        let res_exp = cols.eff_exp_hi + AB::Expr::from(f63) + cols.is_hi + cols.rc2;
        builder
            .when(s)
            .assert_eq(cols.r_exp_lo + cols.r_exp_hi * f2, res_exp);

        // Result bytes.
        builder
            .when(s)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        builder.when(s).assert_eq(cols.result[3], cols.r_exp_hi);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FsqrtCoreRecord {
    pub a: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FsqrtExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub read_as: u32,
    pub write_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FsqrtExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FsqrtCoreRecord),
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
        debug_assert_eq!(FpOpcode::from_usize(local), FpOpcode::FSQRT_S);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;

        let result = fsqrt_s(u32::from_le_bytes(a));
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

pub struct FsqrtFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FsqrtFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FsqrtFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FsqrtCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FsqrtCoreCols<F> = core_row.borrow_mut();
        let a_bits = u32::from_le_bytes(record.a);
        fill_fsqrt_witness(core_row, a_bits, fsqrt_s(a_bits), &self.range_checker);
    }
}

/// Fill a complete FSQRT.S core witness row (production filler + corruption tests).
fn fill_fsqrt_witness<F: PrimeField32>(
    core_row: &mut FsqrtCoreCols<F>,
    a_bits: u32,
    result: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    for i in 0..4 {
        core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
        core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
        range_checker.add_count((result >> (i * 8)) & 0xFF, 8);
    }

    let (a_sign, a_exp, a_mant) = decompose(a_bits);
    core_row.a_sign = F::from_u32(a_sign);
    core_row.a_exp_lo = F::from_u32(a_exp & 1);
    core_row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
    core_row.a_hi = F::from_u32(a_exp >> 1);
    range_checker.add_count((a_mant >> 16) & 0x7F, 7);
    range_checker.add_count(a_exp >> 1, 7);

    let a_exp_f = F::from_u32(a_exp);
    core_row.a_exp_is_zero = F::from_bool(a_exp == 0);
    core_row.a_exp_is_max = F::from_bool(a_exp == 255);
    core_row.a_exp_zero_inv = if a_exp != 0 {
        a_exp_f.try_inverse().unwrap_or(F::ZERO)
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
    let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
    core_row.a_mant_is_zero = F::from_bool(a_mant == 0);
    core_row.a_mant_sum_inv = if a_mant != 0 {
        F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    let a_nan = a_exp == 255 && a_mant != 0;
    let a_inf = a_exp == 255 && a_mant == 0;
    let a_zero = a_exp == 0 && a_mant == 0;
    let canon_sel = a_nan || (a_sign == 1 && !a_zero);
    let inf_res = a_inf && a_sign == 0;
    let zero_res = a_zero;

    core_row.a_nan = F::from_bool(a_nan);
    core_row.a_inf = F::from_bool(a_inf);
    core_row.a_zero = F::from_bool(a_zero);
    core_row.canon_sel = F::from_bool(canon_sel);
    core_row.inf_res = F::from_bool(inf_res);
    core_row.zero_res = F::from_bool(zero_res);
    core_row.is_special = F::from_bool(canon_sel || inf_res || zero_res);

    let a_normal = (1..=254).contains(&a_exp);
    let a_subnormal = a_exp == 0 && a_mant != 0 && a_sign == 0;
    let s_finite = !canon_sel && !inf_res && !zero_res && (a_normal || a_subnormal);
    core_row.s_finite = F::from_bool(s_finite);
    core_row.a_normal = F::from_bool(a_normal);
    core_row.a_subnormal = F::from_bool(a_subnormal);

    if s_finite {
        fill_finite_sqrt(core_row, a_exp, a_mant, result, range_checker);
    } else {
        zero_finite_columns(core_row);
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

/// floor(sqrt(n)) for n < 2^128.
fn isqrt_u128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut x = 1u128 << ((128 - n.leading_zeros() as usize).div_ceil(2));
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            break;
        }
        x = y;
    }
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

/// Zero all finite-path columns (for non-finite rows).
fn zero_finite_columns<F: PrimeField32>(row: &mut FsqrtCoreCols<F>) {
    row.s_finite = F::ZERO;
    row.a_subnormal = F::ZERO;
    row.eff_exp_odd = F::ZERO;
    row.eff_exp_hi = F::ZERO;
    row.norm_shift = F::ZERO;
    row.ns_lo = F::ZERO;
    row.ns_half = F::ZERO;
    row.a_sig2_lo7 = F::ZERO;
    row.norm_shln = ShlnCols::<F, 4>::default();
    for i in 0..3 {
        row.a_sig[i] = F::ZERO;
        row.a_sig_hi[i] = F::ZERO;
    }
    for i in 0..5 {
        row.q[i] = F::ZERO;
        row.q_hi[i] = F::ZERO;
    }
    row.q4_lo4 = F::ZERO;
    row.is_hi = F::ZERO;
    for i in 0..6 {
        row.rem[i] = F::ZERO;
        row.twoq[i] = F::ZERO;
        row.lt_borrow[i] = F::ZERO;
        row.diff[i] = F::ZERO;
    }
    row.rem_nz = F::ZERO;
    row.rem_nz_inv = F::ZERO;
    for i in 0..9 {
        row.sq_carry[i] = F::ZERO;
    }
    row.sqrt_shrn = ShrnCols::<F, 7>::default();
    row.sqrt_shift = F::ZERO;
    row.y0_high7 = F::ZERO;
    row.eff_sticky = F::ZERO;
    row.round_up = F::ZERO;
    row.rc0 = F::ZERO;
    row.rc1 = F::ZERO;
    row.rc2 = F::ZERO;
    row.rc2_rem = F::ZERO;
    row.r_exp_lo = F::ZERO;
    row.r_exp_hi = F::ZERO;
}

/// Fill the finite square-root witness columns.
fn fill_finite_sqrt<F: PrimeField32>(
    row: &mut FsqrtCoreCols<F>,
    a_exp: u32,
    a_mant: u32,
    result: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let is_sub = a_exp == 0;

    // ── Normalize significand ───────────────────────────────────────
    let (a_sig, eff_exp_odd_u32, res_exp_base) = if is_sub {
        let msb = 31 - a_mant.leading_zeros();
        let ns = 23 - msb; // norm_shift ∈ [1, 23]
        let sig = a_mant << ns;
        let ns_lo = ns & 1;
        let eo = 1 - ns_lo;
        let reb = 63i32 - (ns >> 1) as i32; // res_exp = reb + is_hi + rc2
        (sig, eo, reb)
    } else {
        (a_mant | 0x800000, a_exp & 1, (a_exp >> 1) as i32 + 63)
    };

    // ── Set normalization columns ───────────────────────────────────
    row.eff_exp_odd = F::from_u32(eff_exp_odd_u32);
    if is_sub {
        let ns = 23 - (31 - a_mant.leading_zeros());
        row.norm_shift = F::from_u32(ns);
        row.ns_lo = F::from_u32(ns & 1);
        row.ns_half = F::from_u32(ns >> 1);
        range_checker.add_count(ns >> 1, 4);
        row.eff_exp_hi = F::ZERO - F::from_u32(ns >> 1);
    } else {
        row.norm_shift = F::ZERO;
        row.ns_lo = F::ZERO;
        row.ns_half = F::ZERO;
        row.eff_exp_hi = F::from_u32(a_exp >> 1);
    }

    // ── ShlnGadget columns ──────────────────────────────────────────
    // x = mantissa bytes [a_mant&0xFF, (a_mant>>8)&0xFF, (a_mant>>16)&0x7F, 0].
    row.norm_shln = ShlnCols::<F, 4>::default();
    row.norm_shln.x[0] = F::from_u32(a_mant & 0xFF);
    row.norm_shln.x[1] = F::from_u32((a_mant >> 8) & 0xFF);
    row.norm_shln.x[2] = F::from_u32((a_mant >> 16) & 0x7F);
    row.norm_shln.x[3] = F::ZERO;
    if is_sub {
        let ns = row.norm_shift.as_canonical_u32() as usize;
        fill_shln_cols(&mut row.norm_shln, ns, range_checker);
    }

    let asig_bytes: [u32; 3] = [a_sig & 0xFF, (a_sig >> 8) & 0xFF, (a_sig >> 16) & 0xFF];

    for (i, &sb) in asig_bytes.iter().enumerate() {
        row.a_sig[i] = F::from_u32(sb);
        row.a_sig_hi[i] = F::from_u32(sb >> 7);
        range_checker.add_count(sb, 8);
        range_checker.add_count(sb & 0x7F, 7);
    }
    if is_sub {
        row.a_sig2_lo7 = F::from_u32(asig_bytes[2] - 128);
        range_checker.add_count(asig_bytes[2] - 128, 7);
    }

    // sig_shifted = a_sig << eff_exp_odd. n = sig_shifted << 48.
    let sig_shifted = (a_sig as u64) << eff_exp_odd_u32;
    let n = (sig_shifted as u128) << 48;

    // q = floor(sqrt(n)), rem = n - q².
    let q_val = isqrt_u128(n);
    let q_sq = q_val * q_val;
    let rem_val = n - q_sq;

    let q_bytes: [u32; 5] = core::array::from_fn(|i| ((q_val >> (i * 8)) & 0xFF) as u32);
    let rem_bytes: [u32; 6] = core::array::from_fn(|i| ((rem_val >> (i * 8)) & 0xFF) as u32);

    for (i, &qb) in q_bytes.iter().enumerate() {
        row.q[i] = F::from_u32(qb);
        row.q_hi[i] = F::from_u32(qb >> 7);
        range_checker.add_count(qb, 8);
        range_checker.add_count(qb & 0x7F, 7);
    }

    // q[4] < 32 (q < 2^37). Decompose: q[4] = q4_lo4 + is_hi * 16.
    let is_hi_val = q_bytes[4] >= 16;
    row.q4_lo4 = F::from_u32(q_bytes[4] & 0xF);
    row.is_hi = F::from_bool(is_hi_val);
    range_checker.add_count(q_bytes[4] & 0xF, 4);

    for (i, &rb) in rem_bytes.iter().enumerate() {
        row.rem[i] = F::from_u32(rb);
        range_checker.add_count(rb, 8);
    }

    // ── n bytes: n[0..6] = 0, n[6..10] from shifted sig ────────────
    let lo7 = |i: usize| asig_bytes[i] & 0x7F;
    let n_bytes: [u32; 10] = {
        let mut nb = [0u32; 10];
        if eff_exp_odd_u32 == 0 {
            nb[6] = asig_bytes[0];
            nb[7] = asig_bytes[1];
            nb[8] = asig_bytes[2];
            nb[9] = 0;
        } else {
            nb[6] = lo7(0) * 2;
            nb[7] = lo7(1) * 2 + (asig_bytes[0] >> 7);
            nb[8] = lo7(2) * 2 + (asig_bytes[1] >> 7);
            nb[9] = asig_bytes[2] >> 7;
        }
        nb
    };

    // ── Carry chain: q² + rem = n (10 bytes) ────────────────────────
    let q64: [u64; 5] = [
        q_bytes[0] as u64,
        q_bytes[1] as u64,
        q_bytes[2] as u64,
        q_bytes[3] as u64,
        q_bytes[4] as u64,
    ];
    let mut carry_in: u64 = 0;
    for i in 0..10 {
        let mut conv: u64 = 0;
        for k in 0..5 {
            if k <= i && (i - k) < 5 {
                conv += q64[k] * q64[i - k];
            }
        }
        let rem_byte = if i < 6 { rem_bytes[i] as u64 } else { 0 };
        let expected = carry_in + conv + rem_byte;
        let carry_out = (expected - n_bytes[i] as u64) / 256;
        debug_assert_eq!(
            expected - n_bytes[i] as u64,
            carry_out * 256,
            "fsqrt carry byte {}",
            i
        );
        if i < 9 {
            row.sq_carry[i] = F::from_u32(carry_out as u32);
            range_checker.add_count(carry_out as u32, 12);
        } else {
            debug_assert_eq!(carry_out, 0, "fsqrt final carry nonzero");
        }
        carry_in = carry_out;
    }

    // ── 2q bytes and borrow chain: 2q >= rem ────────────────────────
    let twoq_bytes: [u32; 6] = {
        let mut tq = [0u32; 6];
        let mut carry = 0u32;
        for i in 0..5 {
            let val = q_bytes[i] * 2 + carry;
            tq[i] = val & 0xFF;
            carry = val >> 8;
        }
        tq[5] = carry;
        tq
    };
    for (i, &tq) in twoq_bytes.iter().enumerate() {
        row.twoq[i] = F::from_u32(tq);
        range_checker.add_count(tq, 8);
    }
    let mut borrow: i64 = 0;
    for i in 0..6 {
        let d = twoq_bytes[i] as i64 - rem_bytes[i] as i64 - borrow;
        let (bv, nb) = if d < 0 { (d + 256, 1) } else { (d, 0) };
        row.lt_borrow[i] = F::from_u32(nb);
        row.diff[i] = F::from_u32(bv as u32);
        range_checker.add_count(bv as u32, 8);
        borrow = nb as i64;
    }
    debug_assert_eq!(borrow, 0, "fsqrt rem > 2q");

    // ── rem_nz ──────────────────────────────────────────────────────
    let rem_sum: u32 = rem_bytes.iter().sum();
    let rem_nz = rem_sum != 0;
    row.rem_nz = F::from_bool(rem_nz);
    row.rem_nz_inv = if rem_nz {
        F::from_u32(rem_sum).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };

    // ── Mantissa extraction via ShrnGadget<7> ───────────────────────
    let sqrt_shift = 12 + is_hi_val as usize;
    row.sqrt_shift = F::from_u32(sqrt_shift as u32);
    let x: [u32; 7] = core::array::from_fn(|i| if i < 5 { q_bytes[i] } else { 0 });
    row.sqrt_shrn = make_shrn_cols7::<F>(x, sqrt_shift, range_checker);

    let y0 = row.sqrt_shrn.y[0].as_canonical_u32();
    let y1 = row.sqrt_shrn.y[1].as_canonical_u32();
    let y2 = row.sqrt_shrn.y[2].as_canonical_u32();
    row.y0_high7 = F::from_u32(y0 >> 1);
    range_checker.add_count(y0 >> 1, 7);

    let guard = row.sqrt_shrn.guard.as_canonical_u32();
    let sticky = row.sqrt_shrn.sticky.as_canonical_u32() != 0;
    let lsb_y0 = y0 & 1;
    let eff_sticky = sticky || rem_nz;
    row.eff_sticky = F::from_bool(eff_sticky);

    let round_up = guard == 1 && (eff_sticky || lsb_y0 == 1);
    row.round_up = F::from_bool(round_up);

    // ── Output reconstruction ───────────────────────────────────────
    let s0 = y0 + round_up as u32;
    let rc0 = s0 >> 8;
    let s1 = y1 + rc0;
    let rc1 = s1 >> 8;
    let top7 = y2 - 128; // remove implicit bit
    let top7c = top7 + rc1;
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    row.rc0 = F::from_u32(rc0);
    row.rc1 = F::from_u32(rc1);
    row.rc2 = F::from_u32(rc2);
    row.rc2_rem = F::from_u32(rc2_rem);
    range_checker.add_count(rc2_rem, 7);
    range_checker.add_count(top7, 7);

    let res_exp = (res_exp_base + is_hi_val as i32 + rc2 as i32) as u32;
    row.r_exp_lo = F::from_u32(res_exp & 1);
    row.r_exp_hi = F::from_u32(res_exp >> 1);

    // Sanity checks.
    let r_exp_field = (result >> 23) & 0xFF;
    debug_assert_eq!(
        res_exp, r_exp_field,
        "fsqrt exp mismatch: a_exp={} res_exp={} field={}",
        a_exp, res_exp, r_exp_field
    );
    debug_assert_eq!(s0 & 0xFF, result & 0xFF, "fsqrt byte0");
    debug_assert_eq!(s1 & 0xFF, (result >> 8) & 0xFF, "fsqrt byte1");
    debug_assert_eq!(
        (rc2_rem + (res_exp & 1) * 128) & 0xFF,
        (result >> 16) & 0xFF,
        "fsqrt byte2"
    );
    debug_assert_eq!(res_exp >> 1, (result >> 24) & 0x7F, "fsqrt byte3");
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

/// Fill ShlnCols<4> witness for subnormal normalization.
fn fill_shln_cols<F: PrimeField32>(
    cols: &mut ShlnCols<F, 4>,
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    const N: usize = 4;
    let shift = shift.min(N * SHLIMB_BITS - 1);
    let x: [u32; N] = core::array::from_fn(|i| cols.x[i].as_canonical_u32());
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
    cols.shift = F::from_u32(shift as u32);
    cols.enable = F::ONE;
    cols.limb_shift_marker = limb_shift_marker;
    cols.bit_shift_marker = bit_shift_marker;
    cols.bit_mult = F::from_u32(1u32 << r);
    cols.bit_shift_carry = core::array::from_fn(|i| F::from_u32(carry_val(x[i])));
    cols.y = core::array::from_fn(|i| F::from_u32(y[i]));
}

#[cfg(test)]
mod finite_numeric_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;

    fn run(a: u32) -> NumericBuilder<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let mut row = FsqrtCoreCols::<F>::default();
        let result = fsqrt_s(a);
        let (a_sign, a_exp, a_mant) = decompose(a);
        let a_nan = a_exp == 255 && a_mant != 0;
        let a_inf = a_exp == 255 && a_mant == 0;
        let a_zero = a_exp == 0 && a_mant == 0;
        let canon_sel = a_nan || (a_sign == 1 && !a_zero);
        let inf_res = a_inf && a_sign == 0;
        let zero_res = a_zero;
        let is_special = canon_sel || inf_res || zero_res;
        let a_normal = (1..=254).contains(&a_exp);
        let s_finite = !is_special && a_normal && a_sign == 0;

        row.is_valid = F::ONE;
        row.s_finite = F::from_bool(s_finite);
        row.a_normal = F::from_bool(a_normal);
        row.is_special = F::from_bool(is_special);
        row.a_sign = F::from_u32(a_sign);
        row.a_exp_lo = F::from_u32(a_exp & 1);
        row.a_hi = F::from_u32(a_exp >> 1);
        row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        row.a_exp_is_zero = F::from_bool(a_exp == 0);
        row.a_exp_is_max = F::from_bool(a_exp == 255);
        row.a_exp_zero_inv = if a_exp != 0 {
            F::from_u32(a_exp).try_inverse().unwrap_or(F::ZERO)
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
        let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
        row.a_mant_is_zero = F::from_bool(a_mant == 0);
        row.a_mant_sum_inv = if a_mant != 0 {
            F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.a_nan = F::from_bool(a_nan);
        row.a_inf = F::from_bool(a_inf);
        row.a_zero = F::from_bool(a_zero);
        row.canon_sel = F::from_bool(canon_sel);
        row.inf_res = F::from_bool(inf_res);
        row.zero_res = F::from_bool(zero_res);
        for i in 0..4 {
            row.a[i] = F::from_u32((a >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
        if s_finite {
            fill_finite_sqrt(&mut row, a_exp, a_mant, result, &chip);
        }
        let air = FsqrtCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0);
        let mut builder = NumericBuilder::new(vec![]);
        air.eval_finite_sqrt(&mut builder, &row);
        builder
    }

    #[test]
    fn fsqrt_finite_numeric_passes() {
        for &a in &[
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            4.0f32.to_bits(),
            0.25f32.to_bits(),
            1e10f32.to_bits(),
            1.5f32.to_bits(),
            3.0f32.to_bits(),
            100.0f32.to_bits(),
            f32::MAX.to_bits(),
            0x3F800000, // 1.0
            0x40400000, // 3.0
            0x7F7FFFFF, // MAX
        ] {
            let builder = run(a);
            assert!(
                builder.ok(),
                "fsqrt failed: {:?}\n a={:08x} result={:08x}",
                builder.failures,
                a,
                fsqrt_s(a)
            );
        }
        let mut rng = 0xB0B0_005Eu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..2000 {
            let a_exp = (next() % 254) + 1;
            let a = (a_exp << 23) | (next() & 0x7FFFFF);
            let builder = run(a);
            if !builder.ok() {
                panic!(
                    "fsqrt fuzz failed: {:?}\n a={:08x} result={:08x}",
                    builder.failures,
                    a,
                    fsqrt_s(a)
                );
            }
        }
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
    type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 1, 1, 4, 4>;

    fn rc() -> SharedVariableRangeCheckerChip {
        Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )))
    }

    fn air() -> FsqrtCoreAir {
        FsqrtCoreAir::new(VariableRangeCheckerBus::new(0, 16), 0)
    }

    fn filler_cols(a: u32) -> FsqrtCoreCols<F> {
        let chip = rc();
        let mut cols = FsqrtCoreCols::<F>::default();
        fill_fsqrt_witness(&mut cols, a, fsqrt_s(a), &chip);
        cols
    }

    /// Result bytes, routing selectors, and newly pinned a_sig_hi/q_hi must reject +1.
    #[test]
    fn corruption_sweep_pins_result_and_selectors() {
        let names = FsqrtCoreCols::<F>::struct_reflection().unwrap_or_default();
        let eval_full = |b: &mut crate::test_utils::NumericBuilder<F>, cols: &[F]| {
            VmCoreAir::<crate::test_utils::NumericBuilder<F>, TestI>::eval(
                &air(),
                b,
                cols,
                F::ZERO,
            );
        };
        // a_sig_hi only enters the radicand algebraically when eff_exp_odd=1;
        // NumericBuilder ignores the range-bus half of the half-decomp, so pin
        // a_sig_hi on an odd-exp input. q_hi always feeds twoq, so pin on both.
        let classes: &[(u32, &[&str])] = &[
            (
                4.0f32.to_bits(), // exp 129 odd → a_sig_hi live
                &[
                    "is_hi",
                    "round_up",
                    "a_sig_hi__0",
                    "a_sig_hi__1",
                    "a_sig_hi__2",
                    "q_hi__0",
                    "q_hi__1",
                    "q_hi__2",
                    "q_hi__3",
                    "q_hi__4",
                ][..],
            ),
            (
                2.0f32.to_bits(), // exp 128 even → a_sig_hi not algebraically live
                &["q_hi__0", "q_hi__1", "q_hi__2", "q_hi__3", "q_hi__4"][..],
            ),
            (0x7F800000, &[]), // +inf
            (0x7FC00000, &[]), // NaN
            (0x00000000, &[]), // +0
            (0x80000000, &[]), // -0
            (0xBF800000, &[]), // -1 -> canon NaN
            (0x00400000, &[]), // positive subnormal
        ];
        for &(a, selectors) in classes {
            let cols = filler_cols(a);
            let mut buf = vec![F::ZERO; FsqrtCoreCols::<F>::width()];
            *buf.as_mut_slice().borrow_mut() = cols;
            let mut bld = crate::test_utils::NumericBuilder::new(vec![]);
            VmCoreAir::<crate::test_utils::NumericBuilder<F>, TestI>::eval(
                &air(),
                &mut bld,
                &buf,
                F::ZERO,
            );
            assert!(bld.ok(), "honest fsqrt a={a:08x}: {:?}", bld.failures);
            let mut pinned = vec!["result__0", "result__1", "result__2", "result__3"];
            pinned.extend_from_slice(selectors);
            assert_columns_constrained(
                &buf,
                eval_full,
                &names,
                &pinned,
                &format!("fsqrt a={a:08x}"),
            );
        }
    }
}
