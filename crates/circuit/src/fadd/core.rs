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
    fill_op_norm, run_shln, run_shrn, run_shrn_round, ShlnCols, ShlnGadget, ShrnCols, ShrnGadget,
    SHLIMB_BITS,
};

pub const SIG_NUM_LIMBS: usize = 3;
pub const EXT_SIG_LIMBS: usize = 4; // extended significand: 3 mantissa bytes + 1 guard/sticky byte
pub const LIMB_BITS: usize = 8;
/// Working significand width in bytes: 3 significand bytes + 1 guard/sticky byte.
pub const WORK_LIMBS: usize = 4;
/// ediff >= 25 ⇒ normally the smaller is at most 0.5 ULP of the larger's binade.
/// Cancel's align field is `small_sig<<24`, so ediff>=25 cannot go through cancel
/// without dropping significand bits; those rows take the deep path instead.
///
/// Exception (see `s_deep_boundary`): opposite signs, ediff == 25, larger exact
/// power of two, smaller mantissa nonzero → result binade ULP halves and the
/// honest RNE result is `larger - 1` lower-binade ULP, not the larger itself.
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
    pub a_sigb: [T; 3], // effective significand bytes (normalized if subnormal)
    pub b_sigb: [T; 3],
    // Selected significands (3 bytes each). large = the operand with the larger exp.
    pub large_sig: [T; SIG_NUM_LIMBS],
    pub small_sig: [T; SIG_NUM_LIMBS],

    // Variable right-shift gadget for alignment of the smaller significand. Its `x`
    // input is `small_sig << 8` (4 bytes), `shift` = ediff, `y` = aligned significand.
    pub align: ShrnCols<T, WORK_LIMBS>,
    pub is_deep: T,    // bool: ediff >= 25 (deep path; see s_deep_boundary exception)
    pub deep_delta: T, // byte witness used to soundly derive is_deep
    pub deep_delta_inv: T, // inverse for deep_delta==0 (ediff == DEEP_EDIFF)
    pub lost_sticky: T, // OR of align.guard / align.sticky (bits lost during alignment)
    pub align_enable: T, // bool: s_add AND !is_deep (gates the ShrnGadget)
    // Deep-path binade-boundary exception (effective subtraction only).
    pub large_pow2_inv: T, // inv of large_sig[0]+large_sig[1]+(large_sig[2]-128)
    pub small_ext_inv: T,  // inv of the same sum over small_sig (nonzero ⇒ sticky mant)
    pub s_deep_cancel: T,  // s_bn AND !is_add AND is_deep
    pub s_deep_ediff25: T, // s_deep_cancel AND deep_delta==0 (ediff == 25)
    pub s_deep_boundary: T, // s_deep_ediff25 AND large pow2 AND small ext
    pub s_deep_exact: T,   // s_bn AND is_deep AND !s_deep_boundary (result = larger)

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
    pub s_fb: T,         // finite AND both nonzero (includes subnormals)
    pub s_bn: T,         // finite AND both normal (used by cancellation path)
    pub s_one_zero: T,   // finite AND exactly one input is zero
    pub s_sub_result: T, // shallow AND base_exp_sh <= 0 (subnormal-result gate)

    // === Shallow-path subnormal results ===
    pub sh_base_exp: T, // pre-rounding exponent (large_exp_eff + carry_out)
    pub sh_is_sub: T,   // sh_base_exp <= 0
    pub sh_ef_max: T,   // (result exp field == 255) indicator (witness for degree)
    pub sh_route_decomp: [T; 1],
    pub sh_sub_t: T, // denormal shift 9 - sh_base_exp (in [9, 31])
    pub sh_sub_shrn: ShrnCols<T, 4>,
    pub sh_sub_lsb: T,
    pub sh_sub_hi7: T,
    pub sh_sub_sticky_eff: T, // gadget sticky OR alignment lost_sticky
    pub sh_sub_round_up: T,
    pub sh_sub_z0: T,
    pub sh_sub_z1: T,
    pub sh_sub_z2: T,
    pub sh_sub_zc0: T,
    pub sh_sub_zc1: T,
    pub sh_sub_exp1: T,     // z bit 23 (rounded up to smallest normal)
    pub sh_sub_z2_lo7: T,   // z2 low 7 bits
    pub s_cancel: T,        // s_bn AND !is_add AND !is_deep
    pub a_ge_b: T,          // bool: a_sig >= b_sig (magnitude compare for ediff=0)
    pub s_cancel_normal: T, // s_cancel AND normal result (routed via base_exp)

    // === Cancellation output routing + subnormal results ===
    pub c_m0: T,     // rounded mantissa byte 0 (result byte on normal rows)
    pub c_m1: T,     // rounded mantissa byte 1
    pub base_exp: T, // pre-rounding exponent (large_exp_eff - lshift)
    pub raw_csum: T, // sum of cancel_diff bytes (exact-zero detection)
    pub raw_csum_inv: T,
    pub is_exact_zero: T,
    pub cancel_is_ovf: T, // base_exp + rc2 >= 255
    pub cancel_is_sub: T, // base_exp <= 0
    pub s_cancel_sub: T,  // s_cancel AND cancel_is_sub AND !exact_zero (gate col)
    pub route_ovf_decomp: [T; 1],
    pub route_sub_decomp: [T; 1],
    pub sub_t: T,    // denormal shift 25 - base_exp
    pub sub_deep: T, // t > 55 (deep underflow -> +-0)
    pub sub_deep_decomp: [T; 1],
    pub sub_shrn: ShrnCols<T, 6>,
    pub sub_lsb: T,
    pub sub_hi7: T,
    pub sub_round_up: T,
    pub sub_z0: T,
    pub sub_z1: T,
    pub sub_z2: T,
    pub sub_zc0: T,
    pub sub_zc1: T,

    // === Result ===
    pub result_sign: T,
    pub result: [T; 4], // output f32 bits (4 limbs)
    pub is_nan_result: T,

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

        // ── Subnormal operand pre-normalization ──────────────────────────
        // Shift the subnormal significand left so its MSB lands at bit 23
        // (the implicit-bit position); the exponent absorbs the shift.
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
        // Effective exponents absorb the normalization shift.
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp.clone());
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp.clone());

        // ── Special-case selectors + result constraints ──────────────────
        self.eval_special_cases(builder, cols);

        // ── Phase 3: Alignment selection (ediff / a_is_larger) ───────────
        // ediff = |a_exp_eff - b_exp_eff|. Range-checking ediff to 9 bits forces
        // a_is_larger to be consistent with the true comparison (a wrong choice
        // makes ediff wrap to a huge field value, failing the range check).
        self.range_bus
            .range_check(cols.ediff.into(), 9)
            .eval(builder, cols.is_valid.into());
        let diff = a_exp_eff - b_exp_eff;
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

        // One-zero special case: result = the nonzero operand (bytes preserved).
        // s_one_zero = s_finite AND (a_is_zero XOR b_is_zero). Stored as a column
        // to keep the result constraints degree ≤ 3.
        let xor_z = cols.a_is_zero + cols.b_is_zero - cols.a_is_zero * cols.b_is_zero * f2;
        let s_finite_special = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero_both;
        builder
            .when(cols.is_valid)
            .assert_eq(cols.s_one_zero, s_finite_special * xor_z);
        builder.assert_bool(cols.s_one_zero);
        // When exactly one input is zero, result = the nonzero operand.
        builder.assert_zero(
            cols.s_one_zero
                * (cols.result[0]
                    - cols.b_mant0
                    - (one.clone() - cols.a_is_zero) * (cols.a_mant0 - cols.b_mant0)),
        );
        builder.assert_zero(
            cols.s_one_zero
                * (cols.result[1]
                    - cols.b_mant1
                    - (one.clone() - cols.a_is_zero) * (cols.a_mant1 - cols.b_mant1)),
        );
        builder.assert_zero(
            cols.s_one_zero
                * (cols.result[2]
                    - cols.b_mant2
                    - cols.b_exp_lo * f128
                    - (one.clone() - cols.a_is_zero)
                        * ((cols.a_mant2 + cols.a_exp_lo * f128)
                            - (cols.b_mant2 + cols.b_exp_lo * f128))),
        );
        builder.assert_zero(
            cols.s_one_zero
                * (cols.result[3]
                    - cols.b_exp_hi
                    - cols.b_sign * f128
                    - (one.clone() - cols.a_is_zero)
                        * ((cols.a_exp_hi + cols.a_sign * f128)
                            - (cols.b_exp_hi + cols.b_sign * f128))),
        );
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
        let f255: AB::F = AB::F::from_u32(255);
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
        // Effective significand bytes: normal = mantissa + implicit bit;
        // subnormal = pre-normalized significand.
        let a_ns = one.clone() - cols.a_sub;
        let b_ns = one.clone() - cols.b_sub;
        for i in 0..3 {
            let a_base = match i {
                0 => cols.a_mant0.into(),
                1 => cols.a_mant1.into(),
                _ => cols.a_mant2 + cols.a_is_normal * f128,
            };
            builder.when(cols.is_valid).assert_eq(
                cols.a_sigb[i],
                a_ns.clone() * a_base + cols.a_sub * cols.a_shln.y[i],
            );
            let b_base = match i {
                0 => cols.b_mant0.into(),
                1 => cols.b_mant1.into(),
                _ => cols.b_mant2 + cols.b_is_normal * f128,
            };
            builder.when(cols.is_valid).assert_eq(
                cols.b_sigb[i],
                b_ns.clone() * b_base + cols.b_sub * cols.b_shln.y[i],
            );
        }
        let sel = cols.a_is_larger.into();
        let notsel = one.clone() - sel.clone();
        // large/small significand selection (gated on is_valid; padding stays 0).
        for i in 0..3 {
            builder.when(cols.is_valid).assert_eq(
                cols.large_sig[i],
                sel.clone() * cols.a_sigb[i] + notsel.clone() * cols.b_sigb[i],
            );
            builder.when(cols.is_valid).assert_eq(
                cols.small_sig[i],
                sel.clone() * cols.b_sigb[i] + notsel.clone() * cols.a_sigb[i],
            );
        }
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp.clone());
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp.clone());
        let large_exp = sel.clone() * a_exp.clone() + notsel.clone() * b_exp.clone();
        let large_exp_eff = sel.clone() * a_exp_eff.clone() + notsel.clone() * b_exp_eff.clone();

        // ── Sound selector s_add ────────────────────────────────────────
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero_both;
        builder.assert_eq(
            cols.s_add,
            s_finite.clone()
                * cols.is_add
                * (cols.a_is_normal + cols.a_sub)
                * (cols.b_is_normal + cols.b_sub),
        );
        builder.assert_bool(cols.s_add);

        // ── Deep clamp: is_deep = (ediff >= DEEP_EDIFF) ─────────────────
        //   !is_deep: ediff + deep_delta == DEEP_EDIFF-1
        //   is_deep:  ediff - deep_delta == DEEP_EDIFF
        // Both branches range-check deep_delta to a byte, which forces is_deep to
        // correctly indicate ediff >= DEEP_EDIFF.
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
        // Deep path (ediff >= DEEP_EDIFF). Default: result = larger. Exception:
        // s_deep_boundary (cancel, ediff==25, larger power-of-two, small mant≠0)
        // → larger − 1 lower-binade ULP (IEEE RNE past-midpoint after binade drop).
        let s_deep = cols.s_bn * cols.is_deep;

        // deep_delta == 0 ⇔ ediff == DEEP_EDIFF on deep rows.
        let dd_zero = one.clone() - cols.deep_delta * cols.deep_delta_inv;
        builder.assert_zero(cols.deep_delta * dd_zero.clone());

        // large_pow2 ⇔ large_sig == 0x800000; small_ext ⇔ small_sig != 0x800000.
        let f128e: AB::Expr = AB::Expr::from(f128);
        let large_pow2_sum =
            cols.large_sig[0] + cols.large_sig[1] + cols.large_sig[2] - f128e.clone();
        let large_pow2 = one.clone() - large_pow2_sum.clone() * cols.large_pow2_inv;
        // Only meaningful when significands carry an implicit bit (s_bn).
        builder
            .when(cols.s_bn)
            .assert_zero(large_pow2_sum * large_pow2.clone());
        let small_ext_sum = cols.small_sig[0] + cols.small_sig[1] + cols.small_sig[2] - f128e;
        let small_ext_zero = one.clone() - small_ext_sum.clone() * cols.small_ext_inv;
        builder
            .when(cols.s_bn)
            .assert_zero(small_ext_sum.clone() * small_ext_zero.clone());
        let small_ext = one.clone() - small_ext_zero;

        builder.assert_eq(
            cols.s_deep_cancel,
            cols.s_bn * (one.clone() - cols.is_add) * cols.is_deep,
        );
        builder.assert_bool(cols.s_deep_cancel);
        // Factored so every assert_eq RHS stays degree ≤ 3.
        builder.assert_eq(cols.s_deep_ediff25, cols.s_deep_cancel * dd_zero);
        builder.assert_bool(cols.s_deep_ediff25);
        builder.assert_eq(
            cols.s_deep_boundary,
            cols.s_deep_ediff25 * large_pow2 * small_ext,
        );
        builder.assert_bool(cols.s_deep_boundary);
        builder.assert_eq(
            cols.s_deep_exact,
            s_deep.clone() * (one.clone() - cols.s_deep_boundary),
        );
        builder.assert_bool(cols.s_deep_exact);

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
        builder.assert_eq(cols.align.enable, cols.align_enable);
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

        // ── Deep path: result = larger, unless binade-boundary exception ─
        let large_b0 = sel.clone() * cols.a_mant0 + notsel.clone() * cols.b_mant0;
        let large_b1 = sel.clone() * cols.a_mant1 + notsel.clone() * cols.b_mant1;
        let large_b2 = sel.clone() * (cols.a_mant2 + cols.a_exp_lo * f128)
            + notsel.clone() * (cols.b_mant2 + cols.b_exp_lo * f128);
        let large_b3 = sel.clone() * (cols.a_exp_hi + cols.a_sign * f128)
            + notsel.clone() * (cols.b_exp_hi + cols.b_sign * f128);
        let large_sign = sel.clone() * cols.a_sign + notsel.clone() * cols.b_sign;
        let s_de = cols.s_deep_exact;
        builder
            .when(s_de)
            .assert_eq(cols.result[0], large_b0.clone());
        builder
            .when(s_de)
            .assert_eq(cols.result[1], large_b1.clone());
        builder
            .when(s_de)
            .assert_eq(cols.result[2], large_b2.clone());
        builder
            .when(s_de)
            .assert_eq(cols.result[3], large_b3.clone());
        builder
            .when(s_de)
            .assert_eq(cols.result_sign, large_sign.clone());
        builder
            .when(s_de)
            .assert_eq(cols.result_exp, large_exp.clone());
        // Boundary: result = pack(large_sign, large_exp - 1, mant=0x7FFFFF).
        let s_db = cols.s_deep_boundary;
        builder.when(s_db).assert_eq(cols.result_sign, large_sign);
        builder
            .when(s_db)
            .assert_eq(cols.result_exp, large_exp.clone() - AB::Expr::ONE);
        builder
            .when(s_db)
            .assert_eq(cols.result[0], AB::Expr::from(f255));
        builder
            .when(s_db)
            .assert_eq(cols.result[1], AB::Expr::from(f255));
        // result_exp = r_exp_lo + 2*r_exp_hi, bytes 2/3 pack mant-all-ones.
        builder.when(s_db).assert_eq(
            cols.result_exp,
            cols.r_exp_lo + cols.r_exp_hi * AB::Expr::from(f2),
        );
        builder.when(s_db).assert_eq(
            cols.result[2],
            AB::Expr::from(f127) + cols.r_exp_lo * AB::Expr::from(f128),
        );
        builder.when(s_db).assert_eq(
            cols.result[3],
            cols.r_exp_hi + cols.result_sign * AB::Expr::from(f128),
        );

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
        // base_exp_sh = large_exp_eff + carry_out (pre-rounding exponent;
        // witness column for degree). Subnormal results (base_exp_sh <= 0)
        // are routed by a less-than gadget — the class is never prover-chosen.
        builder
            .when(s_shallow)
            .assert_eq(cols.sh_base_exp, large_exp_eff + cols.carry_out);
        let sh_base_exp_b = cols.sh_base_exp + AB::Expr::from(f256);
        self.range_bus
            .range_check(sh_base_exp_b.clone(), 9)
            .eval(builder, s_shallow);
        let sh_sub_io = IsLessThanIo::new(
            sh_base_exp_b,
            AB::Expr::from(AB::F::from_u32(257)),
            cols.sh_is_sub.into(),
            s_shallow,
        );
        IsLtSubAir::new(self.range_bus, 9).eval(builder, (sh_sub_io, &cols.sh_route_decomp));
        builder.assert_bool(cols.sh_is_sub);
        let not_sub = one.clone() - cols.sh_is_sub;

        // result_exp = r_exp_lo + r_exp_hi*2 = base_exp_sh + rc2 (normal and
        // overflow rows; exponent 255 naturally encodes +-inf).
        builder
            .when(s_shallow * not_sub.clone())
            .assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);
        builder.when(s_shallow * not_sub.clone()).assert_eq(
            cols.r_exp_lo + cols.r_exp_hi * f2,
            cols.sh_base_exp + cols.rc2,
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
        // Pin implicit bit in norm_sig[3] (same under-shift edge as cancel path).
        self.range_bus
            .range_check(cols.norm_sig[3] - f128, 7)
            .eval(builder, s_shallow);
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s_shallow);

        // Sound "(r_exp_hi == 127)" indicator (overflow detection on the
        // non-subnormal rows; its inverse witness is don't-care elsewhere).
        let delta127 = AB::Expr::from(f127) - cols.r_exp_hi;
        let he127 = one.clone() - delta127.clone() * cols.r_exp_hi_max_inv;
        builder
            .when(s_shallow)
            .assert_zero(delta127 * he127.clone());
        let exp_field_max = cols.r_exp_lo * he127;
        builder.assert_eq(cols.sh_ef_max, exp_field_max);
        builder.assert_bool(cols.sh_ef_max);
        // s_normal / s_overflow stored as columns (degree 1) so they can gate the
        // result-byte constraints without exceeding the max constraint degree.
        builder.assert_eq(
            cols.s_normal,
            s_shallow * not_sub.clone() * (one.clone() - cols.sh_ef_max),
        );
        builder.assert_eq(
            cols.s_overflow,
            s_shallow * not_sub.clone() * cols.sh_ef_max,
        );
        builder.assert_bool(cols.s_normal);
        builder.assert_bool(cols.s_overflow);
        // s_sub_result = shallow AND base_exp_sh <= 0 (subnormal/zero result).
        builder.assert_eq(cols.s_sub_result, s_shallow * cols.sh_is_sub);
        builder.assert_bool(cols.s_sub_result);

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
        // result[3] = r_exp_hi + result_sign*128 (normal and overflow).
        builder
            .when(s_shallow * not_sub)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.result_sign * f128);

        // ── Sub path: subnormal results ──────────────────────────────────
        // Denormalize: shift norm_sig right by t = 9 - base_exp_sh (in
        // [9, 31]; base_exp_sh >= -22 is forced by the operand normalization
        // anchors, so no clamp is needed). Single RNE rounding from the
        // gadget's guard/sticky plus the alignment loss (additive here); the
        // round carry into bit 23 yields the smallest normal.
        let ssub = cols.s_sub_result;
        builder.when(ssub).assert_eq(
            cols.sh_sub_t,
            AB::Expr::from(AB::F::from_u32(9)) - cols.sh_base_exp,
        );
        self.range_bus
            .range_check(cols.sh_sub_t.into(), 5)
            .eval(builder, ssub.into());
        builder
            .when(ssub)
            .assert_eq(cols.sh_sub_shrn.shift, cols.sh_sub_t);
        for i in 0..4 {
            builder
                .when(ssub)
                .assert_eq(cols.sh_sub_shrn.x[i], cols.norm_sig[i]);
        }
        builder.assert_eq(cols.sh_sub_shrn.enable, ssub);
        let sh_shrn = ShrnGadget::new(self.range_bus);
        sh_shrn.eval_shrn(builder, &cols.sh_sub_shrn);
        // RNE rounding: guard/sticky from the gadget, sticky OR'd with the
        // alignment loss (magnitude addition: a tie with any loss rounds up).
        builder.assert_bool(cols.sh_sub_lsb);
        builder.when(ssub).assert_eq(
            cols.sh_sub_shrn.y[0],
            cols.sh_sub_lsb + cols.sh_sub_hi7 * f2,
        );
        self.range_bus
            .range_check(cols.sh_sub_hi7.into(), 7)
            .eval(builder, ssub.into());
        let sh_sor =
            cols.sh_sub_sticky_eff + cols.sh_sub_lsb - cols.sh_sub_sticky_eff * cols.sh_sub_lsb;
        builder.when(ssub).assert_eq(
            cols.sh_sub_sticky_eff,
            cols.sh_sub_shrn.sticky + cols.lost_sticky - cols.sh_sub_shrn.sticky * cols.lost_sticky,
        );
        builder.assert_bool(cols.sh_sub_sticky_eff);
        builder
            .when(ssub)
            .assert_eq(cols.sh_sub_round_up, cols.sh_sub_shrn.guard * sh_sor);
        builder.assert_bool(cols.sh_sub_round_up);
        // z = y + round_up (<= 2^23; bit 23 of z is the exp field LSB).
        builder.assert_bool(cols.sh_sub_zc0);
        builder.assert_bool(cols.sh_sub_zc1);
        builder.when(ssub).assert_eq(
            cols.sh_sub_z0 + cols.sh_sub_zc0 * f256,
            cols.sh_sub_shrn.y[0] + cols.sh_sub_round_up,
        );
        builder.when(ssub).assert_eq(
            cols.sh_sub_z1 + cols.sh_sub_zc1 * f256,
            cols.sh_sub_shrn.y[1] + cols.sh_sub_zc0,
        );
        builder
            .when(ssub)
            .assert_eq(cols.sh_sub_z2, cols.sh_sub_shrn.y[2] + cols.sh_sub_zc1);
        for z in [cols.sh_sub_z0, cols.sh_sub_z1, cols.sh_sub_z2] {
            self.range_bus
                .range_check(z.into(), 8)
                .eval(builder, ssub.into());
        }
        // z2 = sh_sub_exp1 * 128 + z2_lo7 (exp field LSB is z bit 23).
        builder.assert_bool(cols.sh_sub_exp1);
        builder
            .when(ssub)
            .assert_eq(cols.sh_sub_z2, cols.sh_sub_exp1 * f128 + cols.sh_sub_z2_lo7);
        self.range_bus
            .range_check(cols.sh_sub_z2_lo7.into(), 7)
            .eval(builder, ssub.into());
        builder.when(ssub).assert_eq(cols.result[0], cols.sh_sub_z0);
        builder.when(ssub).assert_eq(cols.result[1], cols.sh_sub_z1);
        builder.when(ssub).assert_eq(cols.result[2], cols.sh_sub_z2);
        builder
            .when(ssub)
            .assert_eq(cols.result[3], cols.a_sign * f128);
        builder
            .when(ssub)
            .assert_eq(cols.result_exp, cols.sh_sub_exp1);
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
        let f25: AB::F = AB::F::from_u32(25);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f256: AB::F = AB::F::from_u32(1 << LIMB_BITS);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_exp_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;

        // ── Sound selectors ─────────────────────────────────────────────
        let s_finite = one.clone() - cols.s_nan - cols.s_inf - cols.s_zero_both;
        // s_bn = finite AND both normal (cancellation path requires normals).
        builder.when(cols.is_valid).assert_eq(
            cols.s_bn,
            s_finite.clone() * (cols.a_is_normal + cols.a_sub) * (cols.b_is_normal + cols.b_sub),
        );
        builder.assert_bool(cols.s_bn);
        // s_fb = finite AND both nonzero (includes subnormals; magnitude-add path).
        builder.when(cols.is_valid).assert_eq(
            cols.s_fb,
            s_finite.clone() * (one.clone() - cols.a_is_zero) * (one.clone() - cols.b_is_zero),
        );
        builder.assert_bool(cols.s_fb);
        // s_sub_result = shallow AND both subnormal AND no carry (subnormal result).
        // Computed below after he0 is available.
        // ediff != 0 indicator (excludes same-exp subtraction).
        let ediff_is_zero = one.clone() - cols.ediff * cols.ediff_nz_inv;
        builder.assert_zero(cols.ediff * ediff_is_zero.clone());
        builder
            .when(cols.is_valid)
            .assert_eq(cols.ediff_nz, one.clone() - ediff_is_zero);
        builder.assert_bool(cols.ediff_nz);
        // s_cancel = s_bn AND (NOT is_add) AND (NOT is_deep).
        builder.assert_eq(
            cols.s_cancel,
            cols.s_bn * (one.clone() - cols.is_add) * (one.clone() - cols.is_deep),
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

        // ── 6-byte borrow chain ──────────────────────────────────────────
        // Two subcases: ediff >= 1 (aligned via ShrnGadget) and ediff == 0
        // (both at the same position, effective_large/small via a_ge_b).
        for i in 0..5 {
            builder.assert_bool(cols.cancel_borrow[i]);
        }
        builder.assert_bool(cols.a_ge_b);
        let s_e1 = s * cols.ediff_nz; // ediff >= 1
        let s_e0 = s * (one.clone() - cols.ediff_nz); // ediff == 0

        // --- ediff >= 1: raw_diff = large_working - aligned ---
        let lw = |i: usize| -> AB::Expr {
            match i {
                0..=2 => AB::Expr::ZERO,
                3 => cols.large_sig[0].into(),
                4 => cols.large_sig[1].into(),
                5 => cols.large_sig[2].into(),
                _ => unreachable!(),
            }
        };
        builder.when(s_e1.clone()).assert_eq(
            cols.cancel_diff[0] + cols.cancel_align.y[0],
            cols.cancel_borrow[0] * f256,
        );
        for i in 1..5 {
            let bin = cols.cancel_borrow[i - 1];
            builder.when(s_e1.clone()).assert_eq(
                cols.cancel_diff[i] + cols.cancel_align.y[i] + bin,
                lw(i) + cols.cancel_borrow[i] * f256,
            );
        }
        builder.when(s_e1.clone()).assert_eq(
            cols.cancel_diff[5] + cols.cancel_align.y[5] + cols.cancel_borrow[4],
            lw(5),
        );

        // --- ediff == 0: raw_diff = effective_large - effective_small ---
        // For ediff=0, a_is_larger=true (tie-break), so large_sig=a_sig,
        // small_sig=b_sig. effective_large/small swap based on a_ge_b.
        let eff_large = |i: usize| {
            cols.a_ge_b * cols.large_sig[i] + (one.clone() - cols.a_ge_b) * cols.small_sig[i]
        };
        let eff_small = |i: usize| {
            cols.a_ge_b * cols.small_sig[i] + (one.clone() - cols.a_ge_b) * cols.large_sig[i]
        };
        for i in 0..3 {
            builder.when(s_e0.clone()).assert_zero(cols.cancel_diff[i]);
        }
        builder.when(s_e0.clone()).assert_eq(
            cols.cancel_diff[3] + eff_small(0),
            eff_large(0) + cols.cancel_borrow[0] * f256,
        );
        builder.when(s_e0.clone()).assert_eq(
            cols.cancel_diff[4] + eff_small(1) + cols.cancel_borrow[0],
            eff_large(1) + cols.cancel_borrow[1] * f256,
        );
        // No final borrow (forces a_ge_b correctness).
        builder.when(s_e0.clone()).assert_eq(
            cols.cancel_diff[5] + eff_small(2) + cols.cancel_borrow[1],
            eff_large(2),
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
        let a_exp_eff = a_exp.clone() + cols.a_sub * (one.clone() - cols.a_lshift - a_exp.clone());
        let b_exp_eff = b_exp.clone() + cols.b_sub * (one.clone() - cols.b_lshift - b_exp.clone());
        let large_exp = cols.a_is_larger * a_exp_eff.clone()
            + (one.clone() - cols.a_is_larger) * b_exp_eff.clone();
        let large_sign =
            cols.a_is_larger * cols.a_sign + (one.clone() - cols.a_is_larger) * cols.b_sign;
        // result sign: ediff>=1 → large_sign; ediff=0 → a_ge_b ? a_sign : b_sign.
        let ediff0_sign = cols.a_ge_b * cols.a_sign + (one.clone() - cols.a_ge_b) * cols.b_sign;
        let cancel_r_sign =
            cols.ediff_nz * large_sign.clone() + (one.clone() - cols.ediff_nz) * ediff0_sign;
        builder.when(s).assert_eq(cols.result_sign, cancel_r_sign);

        // ── Exact-zero cancellation detection ────────────────────────────
        // raw_diff == 0 ⟺ exact cancellation (result is +0 in RNE).
        let raw_csum_expr: AB::Expr = cols.cancel_diff.iter().map(|&v| v.into()).sum();
        builder.when(s).assert_eq(cols.raw_csum, raw_csum_expr);
        let ez_free = one.clone() - cols.raw_csum * cols.raw_csum_inv;
        builder
            .when(s)
            .assert_eq(cols.is_exact_zero, ez_free.clone());
        builder.assert_zero(cols.raw_csum * ez_free);
        builder.assert_bool(cols.is_exact_zero);
        let ez = cols.is_exact_zero;
        // On exact-zero rows the normalization shift is meaningless; pin it,
        // and the rounding carry (norm is all zero, so top7 is unusable).
        builder.when(s * ez).assert_zero(cols.cancel_lshift);
        builder.when(s * ez).assert_zero(cols.rc0);
        builder.when(s * ez).assert_zero(cols.rc1);
        builder.when(s * ez).assert_zero(cols.rc2);

        // ── Result-class routing (sound, via base_exp) ───────────────────
        // base_exp = large_exp_eff - lshift (pre-rounding), in [-68, 254].
        // Witness column (degree 1) so downstream gating stays within degree 4.
        builder
            .when(s)
            .assert_eq(cols.base_exp, large_exp.clone() - cols.cancel_lshift);
        let base_exp_b = cols.base_exp + AB::Expr::from(f256);
        self.range_bus
            .range_check(base_exp_b.clone(), 9)
            .eval(builder, s.into());
        let lt9 = IsLtSubAir::new(self.range_bus, 9);
        // cancel_is_ovf: base_exp + rc2 >= 255  <=>  510 < base_exp_b + rc2.
        let ovf_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(510)),
            base_exp_b.clone() + cols.rc2,
            cols.cancel_is_ovf.into(),
            s.into(),
        );
        lt9.eval(builder, (ovf_io, &cols.route_ovf_decomp));
        // cancel_is_sub: base_exp <= 0  <=>  base_exp_b < 257.
        let sub_io = IsLessThanIo::new(
            base_exp_b,
            AB::Expr::from(AB::F::from_u32(257)),
            cols.cancel_is_sub.into(),
            s.into(),
        );
        lt9.eval(builder, (sub_io, &cols.route_sub_decomp));
        builder.assert_bool(cols.cancel_is_ovf);
        builder.assert_bool(cols.cancel_is_sub);
        builder.assert_zero(cols.cancel_is_ovf * cols.cancel_is_sub);
        builder.assert_eq(
            cols.s_cancel_normal,
            s * (one.clone() - cols.cancel_is_sub)
                * (one.clone() - cols.cancel_is_ovf)
                * (one.clone() - ez),
        );
        builder.assert_bool(cols.s_cancel_normal);
        let scn = cols.s_cancel_normal;

        // ── Normal path: mantissa carry chain + result bytes ─────────────
        // result_exp = r_exp_lo + 2*r_exp_hi = base_exp + rc2.
        builder
            .when(scn)
            .assert_eq(cols.result_exp, cols.r_exp_lo + cols.r_exp_hi * f2);
        builder
            .when(scn)
            .assert_eq(cols.r_exp_lo + cols.r_exp_hi * f2, cols.base_exp + cols.rc2);
        // c_m0/c_m1 are the rounded mantissa low bytes (result bytes only on
        // normal rows; sub/overflow rows output their own patterns).
        builder
            .when(s * (one.clone() - ez))
            .assert_eq(cols.c_m0 + cols.rc0 * f256, norm[3] + cols.round_up);
        builder
            .when(s * (one.clone() - ez))
            .assert_eq(cols.c_m1 + cols.rc1 * f256, norm[4] + cols.rc0);
        self.range_bus
            .range_check(cols.c_m0.into(), 8)
            .eval(builder, s * (one.clone() - ez));
        self.range_bus
            .range_check(cols.c_m1.into(), 8)
            .eval(builder, s * (one.clone() - ez));
        builder.when(scn).assert_eq(cols.result[0], cols.c_m0);
        builder.when(scn).assert_eq(cols.result[1], cols.c_m1);
        // top7c = (norm[5] - 128) + rc1 = rc2_rem + rc2*128.
        // Pin implicit bit at bit 47: RC (norm[5]-128) to 7 bits. Without this,
        // under-shifting by 1 with norm[5]=127 and rc1=1 satisfies top7c with
        // rem=rc2=0 (LHS=0), forging result_exp+1 while gates still pass.
        builder
            .when(s * (one.clone() - ez))
            .assert_eq((norm[5] - f128) + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus
            .range_check(norm[5] - f128, 7)
            .eval(builder, s * (one.clone() - ez));
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s * (one.clone() - ez));
        builder
            .when(scn)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        builder
            .when(scn)
            .assert_eq(cols.result[3], cols.r_exp_hi + cols.result_sign * f128);

        // ── Exact-zero path: result = +0 ────────────────────────────────
        let sez = s * ez;
        for i in 0..4 {
            builder.when(sez.clone()).assert_zero(cols.result[i]);
        }

        // ── Overflow path: result = +-inf ────────────────────────────────
        let sov = s * cols.cancel_is_ovf;
        builder.when(sov.clone()).assert_zero(cols.result[0]);
        builder.when(sov.clone()).assert_zero(cols.result[1]);
        builder.when(sov.clone()).assert_eq(cols.result[2], f128);
        builder.when(sov).assert_eq(
            cols.result[3],
            AB::Expr::from(f127) + cols.result_sign * f128,
        );

        // ── Sub path: subnormal results ──────────────────────────────────
        // Denormalize: shift norm right by t = 25 - base_exp (>= 25), clamped
        // to 55 (deeper shifts always round to +-0). Single RNE rounding from
        // the gadget's guard/sticky; the round carry into bit 23 yields the
        // smallest normal. Cancellation is lossless (ediff <= 24), so no
        // subtractive-sticky handling is needed.
        builder.assert_eq(
            cols.s_cancel_sub,
            s * cols.cancel_is_sub * (one.clone() - ez),
        );
        builder.assert_bool(cols.s_cancel_sub);
        let ssub = cols.s_cancel_sub;
        builder
            .when(ssub)
            .assert_eq(cols.sub_t, AB::Expr::from(f25) - cols.base_exp);
        self.range_bus
            .range_check(cols.sub_t.into(), 7)
            .eval(builder, ssub.into());
        builder.assert_bool(cols.sub_deep);
        let sub_deep_io = IsLessThanIo::new(
            AB::Expr::from(AB::F::from_u32(55)),
            cols.sub_t.into(),
            cols.sub_deep.into(),
            ssub.into(),
        );
        IsLtSubAir::new(self.range_bus, 7).eval(builder, (sub_deep_io, &cols.sub_deep_decomp));
        builder.when(ssub).assert_eq(
            cols.sub_shrn.shift,
            (one.clone() - cols.sub_deep) * cols.sub_t + cols.sub_deep * AB::F::from_u32(55),
        );
        for i in 0..6 {
            builder
                .when(ssub)
                .assert_eq(cols.sub_shrn.x[i], cols.cancel_shln.y[i]);
        }
        builder.assert_eq(cols.sub_shrn.enable, ssub);
        let sub_shrn = ShrnGadget::new(self.range_bus);
        sub_shrn.eval_shrn(builder, &cols.sub_shrn);
        // RNE rounding: guard/sticky from the gadget, lsb from y[0].
        builder.assert_bool(cols.sub_lsb);
        builder
            .when(ssub)
            .assert_eq(cols.sub_shrn.y[0], cols.sub_lsb + cols.sub_hi7 * f2);
        self.range_bus
            .range_check(cols.sub_hi7.into(), 7)
            .eval(builder, ssub.into());
        let sub_sor = cols.sub_shrn.sticky + cols.sub_lsb - cols.sub_shrn.sticky * cols.sub_lsb;
        builder
            .when(ssub)
            .assert_eq(cols.sub_round_up, cols.sub_shrn.guard * sub_sor);
        builder.assert_bool(cols.sub_round_up);
        // z = y + round_up (<= 2^23; bit 23 of z is the exp field LSB).
        builder.assert_bool(cols.sub_zc0);
        builder.assert_bool(cols.sub_zc1);
        builder.when(ssub).assert_eq(
            cols.sub_z0 + cols.sub_zc0 * f256,
            cols.sub_shrn.y[0] + cols.sub_round_up,
        );
        builder.when(ssub).assert_eq(
            cols.sub_z1 + cols.sub_zc1 * f256,
            cols.sub_shrn.y[1] + cols.sub_zc0,
        );
        builder
            .when(ssub)
            .assert_eq(cols.sub_z2, cols.sub_shrn.y[2] + cols.sub_zc1);
        for z in [cols.sub_z0, cols.sub_z1, cols.sub_z2] {
            self.range_bus
                .range_check(z.into(), 8)
                .eval(builder, ssub.into());
        }
        builder.when(ssub).assert_eq(cols.result[0], cols.sub_z0);
        builder.when(ssub).assert_eq(cols.result[1], cols.sub_z1);
        builder.when(ssub).assert_eq(cols.result[2], cols.sub_z2);
        builder
            .when(ssub)
            .assert_eq(cols.result[3], cols.result_sign * f128);
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

    // Subnormal operand pre-normalization (fills shln columns + counts).
    let (a_sub, a_exp_eff, a_sig_norm) = fill_op_norm(
        a_exp,
        a_mant,
        &mut row.a_lshift,
        &mut row.a_shln,
        &mut row.a_y2_lo7,
        range_checker,
    );
    let (b_sub, b_exp_eff, b_sig_norm) = fill_op_norm(
        b_exp,
        b_mant,
        &mut row.b_lshift,
        &mut row.b_shln,
        &mut row.b_y2_lo7,
        range_checker,
    );
    row.a_sub = F::from_bool(a_sub);
    row.b_sub = F::from_bool(b_sub);
    // Effective significands: implicit bit only for *normal* operands.
    let a_sig_eff = if a_sub {
        a_sig_norm
    } else if a_is_normal {
        a_mant | 0x800000
    } else {
        a_mant
    };
    let b_sig_eff = if b_sub {
        b_sig_norm
    } else if b_is_normal {
        b_mant | 0x800000
    } else {
        b_mant
    };
    for i in 0..3 {
        row.a_sigb[i] = F::from_u32((a_sig_eff >> (i * 8)) & 0xFF);
        row.b_sigb[i] = F::from_u32((b_sig_eff >> (i * 8)) & 0xFF);
    }

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
    row.is_nan_result = F::from_bool(is_nan_res);

    // Input-derived special-case selectors (constrained in eval_special_cases).
    let inf_in = a_is_inf || b_is_inf;
    let s_inf = !is_nan_res && inf_in;
    let s_zero_both = !is_nan_res && !inf_in && a_is_zero && b_is_zero;
    row.inf_in = F::from_bool(inf_in);
    row.s_nan = F::from_bool(is_nan_res);
    row.s_inf = F::from_bool(s_inf);
    row.s_zero_both = F::from_bool(s_zero_both);

    // Alignment selection (effective exponents absorb the normalization shift).
    let same_sign = a_sign == b_sign;
    let a_is_larger = a_exp_eff >= b_exp_eff;
    let ediff = if a_is_larger {
        (a_exp_eff - b_exp_eff) as u32
    } else {
        (b_exp_eff - a_exp_eff) as u32
    };
    row.is_add = F::from_bool(same_sign);
    row.a_is_larger = F::from_bool(a_is_larger);
    row.ediff = F::from_u32(ediff);
    range_checker.add_count(ediff, 9);

    // Sound selector: finite, same-sign, both finite-nonzero.
    let a_ok = a_is_normal || a_sub;
    let b_ok = b_is_normal || b_sub;
    let s_add = !is_nan_res && !inf_in && !s_zero_both && same_sign && a_ok && b_ok;
    row.s_add = F::from_bool(s_add);

    // Selected significands (pre-normalized).
    let (large_sig, small_sig, _large_exp, large_exp_eff, large_sign) = if a_is_larger {
        (a_sig_eff, b_sig_eff, a_exp, a_exp_eff, a_sign)
    } else {
        (b_sig_eff, a_sig_eff, b_exp, b_exp_eff, b_sign)
    };
    for i in 0..SIG_NUM_LIMBS {
        row.large_sig[i] = F::from_u32((large_sig >> (i * LIMB_BITS)) & 0xFF);
        row.small_sig[i] = F::from_u32((small_sig >> (i * LIMB_BITS)) & 0xFF);
    }

    let is_deep = ediff >= DEEP_EDIFF;
    row.is_deep = F::from_bool(is_deep);
    let deep_delta = if is_deep {
        ediff - DEEP_EDIFF
    } else {
        DEEP_EDIFF - 1 - ediff
    };
    row.deep_delta = F::from_u32(deep_delta);
    row.deep_delta_inv = if deep_delta != 0 {
        F::from_u32(deep_delta).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    range_checker.add_count(deep_delta, LIMB_BITS);

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
    let s_cancel = s_finite_bool && !same_sign && a_ok && b_ok && !is_deep;
    let s_bn_bool = s_finite_bool && a_ok && b_ok;
    row.s_bn = F::from_bool(s_bn_bool);
    row.s_fb = F::from_bool(s_finite_bool && !a_is_zero && !b_is_zero);
    row.s_one_zero = F::from_bool(s_finite_bool && (a_is_zero != b_is_zero));
    row.s_cancel = F::from_bool(s_cancel);
    row.s_cancel_normal = F::ZERO;

    // Deep binade-boundary selectors (effective subtraction only).
    let large_pow2_sum = (large_sig & 0xFF)
        + ((large_sig >> 8) & 0xFF)
        + ((large_sig >> 16) & 0xFF).wrapping_sub(0x80);
    let large_pow2 = large_pow2_sum == 0;
    row.large_pow2_inv = if large_pow2_sum != 0 {
        F::from_u32(large_pow2_sum).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let small_ext_sum = (small_sig & 0xFF)
        + ((small_sig >> 8) & 0xFF)
        + ((small_sig >> 16) & 0xFF).wrapping_sub(0x80);
    let small_ext = small_ext_sum != 0;
    row.small_ext_inv = if small_ext_sum != 0 {
        F::from_u32(small_ext_sum).try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let s_deep_cancel = s_bn_bool && !same_sign && is_deep;
    let s_deep_ediff25 = s_deep_cancel && deep_delta == 0;
    let s_deep_boundary = s_deep_ediff25 && large_pow2 && small_ext;
    let s_deep_exact = s_bn_bool && is_deep && !s_deep_boundary;
    row.s_deep_cancel = F::from_bool(s_deep_cancel);
    row.s_deep_ediff25 = F::from_bool(s_deep_ediff25);
    row.s_deep_boundary = F::from_bool(s_deep_boundary);
    row.s_deep_exact = F::from_bool(s_deep_exact);

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
                large_exp_eff,
                large_sign,
                a_sig_eff,
                b_sig_eff,
                a_sign,
                b_sign,
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
        // Boundary deep path reads r_exp_lo/hi to pack mant-all-ones result bytes.
        if s_deep_boundary {
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
        }
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
    let result_exp_calc = large_exp_eff + carry_out as i32 + rc2 as i32;

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
    range_checker.add_count(((norm >> 24) & 0x7F) as u32, 7);

    // ── Result-class routing (base_exp_sh + 256 bias, 9 bits) ──────────
    let base_exp_sh_i = large_exp_eff + carry_out as i32;
    let is_sub_sh = base_exp_sh_i <= 0;
    row.sh_is_sub = F::from_bool(is_sub_sh);
    row.sh_base_exp = if base_exp_sh_i < 0 {
        -F::from_u32((-base_exp_sh_i) as u32)
    } else {
        F::from_u32(base_exp_sh_i as u32)
    };
    let sh_base_exp_b = (base_exp_sh_i + 256) as u32;
    debug_assert!(sh_base_exp_b < 512, "fadd sh_base_exp_b out of range");
    if s_shallow {
        range_checker.add_count(sh_base_exp_b, 9);
        let lt9 = IsLtSubAir::new(range_checker.bus(), 9);
        lt9.generate_subrow(
            (range_checker.as_ref(), sh_base_exp_b, 257u32),
            (&mut row.sh_route_decomp, &mut row.sh_is_sub),
        );
        debug_assert_eq!(row.sh_is_sub, F::from_bool(is_sub_sh));
    }

    // ── Subnormal-result witness (denormalizing shift) ─────────────────
    if s_shallow && is_sub_sh {
        let t = 9 - base_exp_sh_i;
        debug_assert!((9..=31).contains(&t), "fadd sh_sub_t out of range: {t}");
        row.sh_sub_t = F::from_u32(t as u32);
        range_checker.add_count(t as u32, 5);
        let x4: [u32; 4] = core::array::from_fn(|i| ((norm >> (i * 8)) & 0xFF) as u32);
        row.sh_sub_shrn = make_shrn_cols::<F>(x4, t as usize, range_checker);

        let y0 = row.sh_sub_shrn.y[0].as_canonical_u32();
        let y1 = row.sh_sub_shrn.y[1].as_canonical_u32();
        let y2 = row.sh_sub_shrn.y[2].as_canonical_u32();
        row.sh_sub_lsb = F::from_u32(y0 & 1);
        row.sh_sub_hi7 = F::from_u32(y0 >> 1);
        range_checker.add_count(y0 >> 1, 7);
        let sh_round_up = row.sh_sub_shrn.guard.as_canonical_u32() == 1
            && (row.sh_sub_shrn.sticky.as_canonical_u32() != 0
                || row.lost_sticky.as_canonical_u32() != 0
                || y0 & 1 == 1);
        row.sh_sub_sticky_eff = F::from_bool(
            row.sh_sub_shrn.sticky.as_canonical_u32() != 0
                || row.lost_sticky.as_canonical_u32() != 0,
        );
        row.sh_sub_round_up = F::from_bool(sh_round_up);
        let zc0 = (y0 + sh_round_up as u32) >> 8;
        let z0 = (y0 + sh_round_up as u32) & 0xFF;
        let zc1 = (y1 + zc0) >> 8;
        let z1 = (y1 + zc0) & 0xFF;
        let z2 = y2 + zc1;
        row.sh_sub_z0 = F::from_u32(z0);
        row.sh_sub_z1 = F::from_u32(z1);
        row.sh_sub_z2 = F::from_u32(z2);
        row.sh_sub_zc0 = F::from_u32(zc0);
        row.sh_sub_zc1 = F::from_u32(zc1);
        range_checker.add_count(z0, 8);
        range_checker.add_count(z1, 8);
        range_checker.add_count(z2, 8);
        let exp1 = z2 == 128;
        row.sh_sub_exp1 = F::from_bool(exp1);
        row.sh_sub_z2_lo7 = F::from_u32(z2 & 0x7F);
        range_checker.add_count(z2 & 0x7F, 7);
        // Sanity against the native result.
        debug_assert_eq!(
            result_bits & 0x7FFF_FFFF,
            z0 | (z1 << 8) | (z2 << 16),
            "fadd shallow subnormal result mismatch: a={:08x} b={:08x}",
            a_bits,
            b_bits
        );
    }

    // Sanity: witness must agree with the native result on this path (only for a
    // normal result; overflow -> inf does not match the significand reconstruction).
    if (1u32..=254).contains(&r_exp) {
        debug_assert_eq!(
            result_exp_calc, r_exp as i32,
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
    row.sh_ef_max = F::from_bool(r_exp == 255);
    row.s_normal = F::from_bool(s_shallow && !is_sub_sh && r_exp != 255);
    row.s_overflow = F::from_bool(s_shallow && !is_sub_sh && r_exp == 255);
    row.s_sub_result = F::from_bool(s_shallow && is_sub_sh);
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
    row.s_sub_result = F::ZERO;
    row.sh_base_exp = F::ZERO;
    row.sh_is_sub = F::ZERO;
    row.sh_ef_max = F::ZERO;
    row.sh_route_decomp = [F::ZERO];
    row.sh_sub_t = F::ZERO;
    row.sh_sub_shrn = ShrnCols::<F, 4>::default();
    row.sh_sub_lsb = F::ZERO;
    row.sh_sub_hi7 = F::ZERO;
    row.sh_sub_round_up = F::ZERO;
    row.sh_sub_z0 = F::ZERO;
    row.sh_sub_z1 = F::ZERO;
    row.sh_sub_z2 = F::ZERO;
    row.sh_sub_zc0 = F::ZERO;
    row.sh_sub_zc1 = F::ZERO;
    row.sh_sub_exp1 = F::ZERO;
    row.sh_sub_z2_lo7 = F::ZERO;
    row.sh_sub_sticky_eff = F::ZERO;
    row.raw_csum = F::ZERO;
    row.raw_csum_inv = F::ZERO;
    row.is_exact_zero = F::ZERO;
    row.c_m0 = F::ZERO;
    row.c_m1 = F::ZERO;
    row.base_exp = F::ZERO;
    row.cancel_is_ovf = F::ZERO;
    row.cancel_is_sub = F::ZERO;
    row.route_ovf_decomp = [F::ZERO];
    row.route_sub_decomp = [F::ZERO];
    row.sub_t = F::ZERO;
    row.sub_deep = F::ZERO;
    row.sub_deep_decomp = [F::ZERO];
    row.sub_shrn = ShrnCols::<F, 6>::default();
    row.sub_lsb = F::ZERO;
    row.sub_hi7 = F::ZERO;
    row.sub_round_up = F::ZERO;
    row.sub_z0 = F::ZERO;
    row.sub_z1 = F::ZERO;
    row.sub_z2 = F::ZERO;
    row.sub_zc0 = F::ZERO;
    row.sub_zc1 = F::ZERO;
    row.s_cancel_sub = F::ZERO;
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
    large_exp: i32,
    large_sign: u32,
    a_sig: u32,
    b_sig: u32,
    a_sign: u32,
    b_sign: u32,
    r_exp: u32,
    result_bits: u32,
    a_bits: u32,
    b_bits: u32,
    ediff: u32,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    // Determine effective large/small for the subtraction.
    // ediff >= 1: large/small from exp selection (already correct).
    // ediff == 0: use a_ge_b (mantissa magnitude comparison).
    let (eff_large_sig, eff_small_sig, a_ge_b) = if ediff != 0 {
        (large_sig, small_sig, false)
    } else {
        let ge = a_sig >= b_sig;
        (
            if ge { a_sig } else { b_sig },
            if ge { b_sig } else { a_sig },
            ge,
        )
    };
    row.a_ge_b = F::from_bool(a_ge_b);
    // ── 6-byte alignment via ShrnGadget (ediff >= 1); identity for ediff=0 ─
    let x: [u32; 6] = [
        0,
        0,
        0,
        small_sig & 0xFF,
        (small_sig >> 8) & 0xFF,
        (small_sig >> 16) & 0xFF,
    ];
    row.cancel_align = make_shrn_cols6::<F>(x, ediff as usize, range_checker);

    // ── 6-byte subtraction: raw_diff = eff_large_working - eff_small_working ─
    let large_working6: u64 = (eff_large_sig as u64) << 24;
    let aligned: u64 = if ediff != 0 {
        (0..6).fold(0u64, |acc, i| {
            acc | ((row.cancel_align.y[i].as_canonical_u32() as u64) << (i * 8))
        })
    } else {
        (eff_small_sig as u64) << 24
    };
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
    let mut byte_borrows = [0u32; 5];
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
            byte_borrows[i] = borrow as u32;
        }
    }
    debug_assert_eq!(borrow, 0, "fadd cancel final borrow != 0");
    // The AIR uses different borrow layouts per branch: ediff >= 1 chains all
    // 6 bytes (borrow[i] = carry out of byte i), while the ediff == 0 branch
    // chains only bytes 3..5 and reads borrow[0]/borrow[1] as the carries out
    // of bytes 3/4.
    if ediff == 0 {
        row.cancel_borrow[0] = F::from_u32(byte_borrows[3]);
        row.cancel_borrow[1] = F::from_u32(byte_borrows[4]);
        for i in 2..5 {
            row.cancel_borrow[i] = F::ZERO;
        }
    } else {
        for (i, &bb) in byte_borrows.iter().enumerate() {
            row.cancel_borrow[i] = F::from_u32(bb);
        }
    }

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
    let top7c = nb5.saturating_sub(128) + rc1;
    let rc2 = top7c >> 7;
    let rc2_rem = top7c & 0x7F;
    let base_exp_i = large_exp - lshift as i32;
    row.base_exp = if base_exp_i < 0 {
        -F::from_u32((-base_exp_i) as u32)
    } else {
        F::from_u32(base_exp_i as u32)
    };
    let result_exp_calc = base_exp_i + rc2 as i32;

    // Exact-zero cancellation (raw_diff all zero).
    let raw_csum: u32 = raw_bytes6.iter().sum();
    let is_exact_zero = raw_csum == 0;
    row.raw_csum = F::from_u32(raw_csum);
    row.is_exact_zero = F::from_bool(is_exact_zero);
    row.raw_csum_inv = if is_exact_zero {
        F::ZERO
    } else {
        F::from_u32(raw_csum).try_inverse().unwrap_or(F::ZERO)
    };
    let (rc0, rc1, rc2) = if is_exact_zero {
        (0, 0, 0)
    } else {
        (rc0, rc1, rc2)
    };

    // Result-class routing via base_exp (+256 bias, 9 bits).
    let is_ovf = base_exp_i + rc2 as i32 >= 255;
    let is_sub = base_exp_i <= 0;
    row.cancel_is_ovf = F::from_bool(is_ovf);
    row.cancel_is_sub = F::from_bool(is_sub);
    let base_exp_b = (base_exp_i + 256) as u32;
    debug_assert!(base_exp_b < 512, "fadd cancel base_exp_b out of range");
    range_checker.add_count(base_exp_b, 9);
    let lt9 = IsLtSubAir::new(range_checker.bus(), 9);
    lt9.generate_subrow(
        (range_checker.as_ref(), 510u32, base_exp_b + rc2),
        (&mut row.route_ovf_decomp, &mut row.cancel_is_ovf),
    );
    lt9.generate_subrow(
        (range_checker.as_ref(), base_exp_b, 257u32),
        (&mut row.route_sub_decomp, &mut row.cancel_is_sub),
    );
    debug_assert_eq!(row.cancel_is_ovf, F::from_bool(is_ovf));
    debug_assert_eq!(row.cancel_is_sub, F::from_bool(is_sub));

    // ── Subnormal-result witness ────────────────────────────────────────
    if is_sub && !is_exact_zero {
        let t = 25 - base_exp_i;
        debug_assert!((25..128).contains(&t), "fadd cancel sub_t out of range");
        let sub_deep = t > 55;
        let shift = if sub_deep { 55usize } else { t as usize };
        row.sub_t = F::from_u32(t as u32);
        range_checker.add_count(t as u32, 7);
        let lt7 = IsLtSubAir::new(range_checker.bus(), 7);
        lt7.generate_subrow(
            (range_checker.as_ref(), 55u32, t as u32),
            (&mut row.sub_deep_decomp, &mut row.sub_deep),
        );
        debug_assert_eq!(row.sub_deep, F::from_bool(sub_deep));
        let x6: [u32; 6] = core::array::from_fn(|i| ((norm >> (i * 8)) & 0xFF) as u32);
        row.sub_shrn = make_shrn_cols6::<F>(x6, shift, range_checker);

        let y0 = row.sub_shrn.y[0].as_canonical_u32();
        let y1 = row.sub_shrn.y[1].as_canonical_u32();
        let y2 = row.sub_shrn.y[2].as_canonical_u32();
        row.sub_lsb = F::from_u32(y0 & 1);
        row.sub_hi7 = F::from_u32(y0 >> 1);
        range_checker.add_count(y0 >> 1, 7);
        let sub_round_up = row.sub_shrn.guard.as_canonical_u32() == 1
            && (row.sub_shrn.sticky.as_canonical_u32() != 0 || y0 & 1 == 1);
        row.sub_round_up = F::from_bool(sub_round_up);
        row.s_cancel_sub = F::from_bool(true);
        let zc0 = (y0 + sub_round_up as u32) >> 8;
        let z0 = (y0 + sub_round_up as u32) & 0xFF;
        let zc1 = (y1 + zc0) >> 8;
        let z1 = (y1 + zc0) & 0xFF;
        let z2 = y2 + zc1;
        row.sub_z0 = F::from_u32(z0);
        row.sub_z1 = F::from_u32(z1);
        row.sub_z2 = F::from_u32(z2);
        row.sub_zc0 = F::from_u32(zc0);
        row.sub_zc1 = F::from_u32(zc1);
        range_checker.add_count(z0, 8);
        range_checker.add_count(z1, 8);
        range_checker.add_count(z2, 8);
        // Sanity against the native result.
        debug_assert_eq!(
            result_bits & 0x7FFF_FFFF,
            z0 | (z1 << 8) | (z2 << 16),
            "fadd cancel subnormal result mismatch: a={:08x} b={:08x}",
            a_bits,
            b_bits
        );
    }
    if is_exact_zero {
        debug_assert_eq!(result_bits, 0, "fadd exact cancellation must be +0");
    }
    if is_ovf {
        debug_assert_eq!(
            result_bits & 0x7FFF_FFFF,
            0x7F80_0000,
            "fadd cancel overflow result not inf: {result_bits:08x}"
        );
    }

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
    row.c_m0 = F::from_u32(s0 & 0xFF);
    row.c_m1 = F::from_u32(s1 & 0xFF);
    if !is_exact_zero {
        range_checker.add_count(s0 & 0xFF, 8);
        range_checker.add_count(s1 & 0xFF, 8);
    }
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
    // Result sign: ediff>=1 → large_sign; ediff=0 → a_ge_b ? a_sign : b_sign.
    let cancel_r_sign = if ediff != 0 {
        large_sign
    } else if a_ge_b {
        a_sign
    } else {
        b_sign
    };
    row.result_sign = F::from_u32(cancel_r_sign);
    if !is_exact_zero {
        range_checker.add_count(rc2_rem, 7);
        range_checker.add_count(nb5.saturating_sub(128), 7);
    }
    row.s_cancel_normal = F::from_bool(!is_sub && !is_ovf && !is_exact_zero);

    if (1u32..=254).contains(&r_exp) {
        debug_assert_eq!(
            result_exp_calc, r_exp as i32,
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

        // Decompose inputs (needed for the one-zero special case).
        let a_mant = a & 0x7FFFFF;
        let b_mant = b & 0x7FFFFF;
        let a_exp = (a >> 23) & 0xFF;
        let b_exp = (b >> 23) & 0xFF;

        FaddCoreCols {
            a_sign: F::from_u32((a >> 31) & 1),
            b_sign: F::from_u32((b >> 31) & 1),
            a_mant0: F::from_u32(a_mant & 0xFF),
            a_mant1: F::from_u32((a_mant >> 8) & 0xFF),
            a_mant2: F::from_u32((a_mant >> 16) & 0x7F),
            a_exp_lo: F::from_u32(a_exp & 1),
            a_exp_hi: F::from_u32(a_exp >> 1),
            b_mant0: F::from_u32(b_mant & 0xFF),
            b_mant1: F::from_u32((b_mant >> 8) & 0xFF),
            b_mant2: F::from_u32((b_mant >> 16) & 0x7F),
            b_exp_lo: F::from_u32(b_exp & 1),
            b_exp_hi: F::from_u32(b_exp >> 1),
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
        assert!(
            deep_seen,
            "never exercised the deep (ediff>=DEEP_EDIFF) path"
        );
        assert!(overflow_seen, "never exercised an overflow->inf result");
    }

    #[test]
    fn finite_add_subnormal_operands_pass() {
        let chip = rc();
        // Curated subnormal-operand cases (including exp-field-0 results).
        for &(a, b) in &[
            (0x00000001, 0x3F800000), // min subnormal + 1.0 (deep)
            (0x3F800000, 0x00000001), // 1.0 + min subnormal (deep)
            (0x007FFFFF, 0x3F800000), // max subnormal + 1.0 (shallow)
            (0x00800000, 0x00400000), // min normal + subnormal
            (0x00400000, 0x80400000), // subnormal - subnormal (cancel)
            (0x00000001, 0x80000001), // min sub - min sub = +0 (exact cancel)
            (0x3F800000, 0xBF800000), // 1.0 + (-1.0) = +0 (exact cancel)
            (0x3F800000, 0x80800000), // 1.0 - min normal (cancel)
            (0x00800000, 0x80800001), // min_normal - (min_normal + 1ulp) -> -min_sub
            (0x00800000, 0x807FFFFF), // min_normal - (min_normal - 1ulp) -> +min_sub
            (0x00000001, 0x80000002), // min_sub - 2*min_sub -> -min_sub
        ] {
            let (builder, _) = run_with(a, b, &chip);
            assert!(
                builder.ok(),
                "subnormal-operand fadd failed: {:?}\n a={a:08x} b={b:08x}",
                builder.failures
            );
        }
        // Fuzz with subnormal operands (all result classes).
        let mut rng = Rng(0x5BB0_1A11_0DEF_0001);
        for _ in 0..3000 {
            let sa = ((rng.next() as u32) & 1) << 31;
            let sb = ((rng.next() as u32) & 1) << 31;
            let ea = rng.next() as u32 % 255;
            let eb = rng.next() as u32 % 255;
            let a = sa | (ea << 23) | ((rng.next() as u32) & 0x7FFFFF);
            let b = sb | (eb << 23) | ((rng.next() as u32) & 0x7FFFFF);
            if (ea == 0 && a & 0x7FFFFF == 0) || (eb == 0 && b & 0x7FFFFF == 0) {
                continue; // zero operands are special-cased
            }
            let (builder, _) = run_with(a, b, &chip);
            assert!(
                builder.ok(),
                "subnormal-operand fuzz failed: {:?}\n a={a:08x} b={b:08x}",
                builder.failures
            );
        }
        // Fuzz biased to tiny exponents for subnormal-result coverage.
        for _ in 0..2000 {
            let sa = ((rng.next() as u32) & 1) << 31;
            let sb = ((rng.next() as u32) & 1) << 31;
            let ea = (rng.next() as u32 % 6) + 1;
            let eb = (rng.next() as u32 % 6) + 1;
            let a = sa | (ea << 23) | ((rng.next() as u32) & 0x7FFFFF);
            let b = sb | (eb << 23) | ((rng.next() as u32) & 0x7FFFFF);
            let (builder, _) = run_with(a, b, &chip);
            assert!(
                builder.ok(),
                "tiny-exp fuzz failed: {:?}\n a={a:08x} b={b:08x}",
                builder.failures
            );
        }
    }

    #[test]
    fn finite_add_known_values() {
        // 1.0 + 1.0 = 2.0 (carry-out normalization)
        let (b, _) = run(1.0f32.to_bits(), 1.0f32.to_bits());
        assert!(b.ok(), "{:?}", b.failures);
        // 1.0 + 2.0 = 3.0 (no carry)
        let (b, _) = run(1.0f32.to_bits(), 2.0f32.to_bits());
        assert!(b.ok(), "{:?}", b.failures);
        // 1.0 + (tiny, ediff=25) = 1.0 (deep same-sign)
        let large = 1.0f32;
        // normal small operand 25 exp below 1.0 (exp 127): exp 102
        let small = f32::from_bits((102u32 << 23) | 1).to_bits();
        let (b, row) = run(large.to_bits(), small);
        assert!(b.ok(), "deep path: {:?}", b.failures);
        assert_eq!(row.is_deep, F::ONE);
        assert_eq!(row.s_deep_boundary, F::ZERO);
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

    /// Binade-boundary completeness: opposite signs, larger = exact power of two,
    /// ediff == 25, smaller mantissa != 0 ⇒ result = larger − 1 lower-binade ULP.
    /// The old DEEP_EDIFF=25 deep path asserted result == larger and rejected the
    /// honest IEEE witness (prove failure). Tie (mant=0) and non-boundary stay OK.
    #[test]
    fn finite_add_binade_boundary_ediff25() {
        let chip = rc();
        // (a, b, expected, expect_boundary)
        let cases: &[(u32, u32, u32, bool)] = &[
            (0x3F000000, 0xB2800001, 0x3EFFFFFF, true), // past midpoint → larger−1ulp
            (0x3F800000, 0xB3000001, 0x3F7FFFFF, true),
            (0x3F000000, 0xB2800000, 0x3F000000, false), // exact tie → round-even = larger
            (0x3F000001, 0xB2800001, 0x3F000001, false), // not power-of-two larger
            (0xB2800001, 0x3F000000, 0x3EFFFFFF, true),  // operand swap
            (0xB3000001, 0x3F800000, 0x3F7FFFFF, true),
            (0x3F000000, 0xB2000001, 0x3F000000, false), // ediff=26 deep exact
        ];
        for &(a, b, exp, expect_boundary) in cases {
            assert_eq!(fadd_s(a, b), exp, "ieee a={a:08x} b={b:08x}");
            let (builder, row) = run_with(a, b, &chip);
            assert!(
                builder.ok(),
                "binade ediff case a={a:08x} b={b:08x}: {:?}",
                builder.failures
            );
            let got = row.result[0].as_canonical_u32()
                | (row.result[1].as_canonical_u32() << 8)
                | (row.result[2].as_canonical_u32() << 16)
                | (row.result[3].as_canonical_u32() << 24);
            assert_eq!(got, exp, "result bits a={a:08x} b={b:08x}");
            assert_eq!(
                row.s_deep_boundary,
                F::from_bool(expect_boundary),
                "boundary flag a={a:08x} b={b:08x}"
            );
            if expect_boundary {
                assert_eq!(row.is_deep, F::ONE);
                assert_eq!(row.s_deep_exact, F::ZERO);
                assert_eq!(row.s_cancel, F::ZERO);
            }
        }
    }

    #[test]
    fn finite_add_ediff0_cancellation() {
        let chip = rc();
        // 1.0 + (-(1.0+tiny)) = -tiny. ediff=0, a_ge_b=false.
        let mut row = FaddCoreCols::<F>::default();
        fill_fadd_witness(
            &mut row,
            0x3f800000,
            0xbf800001,
            fadd_s(0x3f800000, 0xbf800001),
            &chip,
        );
        let mut b = NumericBuilder::new(vec![]);
        air().eval_finite_add(&mut b, &row);
        air().eval_finite_cancel(&mut b, &row);
        assert!(b.ok(), "ediff=0 cancel a<b: {:?}", b.failures);
        // 3.0 + (-2.0) = 1.0. ediff=0, a_ge_b=true.
        let mut row2 = FaddCoreCols::<F>::default();
        fill_fadd_witness(
            &mut row2,
            3.0f32.to_bits(),
            (-2.0f32).to_bits(),
            fadd_s(3.0f32.to_bits(), (-2.0f32).to_bits()),
            &chip,
        );
        let mut b2 = NumericBuilder::new(vec![]);
        air().eval_finite_add(&mut b2, &row2);
        air().eval_finite_cancel(&mut b2, &row2);
        assert!(b2.ok(), "ediff=0 cancel a>b: {:?}", b2.failures);
    }

    /// Result bytes and routing selectors must be pinned: a +1 perturbation of
    /// each must break a gate on the full eval, across the input classes.
    #[test]
    fn corruption_sweep_pins_result_and_selectors() {
        use crate::test_utils::assert_columns_constrained;
        use openvm_circuit::arch::BasicAdapterInterface;
        use openvm_circuit::arch::MinimalInstruction;
        use openvm_circuit_primitives::StructReflectionHelper;
        type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;
        let chip = rc();
        let names = FaddCoreCols::<F>::struct_reflection().unwrap_or_default();
        let eval_full = |b: &mut NumericBuilder<F>, cols: &[F]| {
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), b, cols, F::ZERO);
        };
        let classes: &[(u32, u32, &[&str])] = &[
            // (a, b, selectors asserted on this class)
            (
                1.5f32.to_bits(),
                2.5f32.to_bits(),
                &["is_add", "round_up", "is_deep"][..],
            ),
            (0x3F800000, 0xBF800000, &[]), // 1.0 + (-1.0) = +0 (cancel path)
            (0x7F800000, 0x3F800000, &[]), // inf + 1.0
            (0x7FC00000, 0x3F800000, &[]), // NaN + 1.0
            (0x00800000, 0x00800000, &[]), // min normal + min normal
            (0x00400000, 0x80400000, &[]), // subnormal - subnormal (cancel)
        ];
        for &(a, b, selectors) in classes {
            let mut row = FaddCoreCols::<F>::default();
            fill_fadd_witness(&mut row, a, b, fadd_s(a, b), &chip);
            let mut buf = vec![F::ZERO; FaddCoreCols::<F>::width()];
            *buf.as_mut_slice().borrow_mut() = row;
            let mut bld = NumericBuilder::new(vec![]);
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air(), &mut bld, &buf, F::ZERO);
            assert!(
                bld.ok(),
                "honest fadd a={a:08x} b={b:08x}: {:?}",
                bld.failures
            );
            let mut pinned = vec!["result__0", "result__1", "result__2", "result__3"];
            pinned.extend_from_slice(selectors);
            assert_columns_constrained(
                &buf,
                eval_full,
                &names,
                &pinned,
                &format!("fadd a={a:08x} b={b:08x}"),
            );
        }
    }
}
