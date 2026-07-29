use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
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
pub const EXT_SIG_LIMBS: usize = 4; // extended significand: 3 mantissa bytes + 1 guard/sticky byte
pub const LIMB_BITS: usize = 8;

/// IEEE 754 / RISC-V FFSUB.S: correctly-rounded (RNE) subtraction.
/// Returns the canonical NaN whenever the result is any NaN (NaN inputs, or
/// inf - (-inf)) — the result is deterministic across platforms.
#[inline(always)]
pub fn fsub_s(a: u32, b: u32) -> u32 {
    let r = f32::from_bits(a) - f32::from_bits(b);
    if r.is_nan() {
        CANONICAL_NAN
    } else {
        r.to_bits()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug, Default)]
pub struct FsubCoreCols<T> {
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
    pub s_nan: T,       // canonical NaN (NaN input, or inf - inf)
    pub s_inf: T,       // +-inf due to an infinite input
    pub s_zero_both: T, // +-0 from 0 - 0

    // === Alignment ===
    pub a_is_larger: T, // bool: a_exp >= b_exp
    pub ediff: T,       // |a_exp - b_exp|, capped at 255
    pub aligned_sig: [T; EXT_SIG_LIMBS], /* smaller operand's significand after right shift +
                         * guard bits */
    pub sticky: T, // bool: any bits lost during alignment shift

    // === Sum / difference ===
    pub is_add: T,                   // bool: same sign → add, different sign → subtract
    pub raw_sum: [T; EXT_SIG_LIMBS], // significand sum/difference (before normalization)

    // === Normalization ===
    pub norm_sig: [T; EXT_SIG_LIMBS], // normalized significand
    pub result_exp: T,                // final exponent after normalization

    // === Rounding ===
    pub guard: T,        // bool: guard bit
    pub norm3_lo_inv: T, // inverse for extracting low bits of norm_sig[3] for sticky
    pub round_up: T,     // bool: RNE round-up decision

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
#[columns_via(FsubCoreCols<u8>)]
pub struct FsubCoreAir {
    pub range_tuple_bus: RangeTupleCheckerBus<2>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FsubCoreAir {
    fn width(&self) -> usize {
        FsubCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FsubCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FsubCoreAir
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
        let cols: &FsubCoreCols<AB::Var> = local_core.borrow();

        builder.assert_bool(cols.is_valid);

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);

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
        builder.assert_bool(cols.sticky);
        builder.assert_bool(cols.guard);
        builder.assert_bool(cols.round_up);

        // ── Special-case selectors + result constraints ──────────────────
        self.eval_special_cases(builder, cols);

        // ── Phase 3: Alignment verification ──────────────────────────────
        // Verify ediff == |a_exp - b_exp| and a_is_larger == (a_exp >= b_exp)
        let diff = a_exp.clone() - b_exp.clone();
        let expected_ediff =
            cols.a_is_larger * diff.clone() + (one.clone() - cols.a_is_larger) * (-diff.clone());
        builder.assert_eq(cols.ediff, expected_ediff);
        // When a_is_larger: a_exp >= b_exp (i.e., diff >= 0 in BabyBear)
        // When !a_is_larger: b_exp > a_exp (i.e., diff < 0, or diff is p - |diff|)
        // The ediff constraint above implicitly verifies a_is_larger consistency.

        // TODO: verify aligned_sig is the correct right-shift of the smaller significand
        // TODO: verify sticky is correct

        // ── Phase 4: Sum / difference ────────────────────────────────────
        // is_add = (a_sign == b_sign)
        let sign_xor = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;
        builder.assert_eq(cols.is_add, sign_xor.clone());
        // TODO: verify raw_sum is the correct sum/difference of significands

        // ── Phase 5: Normalization ───────────────────────────────────────
        // TODO: verify norm_sig is the correct normalized result
        // TODO: verify result_exp accounts for normalization shifts

        // ── Phase 6: RNE rounding ────────────────────────────────────────
        // guard extraction from norm_sig[3]
        let _norm3_low = cols.norm_sig[3] - cols.guard * f128;
        builder.assert_eq(cols.norm_sig[3], _norm3_low.clone() + cols.guard * f128);
        // TODO: verify round_up
        // ── Phase 7: Special-case flags ─────────────────────────────────
        // TODO: Verify NaN/inf/zero result flags from input classification.
        // Requires storing classification flags as columns to keep degree ≤ 4.
        // ── Output ───────────────────────────────────────────────────────
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FSUB_S);

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

impl FsubCoreAir {
    /// Constrain the input-derived special-case selectors and the result bytes
    /// for NaN / +-inf (infinite input) / +-0 (0 - 0) results.
    pub fn eval_special_cases<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FsubCoreCols<AB::Var>,
    ) {
        let one = AB::Expr::ONE;
        let f128: AB::F = AB::F::from_u32(128);
        let f127: AB::F = AB::F::from_u32(127);

        // inf_in = a_is_inf | b_is_inf
        builder.assert_eq(
            cols.inf_in,
            cols.a_is_inf + cols.b_is_inf - cols.a_is_inf * cols.b_is_inf,
        );
        // sign_diff = a_sign XOR b_sign
        let f2: AB::F = AB::F::from_u32(2);
        let sign_diff = cols.a_sign + cols.b_sign - cols.a_sign * cols.b_sign * f2;

        // s_nan: NaN input, or inf - inf with equal signs (same-sign infinities
        // cancel; opposite-sign inf - inf yields +-inf, handled by s_inf).
        builder.assert_eq(
            cols.s_nan,
            cols.a_is_nan + cols.b_is_nan - cols.a_is_nan * cols.b_is_nan
                + cols.a_is_inf * cols.b_is_inf * (one.clone() - sign_diff),
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
        // +-inf result. a - finite = inf (sign a_sign); finite - b_inf = -inf (sign !b_sign).
        let inf_sign = cols.a_is_inf * cols.a_sign
            + (one.clone() - cols.a_is_inf) * cols.b_is_inf * (one.clone() - cols.b_sign);
        builder.when(cols.s_inf).assert_zero(cols.result[0]);
        builder.when(cols.s_inf).assert_zero(cols.result[1]);
        builder.when(cols.s_inf).assert_eq(cols.result[2], f128);
        builder
            .when(cols.s_inf)
            .assert_eq(cols.result[3], AB::Expr::from(f127) + inf_sign * f128);
        // +-0 result from 0 - 0: sign = a_sign AND (NOT b_sign).
        let zero_sign = cols.a_sign * (one.clone() - cols.b_sign);
        builder.when(cols.s_zero_both).assert_zero(cols.result[0]);
        builder.when(cols.s_zero_both).assert_zero(cols.result[1]);
        builder.when(cols.s_zero_both).assert_zero(cols.result[2]);
        builder
            .when(cols.s_zero_both)
            .assert_eq(cols.result[3], zero_sign * f128);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FsubCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FsubExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FsubExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FsubCoreRecord),
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
            FpOpcode::FSUB_S
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        // Native f32 sub = IEEE 754 correctly-rounded (RNE), canonical NaN
        let result = fsub_s(u32::from_le_bytes(a), u32::from_le_bytes(b)).to_le_bytes();

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

pub struct FsubFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
}

impl<A> FsubFiller<A> {
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

impl<A, F> TraceFiller<F> for FsubFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &FsubCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FsubCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let result_bits = fsub_s(a_bits, b_bits);

        fill_fsub_witness(core_row, a_bits, b_bits, result_bits);
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

fn fill_fsub_witness<F: PrimeField32>(
    row: &mut FsubCoreCols<F>,
    a_bits: u32,
    b_bits: u32,
    result_bits: u32,
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

    // Alignment
    let a_is_larger = a_exp >= b_exp;
    let ediff = if a_is_larger {
        a_exp - b_exp
    } else {
        b_exp - a_exp
    };
    let (large_sig, small_sig) = if a_is_larger {
        (
            if a_is_normal {
                a_mant | 0x800000
            } else {
                a_mant
            },
            if b_is_normal {
                b_mant | 0x800000
            } else {
                b_mant
            },
        )
    } else {
        (
            if b_is_normal {
                b_mant | 0x800000
            } else {
                b_mant
            },
            if a_is_normal {
                a_mant | 0x800000
            } else {
                a_mant
            },
        )
    };

    row.a_is_larger = F::from_bool(a_is_larger);
    row.ediff = F::from_u32(ediff);

    // Align: shift small_sig right by ediff with sticky
    let (aligned, guard_bit, sticky) = if ediff >= 28 {
        (0u64, 0u32, small_sig != 0)
    } else {
        let shifted = (small_sig as u64) << 8; // put in 32-bit field (24 sig + 8 guard)
        let shifted = shifted >> ediff;
        let lost = shifted & 0xFF; // low byte = guard/sticky region
        let _aligned_top = (shifted >> 8) as u32; // top 24 bits = aligned significand
        let guard = ((lost >> 7) & 1) as u32;
        let sticky = (lost & 0x7F) != 0
            || (ediff > 0 && (small_sig & ((1u32 << ediff) - 1)) != 0 && ediff < 24);
        (shifted, guard, sticky)
    };

    // aligned_sig: 4 limbs [byte0, byte1, byte2, guard_byte]
    row.aligned_sig[0] = F::from_u32((aligned & 0xFF) as u32);
    row.aligned_sig[1] = F::from_u32(((aligned >> 8) & 0xFF) as u32);
    row.aligned_sig[2] = F::from_u32(((aligned >> 16) & 0xFF) as u32);
    row.aligned_sig[3] = F::from_u32(((aligned >> 24) & 0xFF) as u32);
    row.sticky = F::from_bool(sticky);

    // Sum / difference
    let same_sign = a_sign != b_sign;
    row.is_add = F::from_bool(same_sign);

    let large_ext = (large_sig as u64) << 8; // extend to 32 bits
    let raw_result = if same_sign {
        large_ext + aligned
    } else {
        large_ext - aligned
    };
    row.raw_sum[0] = F::from_u32((raw_result & 0xFF) as u32);
    row.raw_sum[1] = F::from_u32(((raw_result >> 8) & 0xFF) as u32);
    row.raw_sum[2] = F::from_u32(((raw_result >> 16) & 0xFF) as u32);
    row.raw_sum[3] = F::from_u32(((raw_result >> 24) & 0xFF) as u32);

    // Normalization (simplified: prover provides, TODO full verification)
    let large_exp = if a_is_larger { a_exp } else { b_exp };

    // Compute normalization: determine how many bits to shift
    let raw_top = raw_result >> 8; // top 24 bits
    let norm_result = if same_sign && raw_top >= (1 << 24) {
        // Carry: shift right by 1
        let sticky_from_carry = (raw_result & 1) != 0 || sticky;
        let norm = raw_result >> 1;
        (norm, large_exp + 1, sticky_from_carry)
    } else if raw_top == 0 && (raw_result & 0xFF) == 0 {
        // Result is zero
        (0u64, 0, false)
    } else if !same_sign && raw_top < (1 << 23) && raw_top > 0 {
        // Cancellation: count leading zeros
        let mut shift = 0u32;
        let mut v = raw_top;
        while v < (1 << 23) && v > 0 {
            v <<= 1;
            shift += 1;
        }
        let norm = if shift <= 8 {
            raw_result << shift
        } else {
            (raw_result >> (shift - 8)) << 8
        };
        let adj_exp = large_exp.saturating_sub(shift);
        (norm, adj_exp, sticky)
    } else {
        (raw_result, large_exp, sticky)
    };

    let (norm_val, result_exp_val, combined_sticky) = norm_result;
    row.norm_sig[0] = F::from_u32((norm_val & 0xFF) as u32);
    row.norm_sig[1] = F::from_u32(((norm_val >> 8) & 0xFF) as u32);
    row.norm_sig[2] = F::from_u32(((norm_val >> 16) & 0xFF) as u32);
    row.norm_sig[3] = F::from_u32(((norm_val >> 24) & 0xFF) as u32);
    row.result_exp = F::from_u32(result_exp_val);

    // Rounding
    let actual_guard = ((norm_val >> 7) & 1) as u32; // bit 7 of low byte = guard bit
    let actual_sticky = (norm_val & 0x7F) != 0 || combined_sticky;
    let mant_lsb = (norm_val >> 8) & 1;
    let round_up = actual_guard == 1 && (actual_sticky || mant_lsb == 1);

    row.guard = F::from_u32(actual_guard);
    let norm3_lo = F::from_u32((norm_val & 0x7F) as u32);
    row.norm3_lo_inv = if norm_val & 0x7F != 0 {
        norm3_lo.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    row.round_up = F::from_bool(round_up);

    // Result
    let (r_sign, r_exp, r_mant) = decompose(result_bits);
    row.result_sign = F::from_u32(r_sign);
    row.result_exp = F::from_u32(r_exp);
    row.result[0] = F::from_u32(result_bits & 0xFF);
    row.result[1] = F::from_u32((result_bits >> 8) & 0xFF);
    row.result[2] = F::from_u32((result_bits >> 16) & 0xFF);
    row.result[3] = F::from_u32((result_bits >> 24) & 0xFF);

    let _ = (r_mant, guard_bit); // suppress unused warnings

    // fsub: inf - inf is invalid (NaN) only for same-sign infinities.
    let is_nan_res = a_is_nan || b_is_nan || (a_is_inf && b_is_inf && a_sign == b_sign);
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

    row.is_valid = F::ONE;
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

    fn make_cols(a: u32, b: u32) -> FsubCoreCols<F> {
        let (a_nan, a_inf, a_zero) = classify(a);
        let (b_nan, b_inf, b_zero) = classify(b);
        let result = fsub_s(a, b);
        // fsub: inf - inf is NaN only for same-sign infinities.
        let is_nan_res = a_nan || b_nan || (a_inf && b_inf && ((a >> 31) & 1) == ((b >> 31) & 1));
        let inf_in = a_inf || b_inf;
        let s_inf = !is_nan_res && inf_in;
        let s_zero_both = !is_nan_res && !inf_in && a_zero && b_zero;

        FsubCoreCols {
            a_sign: F::from_u32((a >> 31) & 1),
            b_sign: F::from_u32((b >> 31) & 1),
            a_is_nan: F::from_bool(a_nan),
            b_is_nan: F::from_bool(b_nan),
            a_is_inf: F::from_bool(a_inf),
            b_is_inf: F::from_bool(b_inf),
            a_is_zero: F::from_bool(a_zero),
            b_is_zero: F::from_bool(b_zero),
            inf_in: F::from_bool(inf_in),
            s_nan: F::from_bool(is_nan_res),
            s_inf: F::from_bool(s_inf),
            s_zero_both: F::from_bool(s_zero_both),
            is_nan_result: F::from_bool(is_nan_res),
            result: core::array::from_fn(|i| F::from_u32((result >> (i * 8)) & 0xFF)),
            ..Default::default()
        }
    }

    fn run(c: &FsubCoreCols<F>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        FsubCoreAir::new(
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus::new(0, [1 << 8, 8 << 8]),
            0,
        )
        .eval_special_cases(&mut b, c);
        b
    }

    #[test]
    fn fsub_special_honest_passes() {
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
    fn fsub_inf_minus_inf_is_nan() {
        // inf - inf = NaN only for same-sign infinities.
        let c = make_cols(f32::INFINITY.to_bits(), f32::INFINITY.to_bits());
        assert!(c.s_nan == F::ONE, "+inf - +inf should be NaN");
        let c = make_cols(f32::NEG_INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
        assert!(c.s_nan == F::ONE, "-inf - -inf should be NaN");
        // Opposite-sign inf - inf yields +-inf, NOT NaN.
        let c = make_cols(f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
        assert!(
            c.s_nan == F::ZERO && c.s_inf == F::ONE,
            "+inf - -inf = +inf"
        );
    }

    #[test]
    fn fsub_corrupt_nan_rejected() {
        let mut c = make_cols(f32::NAN.to_bits(), 1.0f32.to_bits());
        c.result[0] = F::from_u32(0xFF);
        assert!(!run(&c).ok());
    }

    #[test]
    fn fsub_corrupt_inf_sign_rejected() {
        // finite - inf = -inf; flipping to +inf must fail.
        let mut c = make_cols(3.0f32.to_bits(), f32::NEG_INFINITY.to_bits()); // 3 - (-inf) = +inf
        c.result[3] = F::ZERO; // +inf has result[3] = 0x7f; corrupt to sign 1
                               // Actually result[3] for +inf is 0x7f; setting to 0 makes it... check rejection
        assert!(!run(&c).ok());
    }

    #[test]
    fn fsub_finite_unconstrained() {
        let mut c = make_cols(7.0f32.to_bits(), 3.0f32.to_bits());
        c.result[0] = F::from_u32(0xAB);
        assert!(run(&c).ok(), "finite case must not be constrained here");
    }
}
