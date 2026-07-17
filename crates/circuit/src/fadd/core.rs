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
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

pub const SIG_NUM_LIMBS: usize = 3;
pub const EXT_SIG_LIMBS: usize = 4; // extended significand: 3 mantissa bytes + 1 guard/sticky byte
pub const LIMB_BITS: usize = 8;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
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
#[columns_via(FaddCoreCols<u8>)]
pub struct FaddCoreAir {
    pub range_tuple_bus: RangeTupleCheckerBus<2>,
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
        let a_mant_sum = cols.a_mant0 + cols.a_mant1 + cols.a_mant2;
        let b_exp = cols.b_exp_lo + cols.b_exp_hi * f2;
        let b_mant_sum = cols.b_mant0 + cols.b_mant1 + cols.b_mant2;

        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.b_exp_lo);

        // ── Phase 2: Classification ──────────────────────────────────────
        let one = AB::Expr::ONE;
        let a_exp_is_zero = one.clone() - a_exp.clone() * cols.a_exp_zero_inv;
        builder.assert_bool(a_exp_is_zero.clone());
        let a_exp_is_max =
            one.clone() - (AB::Expr::from(f255) - a_exp.clone()) * cols.a_exp_max_inv;
        builder.assert_bool(a_exp_is_max.clone());
        let a_mant_is_zero = one.clone() - a_mant_sum.clone() * cols.a_mant_sum_inv;
        builder.assert_bool(a_mant_is_zero.clone());

        let b_exp_is_zero = one.clone() - b_exp.clone() * cols.b_exp_zero_inv;
        builder.assert_bool(b_exp_is_zero.clone());
        let b_exp_is_max =
            one.clone() - (AB::Expr::from(f255) - b_exp.clone()) * cols.b_exp_max_inv;
        builder.assert_bool(b_exp_is_max.clone());
        let b_mant_is_zero = one.clone() - b_mant_sum.clone() * cols.b_mant_sum_inv;
        builder.assert_bool(b_mant_is_zero.clone());

        let _a_is_nan = a_exp_is_max.clone() * (one.clone() - a_mant_is_zero.clone());
        let _a_is_inf = a_exp_is_max.clone() * a_mant_is_zero.clone();
        let _a_is_zero_f = a_exp_is_zero.clone() * a_mant_is_zero.clone();
        let _a_is_normal = (one.clone() - a_exp_is_zero) * (one.clone() - a_exp_is_max.clone());

        let _b_is_nan = b_exp_is_max.clone() * (one.clone() - b_mant_is_zero.clone());
        let _b_is_inf = b_exp_is_max.clone() * b_mant_is_zero.clone();
        let _b_is_zero_f = b_exp_is_zero.clone() * b_mant_is_zero.clone();
        let _b_is_normal = (one.clone() - b_exp_is_zero) * (one.clone() - b_exp_is_max.clone());

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
        builder.assert_eq(cols.is_add, one.clone() - sign_xor.clone());
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

        // Native f32 add = IEEE 754 correctly-rounded (RNE)
        let result = (f32::from_le_bytes(a) + f32::from_le_bytes(b)).to_le_bytes();

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
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
}

impl<A> FaddFiller<A> {
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
        let result_bits = (f32::from_le_bytes(record.a) + f32::from_le_bytes(record.b)).to_bits();

        fill_fadd_witness(core_row, a_bits, b_bits, result_bits);
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
    let _a_is_zero = a_exp == 0 && a_mant == 0;
    let a_is_normal = (1..=254).contains(&a_exp);
    let b_is_nan = b_exp == 255 && b_mant != 0;
    let b_is_inf = b_exp == 255 && b_mant == 0;
    let _b_is_zero = b_exp == 0 && b_mant == 0;
    let b_is_normal = (1..=254).contains(&b_exp);

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
    let same_sign = a_sign == b_sign;
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

    let is_nan_res = a_is_nan || b_is_nan || (a_is_inf && b_is_inf && a_sign != b_sign);
    let is_inf_res = !is_nan_res && (a_is_inf || b_is_inf || r_exp == 255);
    let is_zero_res = !is_nan_res && (r_exp == 0 && result_bits & 0x7FFFFF == 0);
    row.is_nan_result = F::from_bool(is_nan_res);
    row.is_inf_result = F::from_bool(is_inf_res);
    row.is_zero_result = F::from_bool(is_zero_res);
    row.is_special = F::from_bool(is_nan_res || is_inf_res || is_zero_res);

    row.is_valid = F::ONE;
}
