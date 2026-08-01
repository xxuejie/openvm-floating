use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    is_less_than_array::{
        IsLtArrayAuxCols, IsLtArrayAuxColsMut, IsLtArrayAuxColsRef, IsLtArrayIo, IsLtArraySubAir,
    },
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

/// Number of `lt_decomp` limbs in [IsLtArrayAuxCols]: the comparison uses
/// `max_bits = 8` per key limb, so `8.div_ceil(range_max_bits)` limbs are needed.
/// Warning: This requires that `(8 + range_max_bits - 1) / range_max_bits = LT_AUX_LEN`,
/// i.e. the system range checker must have `range_max_bits >= 8`.
pub const LT_AUX_LEN: usize = 1;

/// Maximum bits per key limb for the lexicographic less-than check.
pub const KEY_LIMB_BITS: usize = 8;

/// Monotone map from f32 bits to a u32 such that for non-NaN floats,
/// `a < b` (as floats, with -0.0 < +0.0) iff `fp_key(a) < fp_key(b)` (as u32).
/// Positive floats: set the top bit. Negative floats: flip all bits.
#[inline(always)]
pub fn fp_key(bits: u32) -> u32 {
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// IEEE 754 / RISC-V FMIN.S (is_min = true) and FMAX.S (is_min = false).
/// - If both operands are NaN, the result is the canonical NaN.
/// - If exactly one operand is NaN, the result is the other operand.
/// - Otherwise the smaller (larger) operand is returned, with -0.0 < +0.0.
#[inline(always)]
pub fn fminmax_result(a: u32, b: u32, is_min: bool) -> u32 {
    let a_nan = f32::from_bits(a).is_nan();
    let b_nan = f32::from_bits(b).is_nan();
    if a_nan && b_nan {
        CANONICAL_NAN
    } else if a_nan {
        b
    } else if b_nan {
        a
    } else {
        let (ka, kb) = (fp_key(a), fp_key(b));
        let pick_a = if is_min { ka <= kb } else { ka >= kb };
        if pick_a {
            a
        } else {
            b
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FminmaxCoreCols<T> {
    pub a: [T; 4],
    pub b: [T; 4],

    // === Decomposition of inputs ===
    pub a_sign: T,
    pub b_sign: T,
    pub a_exp_lo: T, // bit 0 of exponent (top bit of byte 2)
    pub b_exp_lo: T,
    pub a_mant2: T, // low 7 bits of byte 2
    pub b_mant2: T,
    pub a_hi: T, // low 7 bits of byte 3 (high 7 bits of exponent)
    pub b_hi: T,

    // === Monotone comparison keys (little-endian limbs) ===
    pub key_a: [T; 4],
    pub key_b: [T; 4],

    // === Classification inverse witnesses and flags ===
    pub a_exp_max_inv: T,
    pub b_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub b_mant_sum_inv: T,
    pub a_exp_is_max: T,
    pub b_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub b_mant_is_zero: T,
    pub a_nan: T,
    pub b_nan: T,
    pub both_nan: T,

    // === Less-than: key_a < key_b lexicographically (big-endian limb order) ===
    pub lt_aux_cols: IsLtArrayAuxCols<T, 4, LT_AUX_LEN>,
    pub a_lt_b: T,

    // === Selection ===
    pub pick_a: T, // comparison winner: take a (ignoring NaN cases)
    pub sel_a: T,  // final: result bytes come from a
    pub fmin_flag: T,
    pub fmax_flag: T,

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FminmaxCoreCols<u8>)]
pub struct FminmaxCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FminmaxCoreAir {
    fn width(&self) -> usize {
        FminmaxCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FminmaxCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FminmaxCoreAir
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
        let cols: &FminmaxCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.b_exp_lo);
        builder.assert_bool(cols.a_exp_is_max);
        builder.assert_bool(cols.b_exp_is_max);
        builder.assert_bool(cols.a_mant_is_zero);
        builder.assert_bool(cols.b_mant_is_zero);
        builder.assert_bool(cols.a_nan);
        builder.assert_bool(cols.b_nan);
        builder.assert_bool(cols.both_nan);
        builder.assert_bool(cols.pick_a);
        builder.assert_bool(cols.sel_a);
        builder.assert_bool(cols.fmin_flag);
        builder.assert_bool(cols.fmax_flag);

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.fmin_flag + cols.fmax_flag);

        // ── Byte decomposition ───────────────────────────────────────────
        // a[2] = a_mant2 + a_exp_lo * 128, a[3] = a_hi + a_sign * 128
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        builder.assert_eq(cols.b[2], cols.b_mant2 + cols.b_exp_lo * f128);
        builder.assert_eq(cols.b[3], cols.b_hi + cols.b_sign * f128);
        // Pin the decomposition: the 7-bit pieces must be range-checked, or
        // a_exp_lo / a_sign are free field choices.
        for v in [cols.a_mant2, cols.b_mant2, cols.a_hi, cols.b_hi] {
            self.range_bus
                .range_check(v.into(), 7)
                .eval(builder, cols.is_valid.into());
        }

        // ── Monotone comparison keys ─────────────────────────────────────
        // Bytes 0-2: key = sign ? 255 - x : x
        for i in 0..3 {
            builder.assert_eq(
                cols.key_a[i],
                cols.a[i] + cols.a_sign * (AB::Expr::from(f255) - cols.a[i] * f2),
            );
            builder.assert_eq(
                cols.key_b[i],
                cols.b[i] + cols.b_sign * (AB::Expr::from(f255) - cols.b[i] * f2),
            );
        }
        // Byte 3: key = sign ? 127 - hi : hi + 128. Gated by is_valid so that
        // the all-zero padding row (key = 0 vs constant 128) is unconstrained.
        builder.when(cols.is_valid).assert_eq(
            cols.key_a[3],
            cols.a_hi + f128 - cols.a_sign * (cols.a_hi * f2 + one.clone()),
        );
        builder.when(cols.is_valid).assert_eq(
            cols.key_b[3],
            cols.b_hi + f128 - cols.b_sign * (cols.b_hi * f2 + one.clone()),
        );
        // Range check key limbs to 8 bits so the lexicographic comparison is sound.
        for i in 0..4 {
            self.range_bus
                .range_check(cols.key_a[i].into(), KEY_LIMB_BITS)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.key_b[i].into(), KEY_LIMB_BITS)
                .eval(builder, cols.is_valid.into());
        }

        // ── Classification ───────────────────────────────────────────────
        // exp = exp_lo + 2 * hi, mant_sum = a[0] + a[1] + a_mant2
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let b_exp = cols.b_exp_lo + cols.b_hi * f2;
        let a_exp_max_delta = AB::Expr::from(f255) - a_exp;
        let b_exp_max_delta = AB::Expr::from(f255) - b_exp;
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_max,
            one.clone() - a_exp_max_delta.clone() * cols.a_exp_max_inv,
        );
        builder.assert_zero(a_exp_max_delta * cols.a_exp_is_max);
        builder.when(cols.is_valid).assert_eq(
            cols.b_exp_is_max,
            one.clone() - b_exp_max_delta.clone() * cols.b_exp_max_inv,
        );
        builder.assert_zero(b_exp_max_delta * cols.b_exp_is_max);

        let a_mant_sum = cols.a[0] + cols.a[1] + cols.a_mant2;
        let b_mant_sum = cols.b[0] + cols.b[1] + cols.b_mant2;
        builder.when(cols.is_valid).assert_eq(
            cols.a_mant_is_zero,
            one.clone() - a_mant_sum.clone() * cols.a_mant_sum_inv,
        );
        builder.assert_zero(a_mant_sum * cols.a_mant_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.b_mant_is_zero,
            one.clone() - b_mant_sum.clone() * cols.b_mant_sum_inv,
        );
        builder.assert_zero(b_mant_sum * cols.b_mant_is_zero);

        // NaN = (exp == 255) && (mant != 0)
        builder.assert_eq(
            cols.a_nan,
            cols.a_exp_is_max * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(
            cols.b_nan,
            cols.b_exp_is_max * (one.clone() - cols.b_mant_is_zero),
        );
        builder.assert_eq(cols.both_nan, cols.a_nan * cols.b_nan);

        // ── Comparison: a_lt_b = (key_a < key_b) lexicographically ───────
        let lt_air = IsLtArraySubAir::<4>::new(self.range_bus, KEY_LIMB_BITS);
        let io = IsLtArrayIo {
            x: [
                cols.key_a[3].into(),
                cols.key_a[2].into(),
                cols.key_a[1].into(),
                cols.key_a[0].into(),
            ],
            y: [
                cols.key_b[3].into(),
                cols.key_b[2].into(),
                cols.key_b[1].into(),
                cols.key_b[0].into(),
            ],
            out: cols.a_lt_b.into(),
            count: cols.is_valid.into(),
        };
        let aux = IsLtArrayAuxColsRef {
            diff_marker: &cols.lt_aux_cols.diff_marker,
            diff_inv: &cols.lt_aux_cols.diff_inv,
            lt_decomp: &cols.lt_aux_cols.lt_aux.lower_decomp,
        };
        lt_air.eval(builder, (io, aux));

        // ── Selection ────────────────────────────────────────────────────
        // Comparison winner (non-NaN case): fmin takes the smaller, fmax the larger.
        // Equal keys imply bitwise-equal operands, so either pick is correct.
        builder.assert_eq(
            cols.pick_a,
            cols.fmin_flag * cols.a_lt_b + cols.fmax_flag * (one.clone() - cols.a_lt_b),
        );
        // Final source selection:
        // - both NaN: canonical NaN
        // - only a NaN: b
        // - only b NaN: a
        // - else: pick_a ? a : b
        let not_a_nan = one.clone() - cols.a_nan;
        let not_b_nan = one.clone() - cols.b_nan;
        builder.assert_eq(
            cols.sel_a,
            not_a_nan.clone() * not_b_nan * cols.pick_a + cols.b_nan * not_a_nan,
        );

        let canon = CANONICAL_NAN.to_le_bytes();
        for (i, &canon_byte) in canon.iter().enumerate() {
            let canon_i = AB::Expr::from(AB::F::from_u32(canon_byte as u32));
            let other = cols.b[i] + cols.both_nan * (canon_i - cols.b[i]);
            builder.assert_eq(
                cols.result[i],
                cols.sel_a * cols.a[i] + (one.clone() - cols.sel_a) * other,
            );
        }

        let expected_opcode = cols.fmin_flag * AB::F::from_u32(FpOpcode::FMIN_S as u32)
            + cols.fmax_flag * AB::F::from_u32(FpOpcode::FMAX_S as u32)
            + AB::F::from_usize(self.offset);

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

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FminmaxCoreRecord {
    pub a: [u8; 4],
    pub b: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FminmaxExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub register_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FminmaxExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 2]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FminmaxCoreRecord),
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

        let [a, b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.b = b;
        core_record.opcode = local as u8;

        let result = match FpOpcode::from_usize(local) {
            FpOpcode::FMIN_S => fminmax_result(u32::from_le_bytes(a), u32::from_le_bytes(b), true),
            FpOpcode::FMAX_S => fminmax_result(u32::from_le_bytes(a), u32::from_le_bytes(b), false),
            _ => unreachable!(),
        };
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

pub struct FminmaxFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub lt_air: IsLtArraySubAir<4>,
}

impl<A> FminmaxFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        let lt_air = IsLtArraySubAir::<4>::new(range_checker.bus(), KEY_LIMB_BITS);
        Self {
            adapter,
            offset,
            range_checker,
            lt_air,
        }
    }
}

impl<A, F> TraceFiller<F> for FminmaxFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FminmaxCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FminmaxCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let b_bits = u32::from_le_bytes(record.b);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let is_min = opcode == FpOpcode::FMIN_S;
        let result_bits = fminmax_result(a_bits, b_bits, is_min);

        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result_bits >> (i * 8)) & 0xFF);
        }
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

        // Comparison keys
        let (ka, kb) = (fp_key(a_bits), fp_key(b_bits));
        let key_a: [F; 4] = core::array::from_fn(|i| F::from_u32((ka >> (i * 8)) & 0xFF));
        let key_b: [F; 4] = core::array::from_fn(|i| F::from_u32((kb >> (i * 8)) & 0xFF));
        for i in 0..4 {
            core_row.key_a[i] = key_a[i];
            core_row.key_b[i] = key_b[i];
            self.range_checker
                .add_count(key_a[i].as_canonical_u32(), KEY_LIMB_BITS);
            self.range_checker
                .add_count(key_b[i].as_canonical_u32(), KEY_LIMB_BITS);
        }

        // Classification witnesses
        let a_exp_f = F::from_u32(a_exp);
        let b_exp_f = F::from_u32(b_exp);
        core_row.a_exp_is_max = F::from_bool(a_exp == 255);
        core_row.b_exp_is_max = F::from_bool(b_exp == 255);
        core_row.a_exp_max_inv = if a_exp != 255 {
            (F::from_u32(255) - a_exp_f)
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.b_exp_max_inv = if b_exp != 255 {
            (F::from_u32(255) - b_exp_f)
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
        let b_mant_sum = (b_mant & 0xFF) + ((b_mant >> 8) & 0xFF) + ((b_mant >> 16) & 0x7F);
        core_row.a_mant_is_zero = F::from_bool(a_mant == 0);
        core_row.b_mant_is_zero = F::from_bool(b_mant == 0);
        core_row.a_mant_sum_inv = if a_mant != 0 {
            F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.b_mant_sum_inv = if b_mant != 0 {
            F::from_u32(b_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        core_row.a_nan = F::from_bool(a_nan);
        core_row.b_nan = F::from_bool(b_nan);
        core_row.both_nan = F::from_bool(a_nan && b_nan);

        // Less-than auxiliary columns (also updates range checker multiplicities)
        let x = [key_a[3], key_a[2], key_a[1], key_a[0]];
        let y = [key_b[3], key_b[2], key_b[1], key_b[0]];
        let aux = IsLtArrayAuxColsMut {
            diff_marker: &mut core_row.lt_aux_cols.diff_marker,
            diff_inv: &mut core_row.lt_aux_cols.diff_inv,
            lt_decomp: &mut core_row.lt_aux_cols.lt_aux.lower_decomp,
        };
        self.lt_air.generate_subrow(
            (self.range_checker.as_ref(), &x, &y),
            (aux, &mut core_row.a_lt_b),
        );

        let a_lt_b = ka < kb;
        core_row.pick_a = F::from_bool(if is_min { a_lt_b } else { !a_lt_b });
        core_row.sel_a = F::from_bool(match (a_nan, b_nan) {
            (true, true) => false,
            (true, false) => false,
            (false, true) => true,
            (false, false) => (is_min && a_lt_b) || (!is_min && !a_lt_b),
        });
        core_row.fmin_flag = F::from_bool(is_min);
        core_row.fmax_flag = F::from_bool(!is_min);
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

#[cfg(test)]
mod air_tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use std::sync::Arc;

    type F = BabyBear;
    type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;

    fn build_row(a_bits: u32, b_bits: u32, is_min: bool) -> Vec<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt_air = IsLtArraySubAir::<4>::new(chip.bus(), KEY_LIMB_BITS);
        let mut buf = vec![F::ZERO; FminmaxCoreCols::<F>::width()];
        let row: &mut FminmaxCoreCols<F> = buf.as_mut_slice().borrow_mut();
        let result_bits = fminmax_result(a_bits, b_bits, is_min);
        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        let (b_sign, b_exp, b_mant) = decompose(b_bits);

        for i in 0..4 {
            row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            row.b[i] = F::from_u32((b_bits >> (i * 8)) & 0xFF);
            row.result[i] = F::from_u32((result_bits >> (i * 8)) & 0xFF);
        }
        row.a_sign = F::from_u32(a_sign);
        row.b_sign = F::from_u32(b_sign);
        row.a_exp_lo = F::from_u32(a_exp & 1);
        row.b_exp_lo = F::from_u32(b_exp & 1);
        row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        row.b_mant2 = F::from_u32((b_mant >> 16) & 0x7F);
        row.a_hi = F::from_u32(a_exp >> 1);
        row.b_hi = F::from_u32(b_exp >> 1);

        let (ka, kb) = (fp_key(a_bits), fp_key(b_bits));
        let key_a: [F; 4] = core::array::from_fn(|i| F::from_u32((ka >> (i * 8)) & 0xFF));
        let key_b: [F; 4] = core::array::from_fn(|i| F::from_u32((kb >> (i * 8)) & 0xFF));
        row.key_a = key_a;
        row.key_b = key_b;

        row.a_exp_is_max = F::from_bool(a_exp == 255);
        row.b_exp_is_max = F::from_bool(b_exp == 255);
        row.a_exp_max_inv = if a_exp != 255 {
            (F::from_u32(255) - F::from_u32(a_exp))
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.b_exp_max_inv = if b_exp != 255 {
            (F::from_u32(255) - F::from_u32(b_exp))
                .try_inverse()
                .unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_mant_sum = (a_mant & 0xFF) + ((a_mant >> 8) & 0xFF) + ((a_mant >> 16) & 0x7F);
        let b_mant_sum = (b_mant & 0xFF) + ((b_mant >> 8) & 0xFF) + ((b_mant >> 16) & 0x7F);
        row.a_mant_is_zero = F::from_bool(a_mant == 0);
        row.b_mant_is_zero = F::from_bool(b_mant == 0);
        row.a_mant_sum_inv = if a_mant != 0 {
            F::from_u32(a_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        row.b_mant_sum_inv = if b_mant != 0 {
            F::from_u32(b_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        let a_nan = a_exp == 255 && a_mant != 0;
        let b_nan = b_exp == 255 && b_mant != 0;
        row.a_nan = F::from_bool(a_nan);
        row.b_nan = F::from_bool(b_nan);
        row.both_nan = F::from_bool(a_nan && b_nan);

        let x = [key_a[3], key_a[2], key_a[1], key_a[0]];
        let y = [key_b[3], key_b[2], key_b[1], key_b[0]];
        let aux = IsLtArrayAuxColsMut {
            diff_marker: &mut row.lt_aux_cols.diff_marker,
            diff_inv: &mut row.lt_aux_cols.diff_inv,
            lt_decomp: &mut row.lt_aux_cols.lt_aux.lower_decomp,
        };
        lt_air.generate_subrow((chip.as_ref(), &x, &y), (aux, &mut row.a_lt_b));

        let a_lt_b = ka < kb;
        row.pick_a = F::from_bool(if is_min { a_lt_b } else { !a_lt_b });
        row.sel_a = F::from_bool(match (a_nan, b_nan) {
            (true, true) => false,
            (true, false) => false,
            (false, true) => true,
            (false, false) => (is_min && a_lt_b) || (!is_min && !a_lt_b),
        });
        row.fmin_flag = F::from_bool(is_min);
        row.fmax_flag = F::from_bool(!is_min);
        row.is_valid = F::ONE;
        buf
    }

    fn eval_row(buf: &[F]) -> NumericBuilder<F> {
        let air = FminmaxCoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut b = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
        b
    }

    #[test]
    fn honest_rows_pass() {
        for (a, b) in [
            (1.0f32.to_bits(), 2.0f32.to_bits()),
            (0x7FC00000, 0x3F800000), // qNaN vs 1.0
            (0x7FC00000, 0x7FA00001), // both NaN
            (0x80000000, 0x00000000), // -0 vs +0
            (0x7F800000, 0xFF800000), // inf vs -inf
        ] {
            for is_min in [true, false] {
                let buf = build_row(a, b, is_min);
                assert!(
                    eval_row(&buf).ok(),
                    "fminmax min={is_min} honest failed: a={a:08x} b={b:08x}"
                );
            }
        }
    }

    // NOTE: decomposition-flip attacks (exp_lo/sign with wrapped lo bytes)
    // are caught only by the 7-bit range-check *bus sends*, which
    // NumericBuilder ignores. They must be covered by prover-level
    // bite-checks (hardening item 2), not numeric tests.

    /// Every column must be constrained: perturbing any single column by +1
    /// must break at least one gate constraint (except columns whose only
    /// constraint is a range-check bus send — those are whitelisted here).
    #[test]
    fn corruption_sweep_finds_no_unconstrained_witness() {
        use crate::test_utils::corruption_sweep;
        use openvm_circuit_primitives::StructReflectionHelper;

        let names = FminmaxCoreCols::<F>::struct_reflection().unwrap_or_default();
        // Whitelist rationale:
        // - range-bus-only: NumericBuilder ignores bus sends, so +1 is accepted;
        //   the real prover range-checks them (see eval).
        // - `a/b_exp_max_inv`, `a/b_mant_sum_inv`: `flag = 1 - delta * inv` with
        //   `assert_zero(delta * flag)`; `inv` is multiplied by 0 exactly when the
        //   corresponding delta is 0 (exp 255 / zero mantissa), where the flag is
        //   pinned to 1 by the other side of the equation.
        // - `lt_aux_cols`: IsLtArraySubAir aux; only the limbs of a zero diff are
        //   free (nothing to decompose), covered by the diff != 0 classes below.
        let whitelist: std::collections::HashSet<usize> = {
            let mut s = std::collections::HashSet::new();
            for name in [
                "a_mant2",
                "b_mant2",
                "a_hi",
                "b_hi",
                "key_a",
                "key_b",
                "a_exp_max_inv",
                "b_exp_max_inv",
                "a_mant_sum_inv",
                "b_mant_sum_inv",
                "lt_aux_cols",
            ] {
                for (i, n) in names.iter().enumerate() {
                    if n == name || n.starts_with(&format!("{name}__")) {
                        s.insert(i);
                    }
                }
            }
            s
        };

        let classes: &[(u32, u32)] = &[
            (1.0f32.to_bits(), 2.0f32.to_bits()),
            (0x7FC00000, 0x3F800000), // qNaN vs 1.0
            (0x7FC00000, 0x7FA00001), // both NaN
            (0x80000000, 0x00000000), // -0 vs +0
            (0x7F800000, 0xFF800000), // inf vs -inf
            (0x00400000, 0x3F800000), // subnormal vs 1.0
        ];

        for is_min in [true, false] {
            for (a, b) in classes {
                let row = build_row(*a, *b, is_min);
                let accepted = corruption_sweep(&row, |builder, cols| {
                    VmCoreAir::<NumericBuilder<F>, TestI>::eval(
                        &FminmaxCoreAir::new(
                            VariableRangeCheckerBus::new(0, 16),
                            FpOpcode::CLASS_OFFSET,
                        ),
                        builder,
                        cols,
                        F::ZERO,
                    );
                });
                let unexpected: Vec<usize> = accepted
                    .into_iter()
                    .filter(|i| !whitelist.contains(i))
                    .collect();
                assert!(
                    unexpected.is_empty(),
                    "fminmax min={is_min} a={a:08x} b={b:08x}: unconstrained columns {:?}",
                    unexpected
                        .iter()
                        .map(|&i| (i, names.get(i).cloned().unwrap_or_default()))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
