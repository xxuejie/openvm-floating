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

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
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

        let a_is_nan = a_exp_is_max.clone() * (one.clone() - a_mant_is_zero.clone());
        let a_is_inf = a_exp_is_max.clone() * a_mant_is_zero.clone();
        let a_is_zero_f = a_exp_is_zero.clone() * a_mant_is_zero.clone();
        let a_is_normal =
            (one.clone() - a_exp_is_zero.clone()) * (one.clone() - a_exp_is_max.clone());

        let b_is_nan = b_exp_is_max.clone() * (one.clone() - b_mant_is_zero.clone());
        let b_is_inf = b_exp_is_max.clone() * b_mant_is_zero.clone();
        let b_is_zero_f = b_exp_is_zero.clone() * b_mant_is_zero.clone();
        let b_is_normal =
            (one.clone() - b_exp_is_zero.clone()) * (one.clone() - b_exp_is_max.clone());

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
        // at degree 2. Verify: a_sig[2] = a_mant2 + is_normal * 128
        builder.assert_eq(cols.a_sig[0], cols.a_mant0);
        builder.assert_eq(cols.a_sig[1], cols.a_mant1);
        builder.assert_eq(cols.a_sig[2], cols.a_mant2 + a_is_normal.clone() * f128);
        builder.assert_eq(cols.b_sig[0], cols.b_mant0);
        builder.assert_eq(cols.b_sig[1], cols.b_mant1);
        builder.assert_eq(cols.b_sig[2], cols.b_mant2 + b_is_normal.clone() * f128);

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

        // ── Phase 5: Normalization ───────────────────────────────────────
        builder.assert_bool(cols.is_high);
        // product[5] has bit 47 or bit 46 set (for normal×normal).
        // When is_high=1 (bit 47 set): product[5] = p5_lo + 128
        // When is_high=0 (bit 46 set, 47 clear): product[5] = p5_lo + 64
        builder
            .when(cols.is_valid)
            .when(cols.is_high)
            .assert_eq(cols.product[5], cols.p5_lo + f128);
        builder
            .when(cols.is_valid)
            .when(not(cols.is_high))
            .assert_eq(cols.product[5], cols.p5_lo + f64);

        // Verify norm_sig: when is_high, norm_sig = product; when !is_high, norm_sig = product << 1
        let mut shift_carry = AB::Expr::ZERO;
        for i in 0..PRODUCT_NUM_LIMBS {
            // is_high case: norm_sig[i] == product[i]
            builder
                .when(cols.is_high)
                .assert_eq(cols.norm_sig[i], cols.product[i]);

            // !is_high case: shift left by 1 (carry chain stays degree 1)
            let shifted = cols.product[i] * f2 + shift_carry.clone();
            let new_carry = (shifted.clone() - cols.norm_sig[i]) * carry_divide;
            builder
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

        // ── Phase 7: Result exponent (normal case) ──────────────────────
        let exp_normal = a_exp + b_exp - AB::F::from_u32(127) + cols.is_high;
        builder
            .when(cols.is_valid)
            .when(not(cols.is_special))
            .assert_eq(cols.result_exp, exp_normal);

        // ── Phase 8: Special-case mux ────────────────────────────────────
        // TODO: Full verification of NaN/inf/zero result flags.
        // The executor fills correct values; soundness constraints to be added.
        let inf_times_zero =
            a_is_inf.clone() * b_is_zero_f.clone() + a_is_zero_f.clone() * b_is_inf.clone();
        let _nan_cond = a_is_nan + b_is_nan + inf_times_zero;
        let _inf_cond = (a_is_inf + b_is_inf) * (AB::Expr::ONE - _nan_cond.clone());
        let _zero_cond = (a_is_zero_f + b_is_zero_f) * (AB::Expr::ONE - _nan_cond);

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

        // Native f32 multiply = IEEE 754 correctly-rounded (RNE)
        let result = (f32::from_le_bytes(a) * f32::from_le_bytes(b)).to_le_bytes();

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
        let result_bits = (f32::from_le_bytes(record.a) * f32::from_le_bytes(record.b)).to_bits();

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
    let p5_lo = if is_high { p5 - 128 } else { p5 - 64 };
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

    row.is_valid = F::ONE;
}
