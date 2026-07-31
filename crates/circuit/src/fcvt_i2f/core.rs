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
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::fp_shift::{run_shrn, run_shrn_round, ShrnCols, ShrnGadget, SHLIMB_BITS};

/// IEEE 754 / RISC-V FCVT.S.W: convert i32 to f32 with RNE rounding.
#[inline(always)]
pub fn fcvt_s_w(x: u32) -> u32 {
    ((x as i32) as f32).to_bits()
}

/// IEEE 754 / RISC-V FCVT.S.WU: convert u32 to f32 with RNE rounding.
#[inline(always)]
pub fn fcvt_s_wu(x: u32) -> u32 {
    (x as f32).to_bits()
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FcvtI2FCoreCols<T> {
    pub a: [T; 4], // integer input bytes

    // === Input decomposition ===
    pub a_sign: T, // bit 31 (signed interpretation)
    pub a_hi: T,   // low 7 bits of byte 3
    pub a_sum_inv: T,
    pub a_is_zero: T,

    // === Result float decomposition ===
    pub r_sign: T,
    pub r_exp_lo: T,
    pub r_mant2: T,
    pub r_hi: T,
    pub r_exp_zero_inv: T,
    pub r_exp_max_inv: T,
    pub r_mant_sum_inv: T,
    pub r_exp_is_zero: T,
    pub r_mant_is_zero: T,
    pub r_is_zero: T,

    pub ws_flag: T,
    pub wu_flag: T,

    // === Conversion (magnitude normalize + RNE round) ===
    pub mag: [T; 4],          // |integer| bytes (.w neg ⇒ two's complement of a)
    pub neg_carry: [T; 4],    // two's-complement borrow chain for the magnitude
    pub s_convert: T,         // !a_is_zero (the normalize/round path runs)
    pub shift: T,             // result_exp - 126 (right-shift amount of x = mag<<24)
    pub shrn: ShrnCols<T, 7>, // sig = (mag << 24) >> shift, with guard/sticky
    pub y0_high7: T,          // shrn.y[0] >> 1
    pub lsb: T,               // bit 0 of shrn.y[0] (mantissa LSB for RNE tie)
    pub round_up: T,
    pub rc0: T,     // rounding carry out of mantissa byte 0
    pub rc1: T,     // rounding carry out of mantissa byte 1
    pub rc2: T,     // rounding carry into the implicit bit (exponent)
    pub rc2_rem: T, // rounded top 7 mantissa bits
    pub y2_lo7: T,  // shrn.y[2] - 128 (implicit-bit anchor, 7 bits)

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FcvtI2FCoreCols<u8>)]
pub struct FcvtI2FCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FcvtI2FCoreAir {
    fn width(&self) -> usize {
        FcvtI2FCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FcvtI2FCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FcvtI2FCoreAir
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
        let cols: &FcvtI2FCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_is_zero);
        builder.assert_bool(cols.r_sign);
        builder.assert_bool(cols.r_exp_lo);
        builder.assert_bool(cols.r_exp_is_zero);
        builder.assert_bool(cols.r_mant_is_zero);
        builder.assert_bool(cols.r_is_zero);
        builder.assert_bool(cols.ws_flag);
        builder.assert_bool(cols.wu_flag);

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.ws_flag + cols.wu_flag);

        // ── Input decomposition ──────────────────────────────────────────
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        // Pin the decomposition: a_hi must be range-checked to 7 bits, or
        // a_sign is a free field choice (result sign / negation path).
        self.range_bus
            .range_check(cols.a_hi.into(), 7)
            .eval(builder, cols.is_valid.into());
        let a_sum = cols.a[0] + cols.a[1] + cols.a[2] + cols.a[3];
        builder
            .when(cols.is_valid)
            .assert_eq(cols.a_is_zero, one.clone() - a_sum.clone() * cols.a_sum_inv);
        builder.assert_zero(a_sum * cols.a_is_zero);

        // ── Result decomposition ─────────────────────────────────────────
        builder.assert_eq(cols.result[2], cols.r_mant2 + cols.r_exp_lo * f128);
        builder.assert_eq(cols.result[3], cols.r_hi + cols.r_sign * f128);
        let r_exp = cols.r_exp_lo + cols.r_hi * f2;
        let r_mant_sum = cols.result[0] + cols.result[1] + cols.r_mant2;

        // The result of int -> float conversion is never NaN or infinite:
        // prove r_exp != 255 via an inverse.
        builder
            .when(cols.is_valid)
            .assert_one((AB::Expr::from(f255) - r_exp.clone()) * cols.r_exp_max_inv);

        // Zero input iff zero result (+0.0)
        builder.when(cols.is_valid).assert_eq(
            cols.r_exp_is_zero,
            one.clone() - r_exp.clone() * cols.r_exp_zero_inv,
        );
        builder.assert_zero(r_exp * cols.r_exp_is_zero);
        builder.when(cols.is_valid).assert_eq(
            cols.r_mant_is_zero,
            one.clone() - r_mant_sum.clone() * cols.r_mant_sum_inv,
        );
        builder.assert_zero(r_mant_sum * cols.r_mant_is_zero);
        builder.assert_eq(cols.r_is_zero, cols.r_exp_is_zero * cols.r_mant_is_zero);
        builder.assert_eq(cols.r_is_zero, cols.a_is_zero);

        // Result sign: 0 for .wu and for zero input; a_sign for .w otherwise.
        builder.assert_eq(
            cols.r_sign,
            (one.clone() - cols.wu_flag) * cols.a_sign * (one.clone() - cols.a_is_zero),
        );

        // ── Conversion: magnitude normalize + RNE round ─────────────────
        self.eval_convert(builder, cols);

        // Range check input and result bytes.
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.result[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        let expected_opcode = cols.ws_flag * AB::F::from_u32(FpOpcode::FCVT_S_W as u32)
            + cols.wu_flag * AB::F::from_u32(FpOpcode::FCVT_S_WU as u32)
            + AB::F::from_usize(self.offset);

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

impl FcvtI2FCoreAir {
    /// Constrain the int→float conversion (RNE) for nonzero inputs.
    ///
    /// The magnitude `m = |integer|` is placed at `x = m << 24` (7 bytes); the
    /// result significand is `sig = x >> (r_exp - 126)`, which lands the leading
    /// bit at position 23 (the implicit bit) — the shift is forced because the
    /// high output bytes must be zero and bit 23 must be set. RNE rounding uses
    /// the ShrnGadget's guard/sticky plus the mantissa LSB. Zero input ⇒ +0
    /// (handled elsewhere).
    fn eval_convert<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FcvtI2FCoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f126: AB::F = AB::F::from_u32(126);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f24: AB::F = AB::F::from_u32(24);
        let f255: AB::F = AB::F::from_u32(255);
        let f256: AB::F = AB::F::from_u32(1 << SHLIMB_BITS);
        let one = AB::Expr::ONE;

        let r_exp = cols.r_exp_lo + cols.r_hi * f2;

        // s_convert = !a_is_zero (the normalize/round path runs on nonzero input).
        builder
            .when(cols.is_valid)
            .assert_eq(cols.s_convert, one.clone() - cols.a_is_zero);
        builder.assert_bool(cols.s_convert);
        let s = cols.s_convert;

        // ── Magnitude: m = a, except .w negative ⇒ m = -a (two's complement) ──
        let negate_input = cols.ws_flag * cols.a_sign;
        for i in 0..4 {
            builder.assert_bool(cols.neg_carry[i]);
        }
        // byte 0: mag[0] = a[0]            when !negate
        //         mag[0] = (255-a[0])+1 - neg_carry[0]*256  when negate
        builder
            .when(s * (one.clone() - negate_input.clone()))
            .assert_eq(cols.mag[0], cols.a[0]);
        builder.when(s * negate_input.clone()).assert_eq(
            cols.mag[0],
            (AB::Expr::from(f255) - cols.a[0]) + AB::Expr::ONE - cols.neg_carry[0] * f256,
        );
        for i in 1..4 {
            builder
                .when(s * (one.clone() - negate_input.clone()))
                .assert_eq(cols.mag[i], cols.a[i]);
            builder.when(s * negate_input.clone()).assert_eq(
                cols.mag[i],
                (AB::Expr::from(f255) - cols.a[i]) + cols.neg_carry[i - 1]
                    - cols.neg_carry[i] * f256,
            );
        }

        // x = mag << 24 = [0, 0, 0, mag[0], mag[1], mag[2], mag[3]] (7 bytes).
        builder.when(s).assert_zero(cols.shrn.x[0]);
        builder.when(s).assert_zero(cols.shrn.x[1]);
        builder.when(s).assert_zero(cols.shrn.x[2]);
        builder.when(s).assert_eq(cols.shrn.x[3], cols.mag[0]);
        builder.when(s).assert_eq(cols.shrn.x[4], cols.mag[1]);
        builder.when(s).assert_eq(cols.shrn.x[5], cols.mag[2]);
        builder.when(s).assert_eq(cols.shrn.x[6], cols.mag[3]);

        // Right shift by (r_exp - 126 - rc2) = un-rounded exponent - 126: the shift
        // that lands the leading bit at position 23. (r_exp = shift + 126 + rc2 is
        // constrained below; rc2 is the rounding carry into the implicit bit.)
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.shrn);
        builder.when(s).assert_eq(cols.shrn.shift, cols.shift);
        builder.assert_eq(cols.shrn.enable, s);

        // Bits above the 24-bit significand must be zero (forces shift large enough).
        for i in 3..7 {
            builder.when(s).assert_zero(cols.shrn.y[i]);
        }
        // Implicit bit at position 23 (forces shift small enough): shrn.y[2] >= 128,
        // i.e. (shrn.y[2] - 128) is a 7-bit value used in the carry chain below.

        // ── RNE rounding ────────────────────────────────────────────────
        // lsb = bit 0 of shrn.y[0] (mantissa LSB); guard/sticky from the gadget.
        builder
            .when(s)
            .assert_eq(cols.shrn.y[0], cols.y0_high7 * f2 + cols.lsb);
        builder.assert_bool(cols.lsb);
        self.range_bus
            .range_check(cols.y0_high7.into(), 7)
            .eval(builder, s);
        let sticky_or_lsb = cols.shrn.sticky + cols.lsb - cols.shrn.sticky * cols.lsb;
        builder.assert_bool(cols.shrn.guard);
        builder.assert_bool(cols.shrn.sticky);
        builder
            .when(s)
            .assert_eq(cols.round_up, cols.shrn.guard * sticky_or_lsb);
        builder.assert_bool(cols.round_up);

        // ── Output reconstruction ───────────────────────────────────────
        // result mantissa = shrn.y[0..2] ++ (shrn.y[2]&0x7F), rounded up by round_up.
        builder.assert_bool(cols.rc0);
        builder.assert_bool(cols.rc1);
        builder.assert_bool(cols.rc2);
        builder.when(s).assert_eq(
            cols.result[0] + cols.rc0 * f256,
            cols.shrn.y[0] + cols.round_up,
        );
        builder
            .when(s)
            .assert_eq(cols.result[1] + cols.rc1 * f256, cols.shrn.y[1] + cols.rc0);
        // top7c = (shrn.y[2] - 128) + rc1 = rc2_rem + rc2*128.
        // The implicit bit must be present: enforce y[2] >= 128 directly
        // (otherwise shift = honest+1 slips through when rc1 = 1).
        builder
            .when(s)
            .assert_eq(cols.shrn.y[2], cols.y2_lo7 + f128);
        self.range_bus
            .range_check(cols.y2_lo7.into(), 7)
            .eval(builder, s);
        builder
            .when(s)
            .assert_eq(cols.y2_lo7 + cols.rc1, cols.rc2_rem + cols.rc2 * f128);
        self.range_bus
            .range_check(cols.rc2_rem.into(), 7)
            .eval(builder, s);
        builder
            .when(s)
            .assert_eq(cols.result[2], cols.rc2_rem + cols.r_exp_lo * f128);
        builder
            .when(s)
            .assert_eq(cols.result[3], cols.r_hi + cols.r_sign * f128);
        // r_exp = shift + 126 + rc2 (exponent gains the rounding carry into the
        // implicit bit). The result is never inf (r_exp != 255 is constrained
        // elsewhere); rc2 ∈ {0,1} and r_exp <= 159 < 255 here.
        builder
            .when(s)
            .assert_eq(r_exp.clone(), cols.shift + f126 + cols.rc2);
        // range-check the magnitude bytes (bounds the integer input).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.mag[i].into(), 8)
                .eval(builder, s);
        }
        let _ = (f127, f24);
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FcvtI2FCoreRecord {
    pub a: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FcvtI2FExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub read_as: u32,
    pub write_as: u32,
}

pub(crate) fn opcode_is_wu(opcode: FpOpcode) -> Option<bool> {
    match opcode {
        FpOpcode::FCVT_S_W => Some(false),
        FpOpcode::FCVT_S_WU => Some(true),
        _ => None,
    }
}

/// Result of fcvt.s.w (is_wu = false) or fcvt.s.wu (is_wu = true).
#[inline(always)]
pub fn fcvt_i2f_result(x: u32, is_wu: bool) -> u32 {
    if is_wu {
        fcvt_s_wu(x)
    } else {
        fcvt_s_w(x)
    }
}

impl<A, F, RA> PreflightExecutor<F, RA> for FcvtI2FExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FcvtI2FCoreRecord),
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

        let [a] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;
        core_record.opcode = local as u8;

        let is_wu = opcode_is_wu(FpOpcode::from_usize(local)).unwrap();
        let result = fcvt_i2f_result(u32::from_le_bytes(a), is_wu);
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

pub struct FcvtI2FFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FcvtI2FFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FcvtI2FFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FcvtI2FCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FcvtI2FCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let is_wu = opcode_is_wu(opcode).unwrap();
        let result = fcvt_i2f_result(a_bits, is_wu);

        let r_sign = (result >> 31) & 1;
        let r_exp = (result >> 23) & 0xFF;
        let r_mant = result & 0x7FFFFF;

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((result >> (i * 8)) & 0xFF, 8);
        }
        core_row.a_sign = F::from_u32((a_bits >> 31) & 1);
        core_row.a_hi = F::from_u32(((a_bits >> 24) & 0xFF) & 0x7F);
        self.range_checker
            .add_count(((a_bits >> 24) & 0xFF) & 0x7F, 7);
        let a_sum: u32 = (a_bits & 0xFF)
            + ((a_bits >> 8) & 0xFF)
            + ((a_bits >> 16) & 0xFF)
            + ((a_bits >> 24) & 0xFF);
        core_row.a_is_zero = F::from_bool(a_bits == 0);
        core_row.a_sum_inv = if a_bits != 0 {
            F::from_u32(a_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };

        core_row.r_sign = F::from_u32(r_sign);
        core_row.r_exp_lo = F::from_u32(r_exp & 1);
        core_row.r_mant2 = F::from_u32((r_mant >> 16) & 0x7F);
        core_row.r_hi = F::from_u32(r_exp >> 1);

        let r_exp_f = F::from_u32(r_exp);
        core_row.r_exp_is_zero = F::from_bool(r_exp == 0);
        core_row.r_exp_zero_inv = if r_exp != 0 {
            r_exp_f.try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        // result is never NaN/inf: r_exp != 255 always holds
        core_row.r_exp_max_inv = (F::from_u32(255) - r_exp_f)
            .try_inverse()
            .unwrap_or(F::ZERO);
        let r_mant_sum = (r_mant & 0xFF) + ((r_mant >> 8) & 0xFF) + ((r_mant >> 16) & 0x7F);
        core_row.r_mant_is_zero = F::from_bool(r_mant == 0);
        core_row.r_mant_sum_inv = if r_mant != 0 {
            F::from_u32(r_mant_sum).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };
        core_row.r_is_zero = F::from_bool(r_exp == 0 && r_mant == 0);

        core_row.ws_flag = F::from_bool(!is_wu);
        core_row.wu_flag = F::from_bool(is_wu);

        // ── Conversion witness ───────────────────────────────────────────
        let s_convert = a_bits != 0;
        core_row.s_convert = F::from_bool(s_convert);
        let a_sign = (a_bits >> 31) & 1;
        let negate_input = !is_wu && a_sign == 1;
        // magnitude m = |integer|
        let m: u32 = if negate_input {
            (!a_bits).wrapping_add(1)
        } else {
            a_bits
        };
        for i in 0..4 {
            core_row.mag[i] = F::from_u32((m >> (i * 8)) & 0xFF);
        }
        // Recover the negation borrow chain (only meaningful when negate_input).
        if negate_input {
            let mut carry_in = 1u32;
            for i in 0..4 {
                let lhs = (255 - ((a_bits >> (i * 8)) & 0xFF)) + carry_in;
                core_row.neg_carry[i] = F::from_u32(lhs >> 8);
                carry_in = lhs >> 8;
            }
        } else {
            for i in 0..4 {
                core_row.neg_carry[i] = F::ZERO;
            }
        }

        if s_convert {
            // x = m << 24 (7 bytes), sig = x >> shift, shift = r_exp - 126.
            for i in 0..4 {
                self.range_checker.add_count((m >> (i * 8)) & 0xFF, 8);
            }
            let x: [u32; 7] = [
                0,
                0,
                0,
                m & 0xFF,
                (m >> 8) & 0xFF,
                (m >> 16) & 0xFF,
                (m >> 24) & 0xFF,
            ];
            // r_exp from the native result; the constraint ties shift = r_exp - 126.
            // Recover the un-rounded exponent (rc2 may bump it by 1).
            let p = 31 - m.leading_zeros(); // top bit position
            let e_unrounded = p + 127;
            core_row.shift = F::from_u32(e_unrounded - 126);
            core_row.shrn =
                make_shrn_cols7::<F>(x, (e_unrounded - 126) as usize, &self.range_checker);
            let y0 = core_row.shrn.y[0].as_canonical_u32();
            core_row.y0_high7 = F::from_u32(y0 >> 1);
            core_row.lsb = F::from_u32(y0 & 1);
            self.range_checker.add_count(y0 >> 1, 7);
            let guard = core_row.shrn.guard.as_canonical_u32();
            let sticky = core_row.shrn.sticky.as_canonical_u32() != 0;
            let round_up = guard == 1 && (sticky || (y0 & 1) == 1);
            core_row.round_up = F::from_bool(round_up);
            // Mantissa rounding carry chain over result bytes.
            let s0 = ((result) & 0xFF) as u32; // result byte 0
            let _ = s0;
            // Reconstruct carries from the native result vs the un-rounded sig.
            let y0v = core_row.shrn.y[0].as_canonical_u32();
            let y1v = core_row.shrn.y[1].as_canonical_u32();
            let y2v = core_row.shrn.y[2].as_canonical_u32();
            let sum0 = y0v + round_up as u32;
            let rc0 = sum0 >> 8;
            let sum1 = y1v + rc0;
            let rc1 = sum1 >> 8;
            let top7c = (y2v - 128) + rc1;
            let rc2 = top7c >> 7;
            let rc2_rem = top7c & 0x7F;
            core_row.rc0 = F::from_u32(rc0);
            core_row.rc1 = F::from_u32(rc1);
            core_row.rc2 = F::from_u32(rc2);
            core_row.rc2_rem = F::from_u32(rc2_rem);
            core_row.y2_lo7 = F::from_u32(y2v - 128);
            self.range_checker.add_count(rc2_rem, 7);
            self.range_checker.add_count(y2v - 128, 7);
            // Sanity: the reconstructed exponent must match the native result.
            debug_assert_eq!(
                (e_unrounded - 126) + 126 + rc2,
                r_exp,
                "fcvt_i2f exp mismatch: a={:08x}",
                a_bits
            );
            debug_assert_eq!((y2v - 128) + rc1, rc2_rem + rc2 * 128);
        } else {
            core_row.shift = F::ZERO;
            core_row.shrn = ShrnCols::<F, 7>::default();
            core_row.y0_high7 = F::ZERO;
            core_row.lsb = F::ZERO;
            core_row.round_up = F::ZERO;
            core_row.rc0 = F::ZERO;
            core_row.rc1 = F::ZERO;
            core_row.rc2 = F::ZERO;
            core_row.rc2_rem = F::ZERO;
            core_row.y2_lo7 = F::ZERO;
        }

        core_row.is_valid = F::ONE;
    }
}

/// Build a `ShrnCols<7>` witness and register range-checker counts for ShrnGadget.
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
