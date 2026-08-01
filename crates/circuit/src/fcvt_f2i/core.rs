use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    is_less_than::{IsLessThanIo, IsLtSubAir, LessThanAuxCols},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
    TraceSubRowGenerator,
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

use crate::fp_shift::{run_shrn, ShrnCols, ShrnGadget, SHLIMB_BITS};

/// Number of `lt_decomp` limbs for the 8-bit exponent comparison.
/// Warning: requires `8.div_ceil(range_max_bits) = LT_AUX_LEN`.
pub const LT_AUX_LEN: usize = 1;

/// IEEE 754 / RISC-V FCVT.W.S: convert f32 to i32 with truncation (RTZ).
/// This matches C's `(int)float` semantics. The compiler always emits RTZ
/// for this instruction.
/// NaN -> i32::MAX; out of range saturates to i32::MIN / i32::MAX.
#[inline(always)]
pub fn fcvt_w_s(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        return i32::MAX as u32;
    }
    let r = f.trunc();
    if r >= 2147483648.0 {
        i32::MAX as u32
    } else if r < -2147483648.0 {
        i32::MIN as u32
    } else {
        (r as i32) as u32
    }
}

/// IEEE 754 / RISC-V FCVT.WU.S: convert f32 to u32 with truncation (RTZ).
/// NaN -> u32::MAX; negative -> 0; out of range saturates to u32::MAX.
#[inline(always)]
pub fn fcvt_wu_s(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        return u32::MAX;
    }
    let r = f.trunc();
    if r >= 4294967296.0 {
        u32::MAX
    } else if r <= 0.0 {
        0
    } else {
        r as u32
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FcvtF2ICoreCols<T> {
    pub a: [T; 4],

    // === Decomposition of input a ===
    pub a_sign: T,
    pub a_exp_lo: T,
    pub a_mant2: T,
    pub a_hi: T,

    // === Classification ===
    pub a_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub a_exp_is_max: T,
    pub a_mant_is_zero: T,
    pub a_nan: T,
    pub a_inf: T,

    // === Saturation: sat = (exp >= 158) for .w, (exp >= 159) for .wu ===
    pub lt_aux: LessThanAuxCols<T, LT_AUX_LEN>,
    pub lt_158: T, // exp < 158
    pub eq_158_inv: T,
    pub eq_158: T, // exp == 158
    pub sat: T,

    // === Result selection ===
    pub special: T,     // nan | inf | sat
    pub inf_sat: T,     // (inf | sat) && !nan
    pub neg_zero_wu: T, // wu_flag && a_sign (negative inputs convert to 0 for .wu)
    pub zero_res: T,    // neg_zero_wu && !special
    pub sel_wit: T,     // !special && !neg_zero_wu
    pub wit: [T; 4],
    pub ws_flag: T,
    pub wu_flag: T,

    // === In-range truncation (RTZ) ===
    pub lt_aux2: LessThanAuxCols<T, LT_AUX_LEN>, // for lt_127 = (exp < 127)
    pub lt_127: T,
    pub s_shift: T, // sel_wit AND exp >= 127 (the truncating shift runs)
    pub shift: T,
    pub shrn: ShrnCols<T, 4>, // magnitude = (sig << 8) >> (158 - exp)
    pub negate: T,            // ws_flag AND a_sign
    pub neg_carry: [T; 4],    // two's-complement borrow chain (wit = -magnitude)

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FcvtF2ICoreCols<u8>)]
pub struct FcvtF2ICoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FcvtF2ICoreAir {
    fn width(&self) -> usize {
        FcvtF2ICoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FcvtF2ICoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FcvtF2ICoreAir
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
        let cols: &FcvtF2ICoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f127: AB::F = AB::F::from_u32(127);
        let f158: AB::F = AB::F::from_u32(158);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.a_exp_is_max);
        builder.assert_bool(cols.a_mant_is_zero);
        builder.assert_bool(cols.a_nan);
        builder.assert_bool(cols.a_inf);
        builder.assert_bool(cols.eq_158);
        builder.assert_bool(cols.sat);
        builder.assert_bool(cols.special);
        builder.assert_bool(cols.inf_sat);
        builder.assert_bool(cols.neg_zero_wu);
        builder.assert_bool(cols.zero_res);
        builder.assert_bool(cols.sel_wit);
        builder.assert_bool(cols.ws_flag);
        builder.assert_bool(cols.wu_flag);

        // Exactly one opcode flag when valid
        builder
            .when(cols.is_valid)
            .assert_one(cols.ws_flag + cols.wu_flag);

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

        // Range check input bytes (bounds exp and mantissa sums) and the
        // in-range result witness.
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
                .eval(builder, cols.is_valid.into());
            self.range_bus
                .range_check(cols.wit[i].into(), 8)
                .eval(builder, cols.is_valid.into());
        }

        // ── Classification ───────────────────────────────────────────────
        let a_exp = cols.a_exp_lo + cols.a_hi * f2;
        let a_exp_max_delta = AB::Expr::from(f255) - a_exp.clone();
        builder.when(cols.is_valid).assert_eq(
            cols.a_exp_is_max,
            one.clone() - a_exp_max_delta.clone() * cols.a_exp_max_inv,
        );
        builder.assert_zero(a_exp_max_delta * cols.a_exp_is_max);
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

        // ── Saturation ───────────────────────────────────────────────────
        // lt_158 = (exp < 158) via the range-checked less-than gadget.
        let lt_air = IsLtSubAir::new(self.range_bus, 8);
        let io = IsLessThanIo::new(
            a_exp.clone(),
            AB::Expr::from(f158),
            cols.lt_158.into(),
            cols.is_valid.into(),
        );
        lt_air.eval(builder, (io, &cols.lt_aux.lower_decomp));
        // eq_158 = (exp == 158)
        let exp_158_delta = a_exp - AB::Expr::from(f158);
        builder.when(cols.is_valid).assert_eq(
            cols.eq_158,
            one.clone() - exp_158_delta.clone() * cols.eq_158_inv,
        );
        builder.assert_zero(exp_158_delta * cols.eq_158);
        // .w saturates when exp >= 158; .wu saturates when exp >= 159.
        // For .wu, exp == 158 is still in range (values < 2^32).
        builder.when(cols.is_valid).assert_eq(
            cols.sat,
            one.clone() - cols.lt_158 - cols.wu_flag * cols.eq_158,
        );

        // ── Result selection ─────────────────────────────────────────────
        builder.assert_eq(
            cols.special,
            one.clone()
                - (one.clone() - cols.a_nan)
                    * (one.clone() - cols.a_inf)
                    * (one.clone() - cols.sat),
        );
        builder.assert_eq(
            cols.inf_sat,
            (one.clone() - cols.a_nan)
                * (one.clone() - (one.clone() - cols.a_inf) * (one.clone() - cols.sat)),
        );
        builder.assert_eq(cols.neg_zero_wu, cols.wu_flag * cols.a_sign);
        builder.assert_eq(
            cols.zero_res,
            cols.neg_zero_wu * (one.clone() - cols.special),
        );
        builder.assert_eq(
            cols.sel_wit,
            (one.clone() - cols.special)
                * (one.clone() - cols.neg_zero_wu)
                * (cols.ws_flag + cols.wu_flag),
        );

        // max = i32::MAX (.w) or u32::MAX (.wu); min = i32::MIN (.w) or 0 (.wu)
        let max3: AB::Expr = AB::Expr::from(f127) + cols.wu_flag * AB::F::from_u32(128);
        let min3: AB::Expr = (one.clone() - cols.wu_flag) * AB::F::from_u32(128);
        for i in 0..4 {
            let (max_i, min_i) = if i == 3 {
                (max3.clone(), min3.clone())
            } else {
                (AB::Expr::from(f255), AB::Expr::ZERO)
            };
            // NaN converts to max.
            builder
                .when(cols.a_nan)
                .assert_eq(cols.result[i], max_i.clone());
            // Inf / saturation: sign ? min : max.
            builder.when(cols.inf_sat).assert_eq(
                cols.result[i],
                cols.a_sign * min_i + (one.clone() - cols.a_sign) * max_i,
            );
            // Negative inputs convert to 0 for .wu.
            builder.when(cols.zero_res).assert_zero(cols.result[i]);
            // In-range: wit is constrained to the RTZ-truncated integer by
            // eval_in_range; result mirrors wit.
            builder
                .when(cols.sel_wit)
                .assert_eq(cols.result[i], cols.wit[i]);
        }

        // ── In-range truncation (RTZ): wit = trunc(value) ────────────────
        self.eval_in_range(builder, cols);

        let expected_opcode = cols.ws_flag * AB::F::from_u32(FpOpcode::FCVT_W_S as u32)
            + cols.wu_flag * AB::F::from_u32(FpOpcode::FCVT_WU_S as u32)
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

impl FcvtF2ICoreAir {
    /// Constrain the in-range integer witness `wit` to equal `trunc(value)` (RTZ).
    ///
    /// The magnitude is `(sig << 7) >> (157 - exp)` where `sig = mant | 0x800000`
    /// (the input is normal for `exp >= 127`, which is the only case where the
    /// magnitude is nonzero). For `.w` with `a_sign`, `wit` is the two's-complement
    /// negation of the magnitude; for `.wu` (and positive `.w`) `wit` equals it.
    /// All other `sel_wit` rows (subnormal/zero input, or `exp < 127` ⇒ value < 1)
    /// have `wit = 0`.
    fn eval_in_range<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &FcvtF2ICoreCols<AB::Var>,
    ) {
        let f2: AB::F = AB::F::from_u32(2);
        let f127: AB::F = AB::F::from_u32(127);
        let f128: AB::F = AB::F::from_u32(128);
        let f158: AB::F = AB::F::from_u32(158);
        let f255: AB::F = AB::F::from_u32(255);
        let f256: AB::F = AB::F::from_u32(1 << SHLIMB_BITS);
        let one = AB::Expr::ONE;

        let a_exp = cols.a_exp_lo + cols.a_hi * f2;

        // lt_127 = (a_exp < 127).
        let lt_air = IsLtSubAir::new(self.range_bus, 8);
        let io = IsLessThanIo::new(
            a_exp.clone(),
            AB::Expr::from(f127),
            cols.lt_127.into(),
            cols.is_valid.into(),
        );
        lt_air.eval(builder, (io, &cols.lt_aux2.lower_decomp));
        builder.assert_bool(cols.lt_127);

        // s_shift = sel_wit AND (exp >= 127). For these rows the input is normal
        // (exp in [127, 158] ⊂ [1, 254], since sat excludes exp >= 158 for .w and
        // exp >= 159 for .wu).
        builder.assert_eq(cols.s_shift, cols.sel_wit * (one.clone() - cols.lt_127));
        builder.assert_bool(cols.s_shift);
        let s_shift = cols.s_shift;

        // x = sig << 8 = [0, a[0], a[1], a_mant2 + 128] (sig = mant | 0x800000).
        // Only constrained when the shift runs; off rows zero-fill the gadget.
        let sig2 = cols.a_mant2 + f128;
        builder.when(s_shift).assert_zero(cols.shrn.x[0]);
        builder.when(s_shift).assert_eq(cols.shrn.x[1], cols.a[0]);
        builder.when(s_shift).assert_eq(cols.shrn.x[2], cols.a[1]);
        builder.when(s_shift).assert_eq(cols.shrn.x[3], sig2);

        // Truncating right shift: magnitude = x >> (158 - exp), shift in [0, 31].
        builder
            .when(s_shift)
            .assert_eq(cols.shift, AB::Expr::from(f158) - a_exp.clone());
        let shrn = ShrnGadget::new(self.range_bus);
        shrn.eval_shrn(builder, &cols.shrn);
        builder.when(s_shift).assert_eq(cols.shrn.shift, cols.shift);
        builder.assert_eq(cols.shrn.enable, s_shift);

        // negate = ws AND a_sign (two's-complement for negative .w).
        builder.assert_eq(cols.negate, cols.ws_flag * cols.a_sign);
        builder.assert_bool(cols.negate);
        for i in 0..4 {
            builder.assert_bool(cols.neg_carry[i]);
        }

        // sel_wit rows where the shift does NOT run ⇒ wit = 0 (value < 1).
        for i in 0..4 {
            builder
                .when(cols.sel_wit * (one.clone() - s_shift))
                .assert_zero(cols.wit[i]);
        }

        // s_shift, !negate: wit = magnitude (shrn.y).
        for i in 0..4 {
            builder
                .when(s_shift * (one.clone() - cols.negate))
                .assert_eq(cols.wit[i], cols.shrn.y[i]);
        }

        // s_shift, negate: wit = -magnitude = ~magnitude + 1 (byte borrow chain).
        builder.when(s_shift * cols.negate).assert_eq(
            cols.wit[0],
            (AB::Expr::from(f255) - cols.shrn.y[0]) + AB::Expr::ONE - cols.neg_carry[0] * f256,
        );
        for i in 1..4 {
            builder.when(s_shift * cols.negate).assert_eq(
                cols.wit[i],
                (AB::Expr::from(f255) - cols.shrn.y[i]) + cols.neg_carry[i - 1]
                    - cols.neg_carry[i] * f256,
            );
        }
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct FcvtF2ICoreRecord {
    pub a: [u8; 4],
    pub opcode: u8,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FcvtF2IExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub read_as: u32,
    pub write_as: u32,
}

pub(crate) fn opcode_is_wu(opcode: FpOpcode) -> Option<bool> {
    match opcode {
        FpOpcode::FCVT_W_S => Some(false),
        FpOpcode::FCVT_WU_S => Some(true),
        _ => None,
    }
}

/// Result of fcvt.w.s (is_wu = false) or fcvt.wu.s (is_wu = true).
#[inline(always)]
pub fn fcvt_f2i_result(bits: u32, is_wu: bool) -> u32 {
    if is_wu {
        fcvt_wu_s(bits)
    } else {
        fcvt_w_s(bits)
    }
}

impl<A, F, RA> PreflightExecutor<F, RA> for FcvtF2IExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FcvtF2ICoreRecord),
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
        let result = fcvt_f2i_result(u32::from_le_bytes(a), is_wu);
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

pub struct FcvtF2IFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub lt_air: IsLtSubAir,
}

impl<A> FcvtF2IFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), 8);
        Self {
            adapter,
            offset,
            range_checker,
            lt_air,
        }
    }
}

impl<A, F> TraceFiller<F> for FcvtF2IFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FcvtF2ICoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FcvtF2ICoreCols<F> = core_row.borrow_mut();
        let a_bits = u32::from_le_bytes(record.a);
        let opcode = FpOpcode::from_usize(record.opcode as usize);
        let is_wu = opcode_is_wu(opcode).unwrap();
        fill_fcvt_f2i_witness(core_row, a_bits, is_wu, &self.range_checker, &self.lt_air);
    }
}

/// Builds a full honest witness row for fcvt.w.s / fcvt.wu.s, mirroring the
/// production filler's core-filling logic.
fn fill_fcvt_f2i_witness<F: PrimeField32>(
    core_row: &mut FcvtF2ICoreCols<F>,
    a_bits: u32,
    is_wu: bool,
    range_checker: &SharedVariableRangeCheckerChip,
    lt_air: &IsLtSubAir,
) {
    let result = fcvt_f2i_result(a_bits, is_wu);

    let a_sign = (a_bits >> 31) & 1;
    let a_exp = (a_bits >> 23) & 0xFF;
    let a_mant = a_bits & 0x7FFFFF;

    for i in 0..4 {
        core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
        core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
        range_checker.add_count((result >> (i * 8)) & 0xFF, 8);
    }
    core_row.a_sign = F::from_u32(a_sign);
    core_row.a_exp_lo = F::from_u32(a_exp & 1);
    core_row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
    core_row.a_hi = F::from_u32(a_exp >> 1);
    range_checker.add_count((a_mant >> 16) & 0x7F, 7);
    range_checker.add_count(a_exp >> 1, 7);

    let a_exp_f = F::from_u32(a_exp);
    core_row.a_exp_is_max = F::from_bool(a_exp == 255);
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
    core_row.a_nan = F::from_bool(a_nan);
    core_row.a_inf = F::from_bool(a_inf);

    // Saturation
    lt_air.generate_subrow(
        (range_checker.as_ref(), a_exp, 158),
        (&mut core_row.lt_aux.lower_decomp, &mut core_row.lt_158),
    );
    let lt_158 = a_exp < 158;
    let eq_158 = a_exp == 158;
    core_row.eq_158 = F::from_bool(eq_158);
    core_row.eq_158_inv = if !eq_158 {
        (a_exp_f - F::from_u32(158))
            .try_inverse()
            .unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    let sat = !lt_158 && (!is_wu || !eq_158);
    core_row.sat = F::from_bool(sat);

    let special = a_nan || a_inf || sat;
    let inf_sat = !a_nan && (a_inf || sat);
    let neg_zero_wu = is_wu && a_sign == 1;
    let zero_res = neg_zero_wu && !special;
    let sel_wit = !special && !neg_zero_wu;
    core_row.special = F::from_bool(special);
    core_row.inf_sat = F::from_bool(inf_sat);
    core_row.neg_zero_wu = F::from_bool(neg_zero_wu);
    core_row.zero_res = F::from_bool(zero_res);
    core_row.sel_wit = F::from_bool(sel_wit);
    for i in 0..4 {
        core_row.wit[i] = core_row.result[i];
    }
    core_row.ws_flag = F::from_bool(!is_wu);
    core_row.wu_flag = F::from_bool(is_wu);

    // ── In-range truncation witness ──────────────────────────────────
    let lt_127 = a_exp < 127;
    lt_air.generate_subrow(
        (range_checker.as_ref(), a_exp, 127),
        (&mut core_row.lt_aux2.lower_decomp, &mut core_row.lt_127),
    );
    core_row.lt_127 = F::from_bool(lt_127);
    let s_shift = sel_wit && !lt_127;
    core_row.s_shift = F::from_bool(s_shift);
    core_row.negate = F::from_bool(!is_wu && a_sign == 1);

    if s_shift {
        // x = sig << 8 = [0, a[0], a[1], a_mant2 + 128]; magnitude = x >> (158-exp).
        let sig: u32 = (a_mant | 0x800000) & 0xFFFFFF;
        let x_val = sig << 8;
        let x: [u32; 4] = [
            x_val & 0xFF,
            (x_val >> 8) & 0xFF,
            (x_val >> 16) & 0xFF,
            (x_val >> 24) & 0xFF,
        ];
        let shift = 158 - a_exp;
        core_row.shift = F::from_u32(shift);
        core_row.shrn = make_shrn_cols::<F>(x, shift as usize, range_checker);
        let mag = (sig as u64) << 8 >> shift;
        let mag_bytes: [u32; 4] = core::array::from_fn(|i| ((mag >> (i * 8)) & 0xFF) as u32);
        if !is_wu && a_sign == 1 {
            let neg = (0u64).wrapping_sub(mag);
            let neg_bytes: [u32; 4] = core::array::from_fn(|i| ((neg >> (i * 8)) & 0xFF) as u32);
            let mut carry_in = 1u32; // +1 of two's complement
            for i in 0..4 {
                let lhs = (255 - mag_bytes[i]) + carry_in;
                core_row.neg_carry[i] = F::from_u32(lhs >> 8);
                carry_in = lhs >> 8;
                debug_assert_eq!(neg_bytes[i], core_row.wit[i].as_canonical_u32());
            }
        } else {
            for (i, &mb) in mag_bytes.iter().enumerate() {
                core_row.neg_carry[i] = F::ZERO;
                debug_assert_eq!(mb, core_row.wit[i].as_canonical_u32());
            }
        }
    } else {
        core_row.shift = F::ZERO;
        core_row.shrn = ShrnCols::<F, 4>::default();
        for i in 0..4 {
            core_row.neg_carry[i] = F::ZERO;
        }
    }

    core_row.is_valid = F::ONE;
}

/// Build a `ShrnCols<4>` witness and register range-checker counts for ShrnGadget.
fn make_shrn_cols<F: PrimeField32>(
    x: [u32; 4],
    shift: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) -> ShrnCols<F, 4> {
    let shift = shift.min(4 * SHLIMB_BITS - 1);
    let (y, q, r) = run_shrn(&x, shift);
    let (guard, sticky) = crate::fp_shift::run_shrn_round(&x, shift);
    let mut limb_shift_marker = [F::ZERO; 4];
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

    for i in 0..4 {
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
mod tests {
    use std::borrow::BorrowMut;
    use std::sync::Arc;

    use openvm_circuit::arch::{BasicAdapterInterface, MinimalInstruction, VmCoreAir};
    use openvm_circuit_primitives::is_less_than::IsLtSubAir;
    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_circuit_primitives::StructReflectionHelper;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::test_utils::NumericBuilder;

    type F = BabyBear;
    type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 1, 1, 4, 4>;

    fn row(a_bits: u32, is_wu: bool) -> Vec<F> {
        let chip = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            0, 16,
        )));
        let lt_air = IsLtSubAir::new(chip.bus(), 8);
        let mut buf = vec![F::ZERO; FcvtF2ICoreCols::<F>::width()];
        let row: &mut FcvtF2ICoreCols<F> = buf.as_mut_slice().borrow_mut();
        fill_fcvt_f2i_witness(row, a_bits, is_wu, &chip, &lt_air);
        buf
    }

    fn eval(buf: &[F]) -> NumericBuilder<F> {
        let air = FcvtF2ICoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET);
        let mut b = NumericBuilder::new(vec![]);
        VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
        b
    }

    #[test]
    fn honest_rows_pass() {
        for (bits, is_wu) in [
            (3.7f32.to_bits(), false),
            ((-3.7f32).to_bits(), false),
            (5e9f32.to_bits(), true), // saturates for .w, in-range for .wu
            ((-3.0f32).to_bits(), true), // negative -> 0 for .wu
            (0x7FC00000, false),      // NaN -> i32::MAX
            (0x7F800000, false),      // +inf
            (0x00000001, false),      // subnormal -> 0
            (0x4F7FFFFF, true),       // ~2^32-128, .wu in-range
        ] {
            for is_wu in [is_wu, !is_wu] {
                let buf = row(bits, is_wu);
                assert!(
                    eval(&buf).ok(),
                    "fcvt_f2i a={bits:08x} wu={is_wu}: {:?}",
                    eval(&buf).failures
                );
            }
        }
    }

    #[test]
    fn zero_padding_row_passes() {
        let mut buf = vec![F::ZERO; FcvtF2ICoreCols::<F>::width()];
        let row: &mut FcvtF2ICoreCols<F> = buf.as_mut_slice().borrow_mut();
        row.is_valid = F::ZERO;
        assert!(eval(&buf).ok());
    }

    #[test]
    fn corrupt_result_byte_rejected() {
        let mut buf = row(3.7f32.to_bits(), false);
        let row: &mut FcvtF2ICoreCols<F> = buf.as_mut_slice().borrow_mut();
        row.result[0] += F::ONE;
        assert!(!eval(&buf).ok(), "corrupting a result byte must fail");
    }

    /// Result bytes must be pinned across input classes.
    #[test]
    fn corruption_sweep_pins_result() {
        use crate::test_utils::assert_columns_constrained;
        let names = FcvtF2ICoreCols::<F>::struct_reflection().unwrap_or_default();
        let eval_full = |b: &mut NumericBuilder<F>, cols: &[F]| {
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(
                &FcvtF2ICoreAir::new(VariableRangeCheckerBus::new(0, 16), FpOpcode::CLASS_OFFSET),
                b,
                cols,
                F::ZERO,
            );
        };
        for (bits, is_wu) in [
            (3.7f32.to_bits(), false),
            ((-3.7f32).to_bits(), false),
            (5e9f32.to_bits(), true),
            (0x7FC00000, false),
            (0x00000001, false),
        ] {
            let buf = row(bits, is_wu);
            assert!(eval(&buf).ok(), "honest fcvt_f2i failed");
            assert_columns_constrained(
                &buf,
                eval_full,
                &names,
                &[
                    "result__0",
                    "result__1",
                    "result__2",
                    "result__3",
                    "sel_wit",
                ],
                &format!("fcvt_f2i a={bits:08x} wu={is_wu}"),
            );
        }
    }
}
