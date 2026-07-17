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

/// Number of `lt_decomp` limbs for the 8-bit exponent comparison.
/// Warning: requires `8.div_ceil(range_max_bits) = LT_AUX_LEN`.
pub const LT_AUX_LEN: usize = 1;

/// IEEE 754 / RISC-V FCVT.W.S: convert f32 to i32 with RNE rounding.
/// NaN -> i32::MAX; out of range saturates to i32::MIN / i32::MAX.
#[inline(always)]
pub fn fcvt_w_s(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        return i32::MAX as u32;
    }
    let r = f.round_ties_even();
    if r >= 2147483648.0 {
        i32::MAX as u32
    } else if r < -2147483648.0 {
        i32::MIN as u32
    } else {
        (r as i32) as u32
    }
}

/// IEEE 754 / RISC-V FCVT.WU.S: convert f32 to u32 with RNE rounding.
/// NaN -> u32::MAX; negative -> 0; out of range saturates to u32::MAX.
#[inline(always)]
pub fn fcvt_wu_s(bits: u32) -> u32 {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        return u32::MAX;
    }
    let r = f.round_ties_even();
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
            // In-range: the rounded integer witness.
            // TODO: verify the RNE rounding relation between a and wit
            // (needs shift/round machinery similar to fadd normalization).
            builder
                .when(cols.sel_wit)
                .assert_eq(cols.result[i], cols.wit[i]);
        }

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
        let result = fcvt_f2i_result(a_bits, is_wu);

        let a_sign = (a_bits >> 31) & 1;
        let a_exp = (a_bits >> 23) & 0xFF;
        let a_mant = a_bits & 0x7FFFFF;

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
            self.range_checker.add_count((result >> (i * 8)) & 0xFF, 8);
        }
        core_row.a_sign = F::from_u32(a_sign);
        core_row.a_exp_lo = F::from_u32(a_exp & 1);
        core_row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        core_row.a_hi = F::from_u32(a_exp >> 1);

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
        self.lt_air.generate_subrow(
            (self.range_checker.as_ref(), a_exp, 158),
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
        core_row.is_valid = F::ONE;
    }
}
