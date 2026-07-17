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
#[derive(AlignedBorrow, StructReflection, Debug)]
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
        // +-0 result: finite / inf.
        builder.assert_eq(
            cols.zero_res,
            cols.b_inf * (one.clone() - cols.a_inf) * (one.clone() - cols.any_nan),
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

        // TODO: verify the finite (non-special) result — correctly-rounded
        // division needs 24-bit long division, normalization, and RNE
        // rounding machinery (similar to fmul normalization).

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
        let zero_res = b_inf && !a_inf && !any_nan;

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
