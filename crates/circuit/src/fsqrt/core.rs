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
#[derive(AlignedBorrow, StructReflection, Debug)]
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

        // Range check input bytes (bounds exponent/mantissa sums).
        for i in 0..4 {
            self.range_bus
                .range_check(cols.a[i].into(), 8)
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

        // TODO: verify the finite positive result — correctly-rounded square
        // root needs integer isqrt + normalization + RNE rounding machinery.

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
        let result = fsqrt_s(a_bits);

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
            self.range_checker.add_count((a_bits >> (i * 8)) & 0xFF, 8);
        }

        let (a_sign, a_exp, a_mant) = decompose(a_bits);
        core_row.a_sign = F::from_u32(a_sign);
        core_row.a_exp_lo = F::from_u32(a_exp & 1);
        core_row.a_mant2 = F::from_u32((a_mant >> 16) & 0x7F);
        core_row.a_hi = F::from_u32(a_exp >> 1);

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
