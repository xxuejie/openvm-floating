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

/// FCLASS.S result bits (RISC-V spec).
pub const FCLASS_NEG_INF: u32 = 1 << 0;
pub const FCLASS_NEG_NORM: u32 = 1 << 1;
pub const FCLASS_NEG_SUBNORM: u32 = 1 << 2;
pub const FCLASS_NEG_ZERO: u32 = 1 << 3;
pub const FCLASS_POS_ZERO: u32 = 1 << 4;
pub const FCLASS_POS_SUBNORM: u32 = 1 << 5;
pub const FCLASS_POS_NORM: u32 = 1 << 6;
pub const FCLASS_POS_INF: u32 = 1 << 7;
pub const FCLASS_SNAN: u32 = 1 << 8;
pub const FCLASS_QNAN: u32 = 1 << 9;

/// IEEE 754 / RISC-V FCLASS.S: returns a 10-bit mask classifying the input.
#[inline(always)]
pub fn fclass_s(bits: u32) -> u32 {
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        if mant != 0 {
            // quiet bit = MSB of the mantissa field
            if mant & 0x400000 != 0 {
                FCLASS_QNAN
            } else {
                FCLASS_SNAN
            }
        } else if sign == 1 {
            FCLASS_NEG_INF
        } else {
            FCLASS_POS_INF
        }
    } else if exp == 0 {
        if mant != 0 {
            if sign == 1 {
                FCLASS_NEG_SUBNORM
            } else {
                FCLASS_POS_SUBNORM
            }
        } else if sign == 1 {
            FCLASS_NEG_ZERO
        } else {
            FCLASS_POS_ZERO
        }
    } else if sign == 1 {
        FCLASS_NEG_NORM
    } else {
        FCLASS_POS_NORM
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct FclassCoreCols<T> {
    pub a: [T; 4],

    // === Decomposition of input a ===
    pub a_sign: T,
    pub a_exp_lo: T, // bit 0 of exponent (top bit of byte 2)
    pub a_mant2: T,  // low 7 bits of byte 2
    pub a_hi: T,     // low 7 bits of byte 3 (high 7 bits of exponent)

    // === Classification inverse witnesses and flags ===
    pub a_exp_zero_inv: T,
    pub a_exp_max_inv: T,
    pub a_mant_sum_inv: T,
    pub a_exp_is_zero: T,
    pub a_exp_is_max: T,
    pub a_mant_is_zero: T,

    // === Quiet NaN bit (mantissa MSB) ===
    pub a_mant2_lo: T, // low 6 bits of a_mant2
    pub a_quiet: T,

    // === Category selectors (exactly one is 1 on valid rows) ===
    pub nan_sel: T,
    pub inf_sel: T,
    pub zero_sel: T,
    pub sub_sel: T,
    pub is_special: T, // nan_sel + inf_sel + zero_sel + sub_sel (normal = !is_special)

    pub result: [T; 4],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(FclassCoreCols<u8>)]
pub struct FclassCoreAir {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for FclassCoreAir {
    fn width(&self) -> usize {
        FclassCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for FclassCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FclassCoreAir
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
        let cols: &FclassCoreCols<AB::Var> = local_core.borrow();

        let f2: AB::F = AB::F::from_u32(2);
        let f128: AB::F = AB::F::from_u32(128);
        let f255: AB::F = AB::F::from_u32(255);
        let one = AB::Expr::ONE;

        // ── Booleanity ───────────────────────────────────────────────────
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.a_sign);
        builder.assert_bool(cols.a_exp_lo);
        builder.assert_bool(cols.a_exp_is_zero);
        builder.assert_bool(cols.a_exp_is_max);
        builder.assert_bool(cols.a_mant_is_zero);
        builder.assert_bool(cols.a_quiet);
        builder.assert_bool(cols.nan_sel);
        builder.assert_bool(cols.inf_sel);
        builder.assert_bool(cols.zero_sel);
        builder.assert_bool(cols.sub_sel);
        builder.assert_bool(cols.is_special);

        // ── Byte decomposition ───────────────────────────────────────────
        // a[2] = a_mant2 + a_exp_lo * 128, a[3] = a_hi + a_sign * 128
        builder.assert_eq(cols.a[2], cols.a_mant2 + cols.a_exp_lo * f128);
        builder.assert_eq(cols.a[3], cols.a_hi + cols.a_sign * f128);
        // a_mant2 = a_mant2_lo + a_quiet * 64 (quiet = mantissa MSB = bit 22)
        builder.assert_eq(
            cols.a_mant2,
            cols.a_mant2_lo + cols.a_quiet * AB::F::from_u32(64),
        );

        // Range checks: a_mant2_lo is 6 bits; result is 2 bytes (mask < 1024).
        self.range_bus
            .range_check(cols.a_mant2_lo.into(), 6)
            .eval(builder, cols.is_valid.into());
        self.range_bus
            .range_check(cols.result[0].into(), 8)
            .eval(builder, cols.is_valid.into());
        self.range_bus
            .range_check(cols.result[1].into(), 8)
            .eval(builder, cols.is_valid.into());

        // ── Classification ───────────────────────────────────────────────
        // exp = exp_lo + 2 * hi, mant_sum = a[0] + a[1] + a_mant2
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

        // ── Category selectors ───────────────────────────────────────────
        builder.assert_eq(
            cols.nan_sel,
            cols.a_exp_is_max * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(cols.inf_sel, cols.a_exp_is_max * cols.a_mant_is_zero);
        builder.assert_eq(cols.zero_sel, cols.a_exp_is_zero * cols.a_mant_is_zero);
        builder.assert_eq(
            cols.sub_sel,
            cols.a_exp_is_zero * (one.clone() - cols.a_mant_is_zero),
        );
        builder.assert_eq(
            cols.is_special,
            cols.nan_sel + cols.inf_sel + cols.zero_sel + cols.sub_sel,
        );

        // ── Result mask ──────────────────────────────────────────────────
        // snan = 256, qnan = 512 = 256 * (1 + quiet)
        // -inf = 1,    +inf = 128 = 128 - 127 * sign
        // -0 = 8,      +0 = 16    = 16 - 8 * sign
        // -sub = 4,    +sub = 32  = 32 - 28 * sign
        // -norm = 2,   +norm = 64 = 64 - 62 * sign
        // The normal case is multiplied by is_valid so that the all-zero
        // padding row (norm = 1 - is_special = 1) evaluates the mask to 0.
        let mask = cols.nan_sel * AB::F::from_u32(256) * (one.clone() + cols.a_quiet)
            + cols.inf_sel
                * (AB::Expr::from(AB::F::from_u32(128)) - cols.a_sign * AB::F::from_u32(127))
            + cols.zero_sel
                * (AB::Expr::from(AB::F::from_u32(16)) - cols.a_sign * AB::F::from_u32(8))
            + cols.sub_sel
                * (AB::Expr::from(AB::F::from_u32(32)) - cols.a_sign * AB::F::from_u32(28))
            + cols.is_valid
                * (one.clone() - cols.is_special)
                * (AB::Expr::from(AB::F::from_u32(64)) - cols.a_sign * AB::F::from_u32(62));
        builder.assert_eq(cols.result[0] + cols.result[1] * AB::F::from_u32(256), mask);
        builder.assert_zero(cols.result[2]);
        builder.assert_zero(cols.result[3]);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, FpOpcode::FCLASS_S);

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
pub struct FclassCoreRecord {
    pub a: [u8; 4],
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct FclassExecutor<A> {
    adapter: A,
    pub offset: usize,
    pub read_as: u32,
    pub write_as: u32,
}

impl<A, F, RA> PreflightExecutor<F, RA> for FclassExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<[[u8; 4]; 1]>, WriteData: From<[[u8; 4]; 1]>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FclassCoreRecord),
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
        debug_assert_eq!(FpOpcode::from_usize(local), FpOpcode::FCLASS_S);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [a] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.a = a;

        let result = fclass_s(u32::from_le_bytes(a));
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

pub struct FclassFiller<A> {
    adapter: A,
    pub offset: usize,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl<A> FclassFiller<A> {
    pub fn new(adapter: A, range_checker: SharedVariableRangeCheckerChip, offset: usize) -> Self {
        Self {
            adapter,
            offset,
            range_checker,
        }
    }
}

impl<A, F> TraceFiller<F> for FclassFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &FclassCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FclassCoreCols<F> = core_row.borrow_mut();

        let a_bits = u32::from_le_bytes(record.a);
        let result = fclass_s(a_bits);

        let a_sign = (a_bits >> 31) & 1;
        let a_exp = (a_bits >> 23) & 0xFF;
        let a_mant = a_bits & 0x7FFFFF;

        for i in 0..4 {
            core_row.a[i] = F::from_u32((a_bits >> (i * 8)) & 0xFF);
            core_row.result[i] = F::from_u32((result >> (i * 8)) & 0xFF);
        }
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

        let a_mant2 = (a_mant >> 16) & 0x7F;
        core_row.a_quiet = F::from_u32(a_mant2 >> 6);
        core_row.a_mant2_lo = F::from_u32(a_mant2 & 0x3F);

        let is_nan = a_exp == 255 && a_mant != 0;
        let is_inf = a_exp == 255 && a_mant == 0;
        let is_zero = a_exp == 0 && a_mant == 0;
        let is_sub = a_exp == 0 && a_mant != 0;
        core_row.nan_sel = F::from_bool(is_nan);
        core_row.inf_sel = F::from_bool(is_inf);
        core_row.zero_sel = F::from_bool(is_zero);
        core_row.sub_sel = F::from_bool(is_sub);
        core_row.is_special = F::from_bool(is_nan || is_inf || is_zero || is_sub);

        self.range_checker.add_count(a_mant2 & 0x3F, 6);
        self.range_checker.add_count(result & 0xFF, 8);
        self.range_checker.add_count((result >> 8) & 0xFF, 8);

        core_row.is_valid = F::ONE;
    }
}
