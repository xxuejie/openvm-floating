use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VmAdapterAir, VmAdapterInterface,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    utils::select,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_MEMORY_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

/// Processed instruction for the FP load/store adapter: the core pins `is_load`
/// to the opcode (FLW vs FSW) so the adapter knows which side of the move is
/// the FP register file and which is AS 2.
pub struct FpLoadStoreInstruction<T> {
    /// `is_valid` is constrained to be bool.
    pub is_valid: T,
    /// Absolute opcode number.
    pub opcode: T,
    /// `is_load` is constrained to be bool, and can only be 1 if `is_valid` is 1.
    pub is_load: T,
}

/// Custom interface: one value read (the moved 4 bytes) + one value write, with
/// the richer processed-instruction carrying `is_load`.
pub struct Rv32FpMemAdapterInterface<AB: InteractionBuilder>(PhantomData<AB>);

impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr> for Rv32FpMemAdapterInterface<AB> {
    type Reads = [[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1];
    type Writes = [[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1];
    type ProcessedInstruction = FpLoadStoreInstruction<AB::Expr>;
}

/// Adapter columns for FLW / FSW (word-aligned FP memory accesses).
///
/// Both instructions read the base address register from AS 1 (`base_ptr`),
/// compute `mem_ptr = base + sign_extend(imm)` (2×16-bit limbs with carry), and
/// move 4 bytes between AS 2 at `mem_ptr` and the FP register file:
/// - FLW (`is_load=1`): read AS 2 @ `mem_ptr` -> write FP AS @ `fp_ptr`
/// - FSW (`is_load=0`): read FP AS @ `fp_ptr` -> write AS 2 @ `mem_ptr`
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv32FpMemAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub base_ptr: T, // base register offset (AS 1): b for FLW, a for FSW
    pub base_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub base_aux: MemoryReadAuxCols<T>,
    pub fp_ptr: T,                       // FP register offset: a for FLW, b for FSW
    pub imm: T,      // 16-bit immediate (low 16 bits of the sign-extended 12-bit imm)
    pub imm_sign: T, // bool = bit 15 of imm
    pub imm_lo: T,   // low 15 bits of imm (range-checked, pins imm_sign)
    pub mem_ptr_limbs: [T; 2], // mem_ptr = base + sign_extend(imm), 2×16-bit limbs
    pub value_aux: MemoryReadAuxCols<T>, // value read (AS 2 for FLW, FP AS for FSW)
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv32FpMemAdapterCols<u8>)]
pub struct Rv32FpMemAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub fp_as: u32,
    pub pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv32FpMemAdapterAir {
    fn width(&self) -> usize {
        Rv32FpMemAdapterCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32FpMemAdapterAir {}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32FpMemAdapterAir {
    type Interface = Rv32FpMemAdapterInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32FpMemAdapterCols<AB::Var> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let is_load = ctx.instruction.is_load.clone();
        let is_valid = ctx.instruction.is_valid.clone();
        let fp_as = AB::F::from_u32(self.fp_as);
        let mem_as = AB::F::from_u32(RV32_MEMORY_AS);

        // Read the base address register from AS 1.
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::ONE, local.base_ptr),
                local.base_data,
                timestamp_pp(),
                &local.base_aux,
            )
            .eval(builder, is_valid.clone());

        // Pin imm_sign to bit 15 of imm (imm_lo range-checked to 15 bits).
        builder.when(is_valid.clone()).assert_bool(local.imm_sign);
        builder.assert_eq(
            local.imm,
            local.imm_lo + local.imm_sign * AB::F::from_u32(1 << 15),
        );
        self.range_bus
            .range_check(local.imm_lo.into(), 15)
            .eval(builder, is_valid.clone());

        // mem_ptr = base + sign_extend(imm) as a 32-bit add over 2×16-bit limbs.
        let limbs_01 = local.base_data[0] + local.base_data[1] * AB::F::from_u32(1 << 8);
        let limbs_23 = local.base_data[2] + local.base_data[3] * AB::F::from_u32(1 << 8);
        let inv = AB::F::from_u32(1 << 16).inverse();
        let carry = (limbs_01 + local.imm - local.mem_ptr_limbs[0]) * inv;
        builder.when(is_valid.clone()).assert_bool(carry.clone());
        let imm_extend_limb = local.imm_sign * AB::F::from_u32((1 << 16) - 1);
        let carry2 = (limbs_23 + imm_extend_limb + carry - local.mem_ptr_limbs[1]) * inv;
        builder.when(is_valid.clone()).assert_bool(carry2);

        // Word-aligned and in-range: mem_ptr_limbs[0] is a multiple of 4
        // (limb0/4 range-checked to 14 bits), limb1 bounded by pointer_max_bits.
        self.range_bus
            .range_check(local.mem_ptr_limbs[0] * AB::F::from_u32(4).inverse(), 14)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(local.mem_ptr_limbs[1], self.pointer_max_bits - 16)
            .eval(builder, is_valid.clone());

        let mem_ptr = local.mem_ptr_limbs[0] + local.mem_ptr_limbs[1] * AB::F::from_u32(1 << 16);

        // The value source: AS 2 @ mem_ptr for FLW, FP AS @ fp_ptr for FSW.
        let read_as: AB::Expr = select(is_load.clone(), mem_as, fp_as);
        let read_ptr = select(is_load.clone(), mem_ptr.clone(), local.fp_ptr);
        self.memory_bridge
            .read(
                MemoryAddress::new(read_as.clone(), read_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.value_aux,
            )
            .eval(builder, is_valid.clone());

        // The value destination: FP AS @ fp_ptr for FLW, AS 2 @ mem_ptr for FSW.
        let write_as: AB::Expr = select(is_load.clone(), fp_as, mem_as);
        let write_ptr = select(is_load.clone(), local.fp_ptr, mem_ptr.clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(write_as, write_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, is_valid.clone());

        // The program operand layout differs per opcode: FLW is
        // (a=FP rd, b=base, d=fp_as, e=mem_as, f=0), FSW is (a=base, b=FP src,
        // d=mem_as, e=fp_as, f=1). Claim all 7 operands (a..g) so the program
        // bus lookup matches either.
        let a_op = select(is_load.clone(), local.fp_ptr, local.base_ptr);
        let b_op = select(is_load.clone(), local.base_ptr, local.fp_ptr);
        let d_op: AB::Expr = select(is_load.clone(), fp_as, mem_as);
        let e_op: AB::Expr = select(is_load.clone(), mem_as, fp_as);
        let f_op = AB::Expr::ONE - is_load.clone();
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    a_op,
                    b_op,
                    local.imm.into(),
                    d_op,
                    e_op,
                    f_op,
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32FpMemAdapterCols<AB::Var> = local.borrow();
        cols.from_state.pc
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32FpMemAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub base_ptr: u32,
    pub base_val: u32,
    pub base_aux_prev_timestamp: u32,
    pub fp_ptr: u32,
    pub imm: u16,
    pub imm_sign: bool,
    pub mem_ptr: u32,
    pub value_prev_timestamp: u32,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FpMemAdapterExecutor {
    pub fp_as: u32,
    pub pointer_max_bits: usize,
}

impl<F> AdapterTraceExecutor<F> for Rv32FpMemAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32FpMemAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32FpMemAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction {
            a, b, c, d, e, f, ..
        } = instruction;
        let is_load = f.is_zero(); // FLW sets f=0, FSW sets f=1 (see transpiler)
        let (base_ptr, fp_ptr) = if is_load {
            (b.as_canonical_u32(), a.as_canonical_u32())
        } else {
            (a.as_canonical_u32(), b.as_canonical_u32())
        };
        debug_assert_eq!(if is_load { e } else { d }, F::from_u32(RV32_MEMORY_AS));
        debug_assert_eq!(if is_load { d } else { e }, F::from_u32(self.fp_as));

        record.base_ptr = base_ptr;
        record.fp_ptr = fp_ptr;
        let (base_t_prev, base_bytes) =
            unsafe { memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(1, base_ptr) };
        record.base_aux_prev_timestamp = base_t_prev;
        record.base_val = u32::from_le_bytes(base_bytes);

        let imm_u16 = (c.as_canonical_u32() & 0xFFFF) as u16;
        record.imm = imm_u16;
        record.imm_sign = imm_u16 & 0x8000 != 0;
        let sign_ext = imm_u16 as i16 as i32 as u32;
        let mem_ptr = record.base_val.wrapping_add(sign_ext) & !3;
        assert!(
            (mem_ptr as u64) < (1u64 << self.pointer_max_bits),
            "FLW/FSW mem_ptr {mem_ptr:#x} >= 2^{}",
            self.pointer_max_bits
        );
        record.mem_ptr = mem_ptr;

        let (value_t_prev, value) = if is_load {
            unsafe { memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(RV32_MEMORY_AS, mem_ptr) }
        } else {
            unsafe { memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.fp_as, fp_ptr) }
        };
        record.value_prev_timestamp = value_t_prev;

        [value]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        write_data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { f, .. } = instruction;
        let is_load = f.is_zero();
        let [value] = write_data;
        let (t_prev, prev_data) = if is_load {
            unsafe { memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(self.fp_as, record.fp_ptr, value) }
        } else {
            unsafe {
                memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(RV32_MEMORY_AS, record.mem_ptr, value)
            }
        };
        record.writes_aux.prev_timestamp = t_prev;
        record.writes_aux.prev_data = prev_data;
    }
}

// === Filler ===

#[derive(Clone, derive_new::new)]
pub struct Rv32FpMemAdapterFiller {
    pub range_checker: SharedVariableRangeCheckerChip,
    pub fp_as: u32,
    pub pointer_max_bits: usize,
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32FpMemAdapterFiller {
    const WIDTH: usize = size_of::<Rv32FpMemAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32FpMemAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        // NOTE: `record` aliases the start of the adapter region, which the
        // `core_row.borrow_mut()` below also writes. Copy every field we need to
        // locals *before* any cols write, which would otherwise clobber them.
        let from_pc = record.from_pc;
        let from_timestamp = record.from_timestamp;
        let base_ptr = record.base_ptr;
        let base_val = record.base_val;
        let base_aux_prev_timestamp = record.base_aux_prev_timestamp;
        let fp_ptr = record.fp_ptr;
        let imm = record.imm;
        let imm_sign = record.imm_sign;
        let mem_ptr = record.mem_ptr;
        let value_prev_timestamp = record.value_prev_timestamp;
        let writes_prev_timestamp = record.writes_aux.prev_timestamp;
        let writes_prev_data = record.writes_aux.prev_data;
        let adapter_row: &mut Rv32FpMemAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(writes_prev_data.map(F::from_u8));
        mem_helper.fill(
            writes_prev_timestamp,
            timestamp + 2,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            value_prev_timestamp,
            timestamp + 1,
            adapter_row.value_aux.as_mut(),
        );
        mem_helper.fill(
            base_aux_prev_timestamp,
            timestamp,
            adapter_row.base_aux.as_mut(),
        );

        let ptr_limbs = [mem_ptr & 0xFFFF, mem_ptr >> 16];
        self.range_checker.add_count(ptr_limbs[0] >> 2, 14);
        self.range_checker
            .add_count(ptr_limbs[1], self.pointer_max_bits - 16);
        self.range_checker.add_count(imm as u32 & 0x7FFF, 15);
        adapter_row.mem_ptr_limbs = ptr_limbs.map(F::from_u32);
        adapter_row.imm = F::from_u16(imm);
        adapter_row.imm_lo = F::from_u32((imm as u32) & 0x7FFF);
        adapter_row.imm_sign = F::from_bool(imm_sign);

        adapter_row.fp_ptr = F::from_u32(fp_ptr);
        adapter_row.base_ptr = F::from_u32(base_ptr);
        adapter_row.base_data = base_val.to_le_bytes().map(F::from_u8);

        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);
    }
}
