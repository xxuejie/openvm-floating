use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

/// Adapter columns for a cross-address-space register-to-register move
/// (`fmv.x.s` / `fmv.s.x`): reads the source register from `read_as`, writes
/// the destination register to `write_as`.
///
/// Unlike the fixed-AS adapters, `read_as` / `write_as` are *witness columns*:
/// the filler copies them from the record (which the executor reads from the
/// instruction's `d`/`e` fields), and the execution bridge sends them as the
/// instruction operands so the program bus pins them to the program's `d`/`e`.
/// The memory bus also pins them (the address-space is part of each access).
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv32FpMoveAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,   // a: destination register offset
    pub rs1_ptr: T,  // b: source register offset
    pub read_as: T,  // e: source address space
    pub write_as: T, // d: destination address space
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Adapter AIR for cross-AS register moves (FMV.X.S / FMV.S.X).
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv32FpMoveAdapterCols<u8>)]
pub struct Rv32FpMoveAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32FpMoveAdapterAir {
    fn width(&self) -> usize {
        Rv32FpMoveAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32FpMoveAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        1,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32FpMoveAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(local.read_as, local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(local.write_as, local.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    AB::Expr::ZERO,
                    local.write_as.into(),
                    local.read_as.into(),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32FpMoveAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32FpMoveAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub read_as: u32,
    pub write_as: u32,
    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FpMoveAdapterExecutor;

impl<F> AdapterTraceExecutor<F> for Rv32FpMoveAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32FpMoveAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32FpMoveAdapterRecord;

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
        let &Instruction { b, e, .. } = instruction;
        record.read_as = e.as_canonical_u32();
        record.rs1_ptr = b.as_canonical_u32();
        let (t0, rs1) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(e.as_canonical_u32(), b.as_canonical_u32())
        };
        record.reads_aux.prev_timestamp = t0;

        [rs1]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        write_data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;
        record.write_as = d.as_canonical_u32();
        record.rd_ptr = a.as_canonical_u32();
        let [rd] = write_data;
        let (t_prev, prev_data) = unsafe {
            memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(
                d.as_canonical_u32(),
                a.as_canonical_u32(),
                rd,
            )
        };
        record.writes_aux.prev_timestamp = t_prev;
        record.writes_aux.prev_data = prev_data;
    }
}

// === Filler ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FpMoveAdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32FpMoveAdapterFiller {
    const WIDTH: usize = size_of::<Rv32FpMoveAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32FpMoveAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32FpMoveAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = record.from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp + 1,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            record.reads_aux.prev_timestamp,
            timestamp,
            adapter_row.reads_aux.as_mut(),
        );

        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);
        adapter_row.read_as = F::from_u32(record.read_as);
        adapter_row.write_as = F::from_u32(record.write_as);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
