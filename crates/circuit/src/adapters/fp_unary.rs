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

/// Adapter columns for unary FP register read/write.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv32FpUnaryAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Adapter AIR for unary FP operations: reads 1 register, writes 1 register.
/// Used by fclass.s and the fcvt conversion family, where one operand is an
/// FP register and the other is an integer register:
/// - fclass.s / fcvt.w.s / fcvt.wu.s: read `read_as` (FP), write `write_as` (int, AS 1)
/// - fcvt.s.w / fcvt.s.wu: read `read_as` (int, AS 1), write `write_as` (FP)
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv32FpUnaryAdapterCols<u8>)]
pub struct Rv32FpUnaryAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub read_as: u32,
    pub write_as: u32,
}

impl<F: Field> BaseAir<F> for Rv32FpUnaryAdapterAir {
    fn width(&self) -> usize {
        Rv32FpUnaryAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32FpUnaryAdapterAir {
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
        let local: &Rv32FpUnaryAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let read_as = AB::F::from_u32(self.read_as);
        let write_as = AB::F::from_u32(self.write_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(read_as, local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(write_as, local.rd_ptr),
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
                    write_as.into(),
                    read_as.into(),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32FpUnaryAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32FpUnaryAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub reads_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FpUnaryAdapterExecutor {
    pub read_as: u32,
    pub write_as: u32,
}

impl<F> AdapterTraceExecutor<F> for Rv32FpUnaryAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32FpUnaryAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32FpUnaryAdapterRecord;

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
        let &Instruction { b, d, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), self.write_as);

        record.rs1_ptr = b.as_canonical_u32();
        let (t0, rs1) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.read_as, b.as_canonical_u32())
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
        debug_assert_eq!(d.as_canonical_u32(), self.write_as);

        record.rd_ptr = a.as_canonical_u32();
        let [rd] = write_data;
        let (t_prev, prev_data) = unsafe {
            memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(self.write_as, a.as_canonical_u32(), rd)
        };
        record.writes_aux.prev_timestamp = t_prev;
        record.writes_aux.prev_data = prev_data;
    }
}

// === Filler ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FpUnaryAdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32FpUnaryAdapterFiller {
    const WIDTH: usize = size_of::<Rv32FpUnaryAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32FpUnaryAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32FpUnaryAdapterCols<F> = adapter_row.borrow_mut();

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

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
