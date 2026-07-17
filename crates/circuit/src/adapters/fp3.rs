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

/// Adapter columns for ternary FP register read/write (3 reads, 1 write).
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv32Fp3AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub rs3_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 3],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Adapter AIR for ternary FP operations (fused multiply-add family):
/// reads 3 FP registers, writes 1 FP register, all in `register_as`.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv32Fp3AdapterCols<u8>)]
pub struct Rv32Fp3AdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub register_as: u32,
}

impl<F: Field> BaseAir<F> for Rv32Fp3AdapterAir {
    fn width(&self) -> usize {
        Rv32Fp3AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32Fp3AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        3,
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
        let local: &Rv32Fp3AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let fp_as = AB::F::from_u32(self.register_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(fp_as, local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(fp_as, local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(fp_as, local.rs3_ptr),
                ctx.reads[2].clone(),
                timestamp_pp(),
                &local.reads_aux[2],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(fp_as, local.rd_ptr),
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
                    local.rs2_ptr.into(),
                    local.rs3_ptr.into(),
                    fp_as.into(),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32Fp3AdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// === Record ===

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32Fp3AdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub rs3_ptr: u32,
    pub reads_aux: [MemoryReadAuxRecord; 3],
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32Fp3AdapterExecutor {
    pub register_as: u32,
}

impl<F> AdapterTraceExecutor<F> for Rv32Fp3AdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32Fp3AdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 3];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32Fp3AdapterRecord;

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
        let &Instruction { b, c, d, e, .. } = instruction;
        debug_assert_eq!(e.as_canonical_u32(), self.register_as);

        record.rs1_ptr = b.as_canonical_u32();
        let (t0, rs1) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.register_as, b.as_canonical_u32())
        };
        record.rs2_ptr = c.as_canonical_u32();
        let (t1, rs2) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.register_as, c.as_canonical_u32())
        };
        record.rs3_ptr = d.as_canonical_u32();
        let (t2, rs3) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.register_as, d.as_canonical_u32())
        };
        record.reads_aux[0].prev_timestamp = t0;
        record.reads_aux[1].prev_timestamp = t1;
        record.reads_aux[2].prev_timestamp = t2;

        [rs1, rs2, rs3]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        write_data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, e, .. } = instruction;
        debug_assert_eq!(e.as_canonical_u32(), self.register_as);

        record.rd_ptr = a.as_canonical_u32();
        let [rd] = write_data;
        let (t_prev, prev_data) = unsafe {
            memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(self.register_as, a.as_canonical_u32(), rd)
        };
        record.writes_aux.prev_timestamp = t_prev;
        record.writes_aux.prev_data = prev_data;
    }
}

// === Filler ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32Fp3AdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32Fp3AdapterFiller {
    const WIDTH: usize = size_of::<Rv32Fp3AdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32Fp3AdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32Fp3AdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = record.from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp + 3,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            record.reads_aux[2].prev_timestamp,
            timestamp + 2,
            adapter_row.reads_aux[2].as_mut(),
        );
        mem_helper.fill(
            record.reads_aux[1].prev_timestamp,
            timestamp + 1,
            adapter_row.reads_aux[1].as_mut(),
        );
        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs3_ptr = F::from_u32(record.rs3_ptr);
        adapter_row.rs2_ptr = F::from_u32(record.rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
