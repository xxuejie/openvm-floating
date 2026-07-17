use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::MemoryBridge, online::TracingMemory, MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::ColumnsAir;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{Rv32FpAdapterCols, Rv32FpAdapterRecord, RV32_REGISTER_NUM_LIMBS};

/// Adapter AIR for FP comparison operations: reads 2 FP registers from
/// `read_as` and writes 1 integer register to `RV32_REGISTER_AS` (AS 1).
/// feq.s / flt.s / fle.s always write an integer register, even in
/// F extension mode where the FP register file is separate.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv32FpAdapterCols<u8>)]
pub struct Rv32FcmpAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub read_as: u32,
}

impl<F: Field> BaseAir<F> for Rv32FcmpAdapterAir {
    fn width(&self) -> usize {
        Rv32FpAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32FcmpAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
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
        let local: &Rv32FpAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let read_as = AB::F::from_u32(self.read_as);
        let write_as = AB::F::from_u32(RV32_REGISTER_AS);

        self.memory_bridge
            .read(
                MemoryAddress::new(read_as, local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(read_as, local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
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
                    local.rs2_ptr.into(),
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
        let cols: &Rv32FpAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// === Executor ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FcmpAdapterExecutor {
    pub read_as: u32,
}

impl<F> AdapterTraceExecutor<F> for Rv32FcmpAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32FpAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut Rv32FpAdapterRecord;

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
        let &Instruction { b, c, d, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rs1_ptr = b.as_canonical_u32();
        let (t0, rs1) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.read_as, b.as_canonical_u32())
        };
        record.rs2_ptr = c.as_canonical_u32();
        let (t1, rs2) = unsafe {
            memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(self.read_as, c.as_canonical_u32())
        };
        record.reads_aux[0].prev_timestamp = t0;
        record.reads_aux[1].prev_timestamp = t1;

        [rs1, rs2]
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
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        let [rd] = write_data;
        let (t_prev, prev_data) = unsafe {
            memory.write::<u8, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, a.as_canonical_u32(), rd)
        };
        record.writes_aux.prev_timestamp = t_prev;
        record.writes_aux.prev_data = prev_data;
    }
}

// === Filler ===

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32FcmpAdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32FcmpAdapterFiller {
    const WIDTH: usize = size_of::<Rv32FpAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32FpAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32FpAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = record.from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp + 2,
            adapter_row.writes_aux.as_mut(),
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

        adapter_row.rs2_ptr = F::from_u32(record.rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
