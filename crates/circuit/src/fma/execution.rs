use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::fma::{fmadd_result, opcode_to_fmadd_op, FmaExecutor, FmaddOp};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FmaPreCompute {
    a: u8,
    b: u8,
    c: u8,
    rs3: u8,
    register_as: u8,
    op: u8, // 0 = madd, 1 = msub, 2 = nmsub, 3 = nmadd
}

impl<A> FmaExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FmaPreCompute,
    ) -> Result<(), StaticProgramError> {
        let local = inst.opcode.local_opcode_idx(self.offset);
        let op = opcode_to_fmadd_op(FpOpcode::from_usize(local))
            .ok_or(StaticProgramError::InvalidInstruction(pc))?;
        let op = match op {
            FmaddOp::Madd => 0,
            FmaddOp::Msub => 1,
            FmaddOp::Nmsub => 2,
            FmaddOp::Nmadd => 3,
        };
        *data = FmaPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
            rs3: inst.d.as_canonical_u32() as u8,
            register_as: self.register_as as u8,
            op,
        };
        Ok(())
    }
}

impl<A, F> InterpreterExecutor<F> for FmaExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FmaPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut FmaPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fma_impl)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut FmaPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fma_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fma_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FmaPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FmaPreCompute>()).borrow();
    execute_fma_core(pre_compute, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fma_handler<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FmaPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FmaPreCompute>()).borrow();
    execute_fma_core(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_fma_core<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FmaPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let as_u32 = pre_compute.register_as as u32;
    let rs1: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.b as u32);
    let rs2: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.c as u32);
    let rs3: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.rs3 as u32);
    let op = match pre_compute.op {
        0 => FmaddOp::Madd,
        1 => FmaddOp::Msub,
        2 => FmaddOp::Nmsub,
        _ => FmaddOp::Nmadd,
    };
    let result = fmadd_result(
        u32::from_le_bytes(rs1),
        u32::from_le_bytes(rs2),
        u32::from_le_bytes(rs3),
        op,
    );
    exec_state.vm_write(as_u32, pre_compute.a as u32, &result.to_le_bytes());
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

// === Metered executor ===

impl<A, F> InterpreterMeteredExecutor<F> for FmaExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FmaPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<FmaPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fma_metered)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<FmaPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fma_metered_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fma_metered<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FmaPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FmaPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fma_core(&pre_compute.data, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fma_metered_handler<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FmaPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FmaPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fma_core(&pre_compute.data, exec_state);
}
