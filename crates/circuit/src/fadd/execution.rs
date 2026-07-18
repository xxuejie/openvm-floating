use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::fadd::{fadd_s, FaddExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FpPreCompute {
    a: u8,
    b: u8,
    c: u8,
    register_as: u8,
}

impl<A> FaddExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FpPreCompute,
    ) -> Result<(), StaticProgramError> {
        if FpOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset)) != FpOpcode::FADD_S {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = FpPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
            register_as: self.register_as as u8,
        };
        Ok(())
    }
}

impl<A, F> InterpreterExecutor<F> for FaddExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FpPreCompute>()
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
        let pre_compute: &mut FpPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fadd_impl)
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
        let pre_compute: &mut FpPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fadd_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fadd_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FpPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FpPreCompute>()).borrow();
    execute_fadd_core(pre_compute, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fadd_handler<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FpPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FpPreCompute>()).borrow();
    execute_fadd_core(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_fadd_core<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FpPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let as_u32 = pre_compute.register_as as u32;
    let rs1: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.b as u32);
    let rs2: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.c as u32);
    let result = fadd_s(u32::from_le_bytes(rs1), u32::from_le_bytes(rs2)).to_le_bytes();
    exec_state.vm_write(as_u32, pre_compute.a as u32, &result);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

// === Metered executor (for proving / preflight) ===

impl<A, F> InterpreterMeteredExecutor<F> for FaddExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FpPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<FpPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fadd_metered)
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
        let pre_compute: &mut E2PreCompute<FpPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fadd_metered_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fadd_metered<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FpPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FpPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fadd_core(&pre_compute.data, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fadd_metered_handler<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FpPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FpPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fadd_core(&pre_compute.data, exec_state);
}
