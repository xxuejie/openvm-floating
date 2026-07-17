use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::fminmax::{fminmax_result, FminmaxExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FminmaxPreCompute {
    a: u8,
    b: u8,
    c: u8,
    register_as: u8,
    is_min: u8,
}

impl<A> FminmaxExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FminmaxPreCompute,
    ) -> Result<(), StaticProgramError> {
        let local = inst.opcode.local_opcode_idx(self.offset);
        let op = FpOpcode::from_usize(local);
        let is_min = match op {
            FpOpcode::FMIN_S => true,
            FpOpcode::FMAX_S => false,
            _ => return Err(StaticProgramError::InvalidInstruction(pc)),
        };
        *data = FminmaxPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
            register_as: self.register_as as u8,
            is_min: is_min as u8,
        };
        Ok(())
    }
}

impl<A, F> InterpreterExecutor<F> for FminmaxExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FminmaxPreCompute>()
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
        let pre_compute: &mut FminmaxPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fminmax_impl)
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
        let pre_compute: &mut FminmaxPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fminmax_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fminmax_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FminmaxPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FminmaxPreCompute>()).borrow();
    execute_fminmax_core(pre_compute, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fminmax_handler<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FminmaxPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FminmaxPreCompute>()).borrow();
    execute_fminmax_core(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_fminmax_core<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FminmaxPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let as_u32 = pre_compute.register_as as u32;
    let rs1: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.b as u32);
    let rs2: [u8; 4] = exec_state.vm_read(as_u32, pre_compute.c as u32);
    let result = fminmax_result(
        u32::from_le_bytes(rs1),
        u32::from_le_bytes(rs2),
        pre_compute.is_min != 0,
    );
    exec_state.vm_write(as_u32, pre_compute.a as u32, &result.to_le_bytes());
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

// === Metered executor ===

impl<A, F> InterpreterMeteredExecutor<F> for FminmaxExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FminmaxPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<FminmaxPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fminmax_metered)
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
        let pre_compute: &mut E2PreCompute<FminmaxPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fminmax_metered_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fminmax_metered<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FminmaxPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FminmaxPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fminmax_core(&pre_compute.data, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fminmax_metered_handler<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FminmaxPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FminmaxPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fminmax_core(&pre_compute.data, exec_state);
}
