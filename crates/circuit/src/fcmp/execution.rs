use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::fcmp::{fcmp_result, opcode_to_fcmp_op, FcmpExecutor, FcmpOp};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FcmpPreCompute {
    a: u8,
    b: u8,
    c: u8,
    read_as: u8,
    op: u8, // 0 = fle, 1 = flt, 2 = feq (matches funct3)
}

impl<A> FcmpExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FcmpPreCompute,
    ) -> Result<(), StaticProgramError> {
        let local = inst.opcode.local_opcode_idx(self.offset);
        let op = opcode_to_fcmp_op(FpOpcode::from_usize(local))
            .ok_or(StaticProgramError::InvalidInstruction(pc))?;
        let op = match op {
            FcmpOp::Le => 0,
            FcmpOp::Lt => 1,
            FcmpOp::Eq => 2,
        };
        *data = FcmpPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
            read_as: self.register_as as u8,
            op,
        };
        Ok(())
    }
}

impl<A, F> InterpreterExecutor<F> for FcmpExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FcmpPreCompute>()
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
        let pre_compute: &mut FcmpPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fcmp_impl)
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
        let pre_compute: &mut FcmpPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fcmp_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fcmp_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FcmpPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FcmpPreCompute>()).borrow();
    execute_fcmp_core(pre_compute, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fcmp_handler<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FcmpPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FcmpPreCompute>()).borrow();
    execute_fcmp_core(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_fcmp_core<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FcmpPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let read_as = pre_compute.read_as as u32;
    let rs1: [u8; 4] = exec_state.vm_read(read_as, pre_compute.b as u32);
    let rs2: [u8; 4] = exec_state.vm_read(read_as, pre_compute.c as u32);
    let op = match pre_compute.op {
        0 => FcmpOp::Le,
        1 => FcmpOp::Lt,
        _ => FcmpOp::Eq,
    };
    let result = fcmp_result(u32::from_le_bytes(rs1), u32::from_le_bytes(rs2), op);
    // feq/flt/fle write an integer register (AS 1), not an FP register
    exec_state.vm_write(
        RV32_REGISTER_AS,
        pre_compute.a as u32,
        &result.to_le_bytes(),
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

// === Metered executor ===

impl<A, F> InterpreterMeteredExecutor<F> for FcmpExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FcmpPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<FcmpPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fcmp_metered)
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
        let pre_compute: &mut E2PreCompute<FcmpPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fcmp_metered_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fcmp_metered<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FcmpPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FcmpPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fcmp_core(&pre_compute.data, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fcmp_metered_handler<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FcmpPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FcmpPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fcmp_core(&pre_compute.data, exec_state);
}
