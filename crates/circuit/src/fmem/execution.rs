use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::fmem::FmemExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FmemPreCompute {
    a: u8,
    b: u8,
    c: u16,
    fp_as: u8,
    is_load: u8,
}

impl<A> FmemExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FmemPreCompute,
    ) -> Result<(), StaticProgramError> {
        let local = inst.opcode.local_opcode_idx(self.offset);
        let op = FpOpcode::from_usize(local);
        if !matches!(op, FpOpcode::FLW | FpOpcode::FSW) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        // The transpiler encodes: a/b hold the FP register (and base register,
        // in the store's `a` slot), c = low 16 bits of the sign-extended imm,
        // f = 1 for FSW (store), 0 for FLW (load).
        *data = FmemPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: (inst.c.as_canonical_u32() & 0xFFFF) as u16,
            fp_as: if inst.f.is_zero() {
                inst.d.as_canonical_u32() as u8
            } else {
                inst.e.as_canonical_u32() as u8
            },
            is_load: inst.f.is_zero() as u8,
        };
        Ok(())
    }
}

impl<A, F> InterpreterExecutor<F> for FmemExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FmemPreCompute>()
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
        let pre_compute: &mut FmemPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fmem_impl)
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
        let pre_compute: &mut FmemPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_fmem_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fmem_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pc: &FmemPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FmemPreCompute>()).borrow();
    execute_fmem_core(pc, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fmem_handler<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pc: &FmemPreCompute =
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<FmemPreCompute>()).borrow();
    execute_fmem_core(pc, exec_state);
}

/// Sign-extends the 16-bit immediate (the low 16 bits of the sign-extended
/// 12-bit RISC-V immediate) to a full u32.
#[inline(always)]
fn sign_extend_imm(c: u16) -> u32 {
    c as i16 as i32 as u32
}

#[inline(always)]
unsafe fn execute_fmem_core<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FmemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let a = pre_compute.a as u32;
    let b = pre_compute.b as u32;
    let imm = sign_extend_imm(pre_compute.c);
    let fp_as = pre_compute.fp_as as u32;

    if pre_compute.is_load == 1 {
        // FLW: base in integer register `b` (AS 1), value from AS 2 at
        // `base + imm`, written to FP register `a` (AS fp_as).
        let base_bytes: [u8; 4] = exec_state.vm_read(1, b);
        let base_addr = u32::from_le_bytes(base_bytes);
        let mem_addr = base_addr.wrapping_add(imm) & !3;
        let data: [u8; 4] = exec_state.vm_read(2, mem_addr);
        exec_state.vm_write(fp_as, a, &data);
    } else {
        // FSW: base in integer register `a` (AS 1), value from FP register
        // `b` (AS fp_as), written to AS 2 at `base + imm`.
        let base_bytes: [u8; 4] = exec_state.vm_read(1, a);
        let base_addr = u32::from_le_bytes(base_bytes);
        let mem_addr = base_addr.wrapping_add(imm) & !3;
        let data: [u8; 4] = exec_state.vm_read(fp_as, b);
        exec_state.vm_write(2, mem_addr, &data);
    }

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

impl<A, F> InterpreterMeteredExecutor<F> for FmemExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FmemPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<FmemPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fmem_metered)
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
        let pre_compute: &mut E2PreCompute<FmemPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_fmem_metered_handler)
    }
}

#[cfg(not(feature = "tco"))]
#[create_handler]
#[inline(always)]
unsafe fn execute_fmem_metered<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FmemPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FmemPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fmem_core(&pre_compute.data, exec_state);
}

#[cfg(feature = "tco")]
#[create_handler]
#[inline(always)]
unsafe fn execute_fmem_metered_handler<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FmemPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        std::mem::size_of::<E2PreCompute<FmemPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_fmem_core(&pre_compute.data, exec_state);
}
