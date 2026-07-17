//! Example integration tests for the OpenVM RISC-V F extension.
//!
//! These tests construct OpenVM instructions directly (bypassing ELF
//! compilation) and execute them through the VM interpreter, verifying
//! that FMUL.S and FADD.S produce correct IEEE 754 results.

#![allow(unexpected_cfgs, dead_code)]

use openvm_circuit::arch::{VmExecutor, VmState};
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_rv32f_circuit::Rv32FConfig;
use openvm_rv32f_guest::RV32_FP_REGISTER_AS;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
};

type F = BabyBear;

fn write_fp_reg(mem: &mut SparseMemoryImage, reg_idx: u32, val: f32) {
    let offset = reg_idx * 4;
    for (i, byte) in val.to_le_bytes().iter().enumerate() {
        mem.insert((RV32_FP_REGISTER_AS, offset + i as u32), *byte);
    }
}

fn read_fp_reg(state: &VmState<F>, reg_idx: u32) -> f32 {
    let offset = reg_idx * 4;
    let bytes: [u8; 4] = unsafe { state.memory.read::<u8, 4>(RV32_FP_REGISTER_AS, offset) };
    f32::from_le_bytes(bytes)
}

fn terminate() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        ..Default::default()
    }
}

fn fmul_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FMUL_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(RV32_FP_REGISTER_AS),
        F::from_u32(RV32_FP_REGISTER_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fadd_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FADD_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(RV32_FP_REGISTER_AS),
        F::from_u32(RV32_FP_REGISTER_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn run_program(instructions: Vec<Instruction<F>>, init_memory: SparseMemoryImage) -> VmState<F> {
    let program = Program::new_without_debug_infos(&instructions, 0);
    let exe = VmExe::new(program).with_init_memory(init_memory);
    let executor = VmExecutor::new(Rv32FConfig::default()).expect("failed to create executor");
    let instance = executor.instance(&exe).expect("failed to create instance");
    instance.execute(vec![], None).expect("execution failed")
}

#[test]
fn test_fmul_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);

    let state = run_program(vec![fmul_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 12.0);
}

#[test]
fn test_fmul_fractions() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.5);
    write_fp_reg(&mut mem, 2, 2.5);

    let state = run_program(vec![fmul_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 3.75);
}

#[test]
fn test_fmul_negative() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, -3.0);
    write_fp_reg(&mut mem, 2, 7.0);

    let state = run_program(vec![fmul_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), -21.0);
}

#[test]
fn test_fadd_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.5);
    write_fp_reg(&mut mem, 2, 2.5);

    let state = run_program(vec![fadd_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 4.0);
}

#[test]
fn test_fadd_negative() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 10.0);
    write_fp_reg(&mut mem, 2, -3.0);

    let state = run_program(vec![fadd_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 7.0);
}

#[test]
fn test_chained_mul_add() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);
    write_fp_reg(&mut mem, 3, 5.0);

    let state = run_program(
        vec![
            fmul_s(0, 1, 2), // f0 = 12.0
            fadd_s(4, 0, 3), // f4 = 17.0
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 0), 12.0);
    assert_eq!(read_fp_reg(&state, 4), 17.0);
}

#[test]
fn test_dot_product_3() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg(&mut mem, 2, 2.0);
    write_fp_reg(&mut mem, 3, 3.0);
    write_fp_reg(&mut mem, 4, 4.0);
    write_fp_reg(&mut mem, 5, 5.0);
    write_fp_reg(&mut mem, 6, 6.0);

    let state = run_program(
        vec![
            fmul_s(0, 1, 4), // f0 = 4
            fmul_s(7, 2, 5), // f7 = 10
            fadd_s(0, 0, 7), // f0 = 14
            fmul_s(7, 3, 6), // f7 = 18
            fadd_s(0, 0, 7), // f0 = 32
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 0), 32.0);
}

#[test]
fn test_fmul_preserves_inputs() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.5);
    write_fp_reg(&mut mem, 2, 2.5);

    let state = run_program(vec![fmul_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 1), 3.5);
    assert_eq!(read_fp_reg(&state, 2), 2.5);
    assert_eq!(read_fp_reg(&state, 0), 3.5 * 2.5);
}
