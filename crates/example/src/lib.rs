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
    riscv::RV32_REGISTER_AS,
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_rv32f_circuit::Rv32FConfig;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
};

type F = BabyBear;

/// Address space for FP registers — AS 1 (integer registers) for Zfinx.
const FP_AS: u32 = RV32_REGISTER_AS;
fn write_fp_reg(mem: &mut SparseMemoryImage, reg_idx: u32, val: f32) {
    let offset = reg_idx * 4;
    for (i, byte) in val.to_le_bytes().iter().enumerate() {
        mem.insert((FP_AS, offset + i as u32), *byte);
    }
}

fn read_fp_reg(state: &VmState<F>, reg_idx: u32) -> f32 {
    let offset = reg_idx * 4;
    let bytes: [u8; 4] = unsafe { state.memory.read::<u8, 4>(FP_AS, offset) };
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
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
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
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
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

// ════════════════════════════════════════════════════════════════════════════
// Guest program test: builds a real rv32im ELF with inline-asm F instructions,
// transpiles with Zfinx, and executes through the VM.
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod guest_program_tests {
    use std::{fs::read, path::PathBuf};

    use openvm_build::{build_guest_package, get_package, GuestOptions, TargetFilter};
    use openvm_instructions::exe::VmExe;
    use openvm_rv32f_transpiler::Rv32FTranspilerExtension;
    use openvm_rv32im_transpiler::{
        Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
    };
    use openvm_transpiler::{
        elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
    };

    use super::*;

    fn programs_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("programs")
    }

    fn build_and_run_dot3(a: [f32; 3], b: [f32; 3]) -> u32 {
        let manifest_dir = programs_dir();
        let pkg = get_package(&manifest_dir);
        let guest_opts = GuestOptions::default();
        let output_dir = build_guest_package(
            &pkg,
            &guest_opts,
            None,
            &Some(TargetFilter {
                name: "float_ops".to_string(),
                kind: "example".to_string(),
            }),
        )
        .expect("failed to build guest program");

        let elf_path = output_dir.join("float_ops");
        let data = read(&elf_path).expect("failed to read ELF");
        let elf = Elf::decode(&data, MEM_SIZE as u32).expect("failed to decode ELF");

        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32FTranspilerExtension::default()), // Zfinx mode
        )
        .expect("failed to transpile");

        // Provide inputs: each read() call consumes one Vec<F> chunk from input_stream.
        // Order: ax, ay, az, bx, by, bz
        let to_fields =
            |v: u32| -> Vec<F> { v.to_le_bytes().iter().map(|b| F::from_u8(*b)).collect() };
        let mut streams = openvm_circuit::arch::Streams::<F>::default();
        for &v in &[
            a[0].to_bits(),
            a[1].to_bits(),
            a[2].to_bits(),
            b[0].to_bits(),
            b[1].to_bits(),
            b[2].to_bits(),
        ] {
            streams.input_stream.push_back(to_fields(v));
        }

        let config = Rv32FConfig::default();
        let executor = VmExecutor::new(config).expect("failed to create executor");
        let instance = executor.instance(&exe).expect("failed to create instance");
        let _state = instance.execute(streams, None).expect("execution failed");

        // Read the dot3 result from public values slot 0
        let bytes: [u8; 4] = unsafe { _state.memory.read::<u8, 4>(3, 0) };
        u32::from_le_bytes(bytes)
    }

    /// Compute dot3 on the host with the same evaluation order as JoltPhysics:
    /// ((ax*bx) + (ay*by)) + (az*bz)
    fn host_dot3(a: [f32; 3], b: [f32; 3]) -> u32 {
        let m1 = a[0] * b[0];
        let m2 = a[1] * b[1];
        let m3 = a[2] * b[2];
        let s1 = m1 + m2;
        (s1 + m3).to_bits()
    }

    #[test]
    fn test_guest_dot3_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = build_and_run_dot3(a, b);
        assert_eq!(result, host_dot3(a, b), "dot3([1,2,3],[4,5,6]) = 32");
    }

    #[test]
    fn test_guest_dot3_fractions() {
        let a = [0.1f32, 0.2, 0.3];
        let b = [1.0f32, 1.0, 1.0];
        let result = build_and_run_dot3(a, b);
        assert_eq!(result, host_dot3(a, b), "dot3 with fractions");
    }

    #[test]
    fn test_guest_dot3_negative() {
        let a = [-1.0f32, 2.0, -3.0];
        let b = [4.0f32, -5.0, 6.0];
        let result = build_and_run_dot3(a, b);
        assert_eq!(result, host_dot3(a, b), "dot3 with negatives = -32");
    }

    #[test]
    fn test_guest_dot3_physics_like() {
        // Physics-like values: position/velocity magnitudes
        let a = [9.81f32, -0.001, 3.5];
        let b = [0.3048f32, 1000.0, -2.0];
        let result = build_and_run_dot3(a, b);
        assert_eq!(result, host_dot3(a, b), "physics-like dot3");
    }
}
