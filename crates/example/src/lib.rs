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

fn fdiv_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FDIV_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fmin_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FMIN_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fmax_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FMAX_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn feq_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FEQ_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn flt_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FLT_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fle_s(rd: u32, rs1: u32, rs2: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(FpOpcode::FLE_S.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

/// Unary FP instruction with a single source register (fclass / fcvt).
fn fp_unary(opcode: FpOpcode, rd: u32, rs1: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::ZERO,
        F::from_u32(FP_AS),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fclass_s(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FCLASS_S, rd, rs1)
}

fn fcvt_w_s(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FCVT_W_S, rd, rs1)
}

fn fcvt_wu_s(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FCVT_WU_S, rd, rs1)
}

fn fcvt_s_w(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FCVT_S_W, rd, rs1)
}

fn fcvt_s_wu(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FCVT_S_WU, rd, rs1)
}

fn fsqrt_s(rd: u32, rs1: u32) -> Instruction<F> {
    fp_unary(FpOpcode::FSQRT_S, rd, rs1)
}

/// Ternary FP instruction (fused multiply-add family): rd, rs1, rs2, rs3.
/// Operand layout: a=rd, b=rs1, c=rs2, d=rs3, e=address space.
fn fp_ternary(opcode: FpOpcode, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode.global_opcode_usize()),
        F::from_usize(4 * rd as usize),
        F::from_usize(4 * rs1 as usize),
        F::from_usize(4 * rs2 as usize),
        F::from_usize(4 * rs3 as usize),
        F::from_u32(FP_AS),
        F::ZERO,
        F::ZERO,
    )
}

fn fmadd_s(rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    fp_ternary(FpOpcode::FMADD_S, rd, rs1, rs2, rs3)
}

fn fmsub_s(rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    fp_ternary(FpOpcode::FMSUB_S, rd, rs1, rs2, rs3)
}

fn fnmsub_s(rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    fp_ternary(FpOpcode::FNMSUB_S, rd, rs1, rs2, rs3)
}

fn fnmadd_s(rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    fp_ternary(FpOpcode::FNMADD_S, rd, rs1, rs2, rs3)
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
// FMIN.S / FMAX.S execution tests
// ════════════════════════════════════════════════════════════════════════════

const CANONICAL_NAN: u32 = 0x7FC0_0000;

fn write_fp_reg_bits(mem: &mut SparseMemoryImage, reg_idx: u32, bits: u32) {
    let offset = reg_idx * 4;
    for (i, byte) in bits.to_le_bytes().iter().enumerate() {
        mem.insert((FP_AS, offset + i as u32), *byte);
    }
}

fn read_fp_reg_bits(state: &VmState<F>, reg_idx: u32) -> u32 {
    let offset = reg_idx * 4;
    let bytes: [u8; 4] = unsafe { state.memory.read::<u8, 4>(FP_AS, offset) };
    u32::from_le_bytes(bytes)
}

#[test]
fn test_fmin_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);

    let state = run_program(vec![fmin_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 3.0);
}

#[test]
fn test_fmax_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);

    let state = run_program(vec![fmax_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 4.0);
}

#[test]
fn test_fmin_negative() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, -3.0);
    write_fp_reg(&mut mem, 2, 7.0);

    let state = run_program(vec![fmin_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), -3.0);
}

#[test]
fn test_fmax_negative() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, -3.0);
    write_fp_reg(&mut mem, 2, -7.0);

    let state = run_program(vec![fmax_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), -3.0);
}

#[test]
fn test_fmin_nan_first() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, CANONICAL_NAN);
    write_fp_reg(&mut mem, 2, 2.0);

    let state = run_program(vec![fmin_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 2.0);
}

#[test]
fn test_fmax_nan_second() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.5);
    write_fp_reg_bits(&mut mem, 2, CANONICAL_NAN);

    let state = run_program(vec![fmax_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 1.5);
}

#[test]
fn test_fminmax_both_nan() {
    let snan = 0x7FA0_0001u32;
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, CANONICAL_NAN);
    write_fp_reg_bits(&mut mem, 2, snan);

    let state = run_program(
        vec![
            fmin_s(0, 1, 2), // f0 = canonical NaN
            fmax_s(3, 1, 2), // f3 = canonical NaN
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), CANONICAL_NAN);
    assert_eq!(read_fp_reg_bits(&state, 3), CANONICAL_NAN);
}

#[test]
fn test_fmin_signed_zero() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x0000_0000); // +0.0
    write_fp_reg_bits(&mut mem, 2, 0x8000_0000); // -0.0

    let state = run_program(vec![fmin_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 0x8000_0000); // -0.0
}

#[test]
fn test_fmax_signed_zero() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x8000_0000); // -0.0
    write_fp_reg_bits(&mut mem, 2, 0x0000_0000); // +0.0

    let state = run_program(vec![fmax_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 0x0000_0000); // +0.0
}

#[test]
fn test_fminmax_chained() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 5.0);
    write_fp_reg(&mut mem, 2, -2.0);
    write_fp_reg(&mut mem, 3, 3.5);

    let state = run_program(
        vec![
            fmin_s(0, 1, 2), // f0 = -2.0
            fmax_s(4, 0, 3), // f4 = 3.5
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 0), -2.0);
    assert_eq!(read_fp_reg(&state, 4), 3.5);
}

// ════════════════════════════════════════════════════════════════════════════
// FEQ.S / FLT.S / FLE.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_feq_equal() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 2.0);
    write_fp_reg(&mut mem, 2, 2.0);

    let state = run_program(vec![feq_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 1);
}

#[test]
fn test_feq_not_equal() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 2.0);
    write_fp_reg(&mut mem, 2, 3.0);

    let state = run_program(vec![feq_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
}

#[test]
fn test_flt_true() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, -1.0);
    write_fp_reg(&mut mem, 2, 1.0);

    let state = run_program(vec![flt_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 1);
}

#[test]
fn test_flt_false_when_equal() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg(&mut mem, 2, 1.0);

    let state = run_program(vec![flt_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
}

#[test]
fn test_fle_true_when_equal() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg(&mut mem, 2, 1.0);

    let state = run_program(vec![fle_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 1);
}

#[test]
fn test_fle_false() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg(&mut mem, 2, -1.0);

    let state = run_program(vec![fle_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
}

#[test]
fn test_fcmp_signed_zero() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x8000_0000); // -0.0
    write_fp_reg_bits(&mut mem, 2, 0x0000_0000); // +0.0

    let state = run_program(
        vec![
            feq_s(0, 1, 2), // -0.0 == +0.0 -> 1
            flt_s(3, 1, 2), // -0.0 < +0.0  -> 0
            fle_s(4, 1, 2), // -0.0 <= +0.0 -> 1
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 1);
    assert_eq!(read_fp_reg_bits(&state, 3), 0);
    assert_eq!(read_fp_reg_bits(&state, 4), 1);
}

#[test]
fn test_fcmp_nan_is_false() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, CANONICAL_NAN);
    write_fp_reg(&mut mem, 2, 1.0);

    let state = run_program(
        vec![
            feq_s(0, 1, 2), // NaN == 1.0 -> 0
            flt_s(3, 1, 2), // NaN < 1.0  -> 0
            fle_s(4, 2, 1), // 1.0 <= NaN -> 0
            feq_s(5, 1, 1), // NaN == NaN -> 0
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
    assert_eq!(read_fp_reg_bits(&state, 3), 0);
    assert_eq!(read_fp_reg_bits(&state, 4), 0);
    assert_eq!(read_fp_reg_bits(&state, 5), 0);
}

#[test]
fn test_fcmp_infinity() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, f32::INFINITY);
    write_fp_reg(&mut mem, 2, 1.0);

    let state = run_program(
        vec![
            flt_s(0, 1, 2), // inf < 1.0  -> 0
            fle_s(3, 2, 1), // 1.0 <= inf -> 1
            feq_s(4, 1, 1), // inf == inf -> 1
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
    assert_eq!(read_fp_reg_bits(&state, 3), 1);
    assert_eq!(read_fp_reg_bits(&state, 4), 1);
}

#[test]
fn test_fcmp_chained() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 5.0);
    write_fp_reg(&mut mem, 2, -2.0);
    write_fp_reg(&mut mem, 3, -2.0);

    let state = run_program(
        vec![
            flt_s(0, 1, 2), // 5.0 < -2.0   -> 0
            fle_s(4, 2, 3), // -2.0 <= -2.0 -> 1
            feq_s(5, 2, 3), // -2.0 == -2.0 -> 1
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 0);
    assert_eq!(read_fp_reg_bits(&state, 4), 1);
    assert_eq!(read_fp_reg_bits(&state, 5), 1);
}

// ════════════════════════════════════════════════════════════════════════════
// FCLASS.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fclass_categories() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.5); // +normal
    write_fp_reg(&mut mem, 2, -1.5); // -normal
    write_fp_reg(&mut mem, 3, f32::INFINITY);
    write_fp_reg(&mut mem, 4, f32::NEG_INFINITY);
    write_fp_reg_bits(&mut mem, 5, 0x0000_0000); // +0
    write_fp_reg_bits(&mut mem, 6, 0x8000_0000); // -0
    write_fp_reg_bits(&mut mem, 7, 0x0000_0001); // +subnormal
    write_fp_reg_bits(&mut mem, 8, 0x8000_0001); // -subnormal
    write_fp_reg_bits(&mut mem, 9, 0x7FC0_0000); // qNaN
    write_fp_reg_bits(&mut mem, 10, 0x7FA0_0001); // sNaN

    let state = run_program(
        vec![
            fclass_s(11, 1),
            fclass_s(12, 2),
            fclass_s(13, 3),
            fclass_s(14, 4),
            fclass_s(15, 5),
            fclass_s(16, 6),
            fclass_s(17, 7),
            fclass_s(18, 8),
            fclass_s(19, 9),
            fclass_s(20, 10),
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 11), 1 << 6);
    assert_eq!(read_fp_reg_bits(&state, 12), 1 << 1);
    assert_eq!(read_fp_reg_bits(&state, 13), 1 << 7);
    assert_eq!(read_fp_reg_bits(&state, 14), 1 << 0);
    assert_eq!(read_fp_reg_bits(&state, 15), 1 << 4);
    assert_eq!(read_fp_reg_bits(&state, 16), 1 << 3);
    assert_eq!(read_fp_reg_bits(&state, 17), 1 << 5);
    assert_eq!(read_fp_reg_bits(&state, 18), 1 << 2);
    assert_eq!(read_fp_reg_bits(&state, 19), 1 << 9);
    assert_eq!(read_fp_reg_bits(&state, 20), 1 << 8);
}

// ════════════════════════════════════════════════════════════════════════════
// FCVT.W.S / FCVT.WU.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fcvt_w_s_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.7);
    write_fp_reg(&mut mem, 2, -3.7);
    write_fp_reg(&mut mem, 3, 2.5); // ties to even -> 2

    let state = run_program(
        vec![fcvt_w_s(0, 1), fcvt_w_s(4, 2), fcvt_w_s(5, 3), terminate()],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 4);
    assert_eq!(read_fp_reg_bits(&state, 4), (-4i32) as u32);
    assert_eq!(read_fp_reg_bits(&state, 5), 2);
}

#[test]
fn test_fcvt_w_s_special() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, CANONICAL_NAN);
    write_fp_reg(&mut mem, 2, f32::NEG_INFINITY);
    write_fp_reg(&mut mem, 3, 3e9); // out of i32 range

    let state = run_program(
        vec![
            fcvt_w_s(0, 1), // NaN -> INT32_MAX
            fcvt_w_s(4, 2), // -inf -> INT32_MIN
            fcvt_w_s(5, 3), // saturate -> INT32_MAX
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), i32::MAX as u32);
    assert_eq!(read_fp_reg_bits(&state, 4), i32::MIN as u32);
    assert_eq!(read_fp_reg_bits(&state, 5), i32::MAX as u32);
}

#[test]
fn test_fcvt_wu_s() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.7);
    write_fp_reg(&mut mem, 2, -3.7);
    write_fp_reg_bits(&mut mem, 3, CANONICAL_NAN);
    write_fp_reg(&mut mem, 4, 3e9); // in u32 range

    let state = run_program(
        vec![
            fcvt_wu_s(0, 1),
            fcvt_wu_s(5, 2),
            fcvt_wu_s(6, 3),
            fcvt_wu_s(7, 4),
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 0), 4);
    assert_eq!(read_fp_reg_bits(&state, 5), 0); // negative -> 0
    assert_eq!(read_fp_reg_bits(&state, 6), u32::MAX); // NaN -> UINT32_MAX
    assert_eq!(read_fp_reg_bits(&state, 7), 3000000000);
}

// ════════════════════════════════════════════════════════════════════════════
// FCVT.S.W / FCVT.S.WU execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fcvt_s_w() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 42);
    write_fp_reg_bits(&mut mem, 2, (-42i32) as u32);
    write_fp_reg_bits(&mut mem, 3, 0x0100_0001); // 2^24+1 -> 2^24 (ties to even)

    let state = run_program(
        vec![fcvt_s_w(0, 1), fcvt_s_w(4, 2), fcvt_s_w(5, 3), terminate()],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 0), 42.0);
    assert_eq!(read_fp_reg(&state, 4), -42.0);
    assert_eq!(read_fp_reg(&state, 5), 16777216.0);
}

#[test]
fn test_fcvt_s_wu() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 42);
    write_fp_reg_bits(&mut mem, 2, u32::MAX); // rounds to 2^32

    let state = run_program(vec![fcvt_s_wu(0, 1), fcvt_s_wu(4, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 42.0);
    assert_eq!(read_fp_reg(&state, 4), 4294967296.0);
}

#[test]
fn test_fcvt_roundtrip() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 17.0);

    let state = run_program(
        vec![
            fcvt_w_s(2, 1), // f2 = 17 (as int)
            fcvt_s_w(3, 2), // f3 = 17.0 (as float)
            feq_s(4, 1, 3), // 17.0 == 17.0 -> 1
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 2), 17);
    assert_eq!(read_fp_reg(&state, 3), 17.0);
    assert_eq!(read_fp_reg_bits(&state, 4), 1);
}

// ════════════════════════════════════════════════════════════════════════════
// FMADD.S / FMSUB.S / FNMSUB.S / FNMADD.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fmadd_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);
    write_fp_reg(&mut mem, 3, 1.0);

    let state = run_program(vec![fmadd_s(0, 1, 2, 3), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 13.0);
}

#[test]
fn test_fmsub_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);
    write_fp_reg(&mut mem, 3, 1.0);

    let state = run_program(vec![fmsub_s(0, 1, 2, 3), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 11.0);
}

#[test]
fn test_fnmsub_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);
    write_fp_reg(&mut mem, 3, 1.0);

    let state = run_program(vec![fnmsub_s(0, 1, 2, 3), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), -11.0); // -(3*4) + 1
}

#[test]
fn test_fnmadd_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 3.0);
    write_fp_reg(&mut mem, 2, 4.0);
    write_fp_reg(&mut mem, 3, 1.0);

    let state = run_program(vec![fnmadd_s(0, 1, 2, 3), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), -13.0); // -(3*4) - 1
}

#[test]
fn test_fma_fused_single_rounding() {
    // (1 + 2^-12)^2 - 1 = 2^-11 + 2^-24: only survives with fused rounding.
    let mut mem = SparseMemoryImage::new();
    let a = 1.0f32 + 2f32.powi(-12);
    write_fp_reg(&mut mem, 1, a);
    write_fp_reg(&mut mem, 2, -1.0);

    let state = run_program(vec![fmadd_s(0, 1, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 2f32.powi(-11) + 2f32.powi(-24));
}

#[test]
fn test_fma_inf_and_nan() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, f32::INFINITY);
    write_fp_reg(&mut mem, 2, 2.0);
    write_fp_reg(&mut mem, 3, 1.0);
    write_fp_reg(&mut mem, 4, f32::NEG_INFINITY);
    write_fp_reg_bits(&mut mem, 5, CANONICAL_NAN);

    let state = run_program(
        vec![
            fmadd_s(6, 1, 2, 3),   // inf * 2 + 1 = inf
            fmadd_s(7, 1, 2, 4),   // inf * 2 + (-inf) = canonical NaN
            fmadd_s(8, 1, 3, 5),   // inf * 1 + NaN = canonical NaN
            fmsub_s(9, 2, 2, 3),   // 2*2 - 1 = 3.0
            fnmsub_s(10, 1, 2, 4), // -(inf * 2) + (-inf) = -inf
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 6), f32::INFINITY);
    assert_eq!(read_fp_reg_bits(&state, 7), CANONICAL_NAN);
    assert_eq!(read_fp_reg_bits(&state, 8), CANONICAL_NAN);
    assert_eq!(read_fp_reg(&state, 9), 3.0);
    assert_eq!(read_fp_reg(&state, 10), f32::NEG_INFINITY);
}

#[test]
fn test_fma_zero_operand() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x0000_0000); // +0.0
    write_fp_reg(&mut mem, 2, f32::INFINITY);
    write_fp_reg(&mut mem, 3, 1.0);

    let state = run_program(vec![fmadd_s(0, 1, 2, 3), terminate()], mem);

    // 0 * inf is invalid -> canonical NaN
    assert_eq!(read_fp_reg_bits(&state, 0), CANONICAL_NAN);
}

// ════════════════════════════════════════════════════════════════════════════
// FDIV.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fdiv_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 12.0);
    write_fp_reg(&mut mem, 2, 4.0);

    let state = run_program(vec![fdiv_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 3.0);
}

#[test]
fn test_fdiv_fraction() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg(&mut mem, 2, 3.0);

    let state = run_program(vec![fdiv_s(0, 1, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 1.0 / 3.0);
}

#[test]
fn test_fdiv_special() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 1.0);
    write_fp_reg_bits(&mut mem, 2, 0x0000_0000); // +0.0
    write_fp_reg(&mut mem, 3, f32::INFINITY);

    let state = run_program(
        vec![
            fdiv_s(4, 1, 2), // 1.0 / +0.0 = +inf
            fdiv_s(5, 2, 2), // 0 / 0 = canonical NaN
            fdiv_s(6, 3, 3), // inf / inf = canonical NaN
            fdiv_s(7, 1, 3), // 1.0 / inf = +0.0
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 4), f32::INFINITY);
    assert_eq!(read_fp_reg_bits(&state, 5), CANONICAL_NAN);
    assert_eq!(read_fp_reg_bits(&state, 6), CANONICAL_NAN);
    assert_eq!(read_fp_reg_bits(&state, 7), 0);
}

#[test]
fn test_fdiv_negative_sign() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, -7.0);
    write_fp_reg(&mut mem, 2, 2.0);

    let state = run_program(
        vec![
            fdiv_s(0, 1, 2), // -7/2 = -3.5
            fdiv_s(3, 2, 1), // 2/-7 = -0.2857...
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg(&state, 0), -3.5);
    assert_eq!(read_fp_reg(&state, 3), 2.0 / -7.0);
}

// ════════════════════════════════════════════════════════════════════════════
// FSQRT.S execution tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fsqrt_basic() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg(&mut mem, 1, 4.0);
    write_fp_reg(&mut mem, 2, 2.0);

    let state = run_program(vec![fsqrt_s(0, 1), fsqrt_s(3, 2), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), 2.0);
    assert_eq!(read_fp_reg(&state, 3), f32::sqrt(2.0));
}

#[test]
fn test_fsqrt_special() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x0000_0000); // +0.0
    write_fp_reg_bits(&mut mem, 2, 0x8000_0000); // -0.0
    write_fp_reg(&mut mem, 3, f32::INFINITY);
    write_fp_reg(&mut mem, 4, -1.0);

    let state = run_program(
        vec![
            fsqrt_s(5, 1), // sqrt(+0.0) = +0.0
            fsqrt_s(6, 2), // sqrt(-0.0) = -0.0
            fsqrt_s(7, 3), // sqrt(+inf) = +inf
            fsqrt_s(8, 4), // sqrt(-1.0) = canonical NaN
            terminate(),
        ],
        mem,
    );

    assert_eq!(read_fp_reg_bits(&state, 5), 0);
    assert_eq!(read_fp_reg_bits(&state, 6), 0x8000_0000);
    assert_eq!(read_fp_reg(&state, 7), f32::INFINITY);
    assert_eq!(read_fp_reg_bits(&state, 8), CANONICAL_NAN);
}

#[test]
fn test_fsqrt_subnormal_input() {
    let mut mem = SparseMemoryImage::new();
    write_fp_reg_bits(&mut mem, 1, 0x0000_0001); // min subnormal

    let state = run_program(vec![fsqrt_s(0, 1), terminate()], mem);

    assert_eq!(read_fp_reg(&state, 0), f32::from_bits(0x0000_0001).sqrt());
}

// ════════════════════════════════════════════════════════════════════════════
// Execution-level fuzz: random operands through the VM interpreter, verified
// against the native semantics helpers from the circuit crate.
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod fuzz_tests {
    use openvm_rv32f_circuit::{
        fclass::fclass_s as fclass_native,
        fcmp::{fcmp_result, FcmpOp},
        fcvt_f2i::fcvt_f2i_result,
        fcvt_i2f::fcvt_i2f_result,
        fdiv::fdiv_s as fdiv_native,
        fma::{fmadd_result, FmaddOp},
        fminmax::fminmax_result,
        fsqrt::fsqrt_s as fsqrt_native,
    };

    use super::*;

    struct XorShift64 {
        state: u64,
    }
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(1),
            }
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x as u32
        }
    }

    const SPECIALS: [u32; 12] = [
        0x0000_0000, // +0.0 / 0
        0x8000_0000, // -0.0 / i32::MIN
        0x7FC0_0000, // qNaN
        0x7FA0_0001, // sNaN
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x0000_0001, // min subnormal / 1
        0x8000_0001, // -min subnormal
        0x007F_FFFF, // max subnormal
        0x0080_0000, // min normal
        0x7F7F_FFFF, // max finite
        0xFF7F_FFFF, // -max finite
    ];

    fn rand_bits(rng: &mut XorShift64) -> u32 {
        if rng.next_u32().is_multiple_of(3) {
            SPECIALS[(rng.next_u32() as usize) % SPECIALS.len()]
        } else {
            rng.next_u32()
        }
    }

    /// Builds one program of binary FP ops (inputs in regs 1..30, rd aliases
    /// rs1 so results stay readable; max 15 cases).
    fn build_binary_chunk(
        chunk: &[(u32, u32)],
        make: &mut impl FnMut(u32, u32, u32, usize) -> Instruction<F>,
    ) -> (Vec<Instruction<F>>, SparseMemoryImage) {
        let mut mem = SparseMemoryImage::new();
        let mut instrs = Vec::new();
        for (i, &(a, b)) in chunk.iter().enumerate() {
            let rs1 = 2 * i as u32 + 1;
            let rs2 = rs1 + 1;
            write_fp_reg_bits(&mut mem, rs1, a);
            write_fp_reg_bits(&mut mem, rs2, b);
            instrs.push(make(rs1, rs1, rs2, i));
        }
        instrs.push(terminate());
        (instrs, mem)
    }

    /// Builds one program of unary FP ops (rd aliases rs1; max 30 cases).
    fn build_unary_chunk(
        chunk: &[u32],
        make: &mut impl FnMut(u32, u32, usize) -> Instruction<F>,
    ) -> (Vec<Instruction<F>>, SparseMemoryImage) {
        let mut mem = SparseMemoryImage::new();
        let mut instrs = Vec::new();
        for (i, &a) in chunk.iter().enumerate() {
            let rs1 = i as u32 + 1;
            write_fp_reg_bits(&mut mem, rs1, a);
            instrs.push(make(rs1, rs1, i));
        }
        instrs.push(terminate());
        (instrs, mem)
    }

    /// Runs binary FP ops through the VM in chunks of 15.
    fn fuzz_binary(
        cases: &[(u32, u32)],
        mut make: impl FnMut(u32, u32, u32, usize) -> Instruction<F>,
        mut expect: impl FnMut(u32, u32, usize) -> u32,
    ) {
        for (chunk_idx, chunk) in cases.chunks(15).enumerate() {
            let (instrs, mem) = build_binary_chunk(chunk, &mut make);
            let state = run_program(instrs, mem);
            for (i, &(a, b)) in chunk.iter().enumerate() {
                let rs1 = 2 * i as u32 + 1;
                let got = read_fp_reg_bits(&state, rs1);
                let want = expect(a, b, i);
                assert_eq!(
                    got, want,
                    "chunk {chunk_idx} case {i}: a={a:#010x} b={b:#010x}"
                );
            }
        }
    }

    /// Runs unary FP ops through the VM in chunks of 30 (rd aliases rs1).
    fn fuzz_unary(
        cases: &[u32],
        mut make: impl FnMut(u32, u32, usize) -> Instruction<F>,
        mut expect: impl FnMut(u32, usize) -> u32,
    ) {
        for (chunk_idx, chunk) in cases.chunks(30).enumerate() {
            let (instrs, mem) = build_unary_chunk(chunk, &mut make);
            let state = run_program(instrs, mem);
            for (i, &a) in chunk.iter().enumerate() {
                let rs1 = i as u32 + 1;
                let got = read_fp_reg_bits(&state, rs1);
                let want = expect(a, i);
                assert_eq!(got, want, "chunk {chunk_idx} case {i}: a={a:#010x}");
            }
        }
    }

    #[test]
    fn fuzz_exec_fminmax() {
        let mut rng = XorShift64::new(0xA11C_E001);
        let cases: Vec<(u32, u32)> = (0..150)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        fuzz_binary(
            &cases,
            |rd, rs1, rs2, i| {
                if i % 2 == 0 {
                    fmin_s(rd, rs1, rs2)
                } else {
                    fmax_s(rd, rs1, rs2)
                }
            },
            |a, b, i| fminmax_result(a, b, i % 2 == 0),
        );
    }

    #[test]
    fn fuzz_exec_fcmp() {
        const OPS: [FcmpOp; 3] = [FcmpOp::Le, FcmpOp::Lt, FcmpOp::Eq];
        let mut rng = XorShift64::new(0xA11C_E002);
        let cases: Vec<(u32, u32)> = (0..150)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        fuzz_binary(
            &cases,
            |rd, rs1, rs2, i| match OPS[i % 3] {
                FcmpOp::Le => fle_s(rd, rs1, rs2),
                FcmpOp::Lt => flt_s(rd, rs1, rs2),
                FcmpOp::Eq => feq_s(rd, rs1, rs2),
            },
            |a, b, i| fcmp_result(a, b, OPS[i % 3]),
        );
    }

    #[test]
    fn fuzz_exec_fclass() {
        let mut rng = XorShift64::new(0xA11C_E003);
        let cases: Vec<u32> = (0..300).map(|_| rand_bits(&mut rng)).collect();
        fuzz_unary(
            &cases,
            |rd, rs1, _| fclass_s(rd, rs1),
            |a, _| fclass_native(a),
        );
    }

    #[test]
    fn fuzz_exec_fcvt_f2i() {
        let mut rng = XorShift64::new(0xA11C_E004);
        let cases: Vec<u32> = (0..150).map(|_| rand_bits(&mut rng)).collect();
        fuzz_unary(
            &cases,
            |rd, rs1, i| {
                if i % 2 == 0 {
                    fcvt_w_s(rd, rs1)
                } else {
                    fcvt_wu_s(rd, rs1)
                }
            },
            |a, i| fcvt_f2i_result(a, i % 2 == 1),
        );
    }

    /// Builds one program of ternary FP ops (inputs in regs 1..30, rd aliases
    /// rs1 so results stay readable; max 10 cases).
    fn build_ternary_chunk(
        chunk: &[(u32, u32, u32)],
        make: &mut impl FnMut(u32, u32, u32, u32, usize) -> Instruction<F>,
    ) -> (Vec<Instruction<F>>, SparseMemoryImage) {
        let mut mem = SparseMemoryImage::new();
        let mut instrs = Vec::new();
        for (i, &(a, b, c)) in chunk.iter().enumerate() {
            let rs1 = 3 * i as u32 + 1;
            let rs2 = rs1 + 1;
            let rs3 = rs1 + 2;
            write_fp_reg_bits(&mut mem, rs1, a);
            write_fp_reg_bits(&mut mem, rs2, b);
            write_fp_reg_bits(&mut mem, rs3, c);
            instrs.push(make(rs1, rs1, rs2, rs3, i));
        }
        instrs.push(terminate());
        (instrs, mem)
    }

    #[test]
    fn fuzz_exec_fma() {
        const OPS: [FmaddOp; 4] = [FmaddOp::Madd, FmaddOp::Msub, FmaddOp::Nmsub, FmaddOp::Nmadd];
        let mut rng = XorShift64::new(0xA11C_E006);
        let cases: Vec<(u32, u32, u32)> = (0..100)
            .map(|_| {
                (
                    rand_bits(&mut rng),
                    rand_bits(&mut rng),
                    rand_bits(&mut rng),
                )
            })
            .collect();
        for (chunk_idx, chunk) in cases.chunks(10).enumerate() {
            let (instrs, mem) =
                build_ternary_chunk(chunk, &mut |rd, rs1, rs2, rs3, i| match OPS[i % 4] {
                    FmaddOp::Madd => fmadd_s(rd, rs1, rs2, rs3),
                    FmaddOp::Msub => fmsub_s(rd, rs1, rs2, rs3),
                    FmaddOp::Nmsub => fnmsub_s(rd, rs1, rs2, rs3),
                    FmaddOp::Nmadd => fnmadd_s(rd, rs1, rs2, rs3),
                });
            let state = run_program(instrs, mem);
            for (i, &(a, b, c)) in chunk.iter().enumerate() {
                let rs1 = 3 * i as u32 + 1;
                let got = read_fp_reg_bits(&state, rs1);
                let want = fmadd_result(a, b, c, OPS[i % 4]);
                assert_eq!(
                    got,
                    want,
                    "chunk {chunk_idx} case {i}: {:?} a={a:#010x} b={b:#010x} c={c:#010x}",
                    OPS[i % 4]
                );
            }
        }
    }

    #[test]
    fn fuzz_exec_fdiv() {
        let mut rng = XorShift64::new(0xA11C_E007);
        let cases: Vec<(u32, u32)> = (0..150)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        fuzz_binary(
            &cases,
            |rd, rs1, rs2, _| fdiv_s(rd, rs1, rs2),
            |a, b, _| fdiv_native(a, b),
        );
    }

    #[test]
    fn fuzz_exec_fsqrt() {
        let mut rng = XorShift64::new(0xA11C_E008);
        let cases: Vec<u32> = (0..300).map(|_| rand_bits(&mut rng)).collect();
        fuzz_unary(
            &cases,
            |rd, rs1, _| fsqrt_s(rd, rs1),
            |a, _| fsqrt_native(a),
        );
    }

    #[test]
    fn fuzz_exec_fcvt_i2f() {
        let mut rng = XorShift64::new(0xA11C_E005);
        let cases: Vec<u32> = (0..150).map(|_| rand_bits(&mut rng)).collect();
        fuzz_unary(
            &cases,
            |rd, rs1, i| {
                if i % 2 == 0 {
                    fcvt_s_w(rd, rs1)
                } else {
                    fcvt_s_wu(rd, rs1)
                }
            },
            |a, i| fcvt_i2f_result(a, i % 2 == 1),
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Proving-level fuzz: random programs through air_test (constraint check).
    // ════════════════════════════════════════════════════════════════════════

    #[cfg(feature = "proving")]
    fn prove_chunk(instrs: Vec<Instruction<F>>, mem: SparseMemoryImage) {
        use openvm_circuit::utils::air_test;
        use openvm_rv32f_circuit::Rv32FCpuBuilder;

        let program = Program::new_without_debug_infos(&instrs, 0);
        let exe = VmExe::new(program).with_init_memory(mem);
        air_test(Rv32FCpuBuilder, Rv32FConfig::default(), exe);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fminmax() {
        let mut rng = XorShift64::new(0xB0B0_0001);
        let cases: Vec<(u32, u32)> = (0..15)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        let (instrs, mem) = build_binary_chunk(&cases, &mut |rd, rs1, rs2, i| {
            if i % 2 == 0 {
                fmin_s(rd, rs1, rs2)
            } else {
                fmax_s(rd, rs1, rs2)
            }
        });
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fcmp() {
        const OPS: [FcmpOp; 3] = [FcmpOp::Le, FcmpOp::Lt, FcmpOp::Eq];
        let mut rng = XorShift64::new(0xB0B0_0002);
        let cases: Vec<(u32, u32)> = (0..15)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        let (instrs, mem) = build_binary_chunk(&cases, &mut |rd, rs1, rs2, i| match OPS[i % 3] {
            FcmpOp::Le => fle_s(rd, rs1, rs2),
            FcmpOp::Lt => flt_s(rd, rs1, rs2),
            FcmpOp::Eq => feq_s(rd, rs1, rs2),
        });
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fclass() {
        let mut rng = XorShift64::new(0xB0B0_0003);
        let cases: Vec<u32> = (0..30).map(|_| rand_bits(&mut rng)).collect();
        let (instrs, mem) = build_unary_chunk(&cases, &mut |rd, rs1, _| fclass_s(rd, rs1));
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fcvt_f2i() {
        let mut rng = XorShift64::new(0xB0B0_0004);
        let cases: Vec<u32> = (0..30).map(|_| rand_bits(&mut rng)).collect();
        let (instrs, mem) = build_unary_chunk(&cases, &mut |rd, rs1, i| {
            if i % 2 == 0 {
                fcvt_w_s(rd, rs1)
            } else {
                fcvt_wu_s(rd, rs1)
            }
        });
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fcvt_i2f() {
        let mut rng = XorShift64::new(0xB0B0_0005);
        let cases: Vec<u32> = (0..30).map(|_| rand_bits(&mut rng)).collect();
        let (instrs, mem) = build_unary_chunk(&cases, &mut |rd, rs1, i| {
            if i % 2 == 0 {
                fcvt_s_w(rd, rs1)
            } else {
                fcvt_s_wu(rd, rs1)
            }
        });
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fma() {
        const OPS: [FmaddOp; 4] = [FmaddOp::Madd, FmaddOp::Msub, FmaddOp::Nmsub, FmaddOp::Nmadd];
        let mut rng = XorShift64::new(0xB0B0_0006);
        let cases: Vec<(u32, u32, u32)> = (0..10)
            .map(|_| {
                (
                    rand_bits(&mut rng),
                    rand_bits(&mut rng),
                    rand_bits(&mut rng),
                )
            })
            .collect();
        let (instrs, mem) =
            build_ternary_chunk(&cases, &mut |rd, rs1, rs2, rs3, i| match OPS[i % 4] {
                FmaddOp::Madd => fmadd_s(rd, rs1, rs2, rs3),
                FmaddOp::Msub => fmsub_s(rd, rs1, rs2, rs3),
                FmaddOp::Nmsub => fnmsub_s(rd, rs1, rs2, rs3),
                FmaddOp::Nmadd => fnmadd_s(rd, rs1, rs2, rs3),
            });
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fdiv() {
        let mut rng = XorShift64::new(0xB0B0_0007);
        let cases: Vec<(u32, u32)> = (0..15)
            .map(|_| (rand_bits(&mut rng), rand_bits(&mut rng)))
            .collect();
        let (instrs, mem) = build_binary_chunk(&cases, &mut |rd, rs1, rs2, _| fdiv_s(rd, rs1, rs2));
        prove_chunk(instrs, mem);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn fuzz_prove_fsqrt() {
        let mut rng = XorShift64::new(0xB0B0_0008);
        let cases: Vec<u32> = (0..30).map(|_| rand_bits(&mut rng)).collect();
        let (instrs, mem) = build_unary_chunk(&cases, &mut |rd, rs1, _| fsqrt_s(rd, rs1));
        prove_chunk(instrs, mem);
    }
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

    /// Helper: build a guest example, transpile, and run air_test.
    #[cfg(feature = "proving")]
    fn prove_guest_example(example_name: &str) {
        use openvm_circuit::utils::air_test;
        use openvm_rv32f_circuit::Rv32FCpuBuilder;

        let manifest_dir = programs_dir();
        let pkg = get_package(&manifest_dir);
        let guest_opts = GuestOptions::default();
        let output_dir = build_guest_package(
            &pkg,
            &guest_opts,
            None,
            &Some(TargetFilter {
                name: example_name.to_string(),
                kind: "example".to_string(),
            }),
        )
        .unwrap_or_else(|e| panic!("failed to build guest program {example_name}: {e:?}"));

        let elf_path = output_dir.join(example_name);
        let data = read(&elf_path).expect("failed to read ELF");
        let elf = Elf::decode(&data, MEM_SIZE as u32).expect("failed to decode ELF");

        let exe = VmExe::from_elf(
            elf,
            Transpiler::<F>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Rv32FTranspilerExtension::default()),
        )
        .expect("failed to transpile");

        let config = Rv32FConfig::default();
        air_test(Rv32FCpuBuilder, config, exe);
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fmul_only() {
        prove_guest_example("fmul_only");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fadd_only() {
        prove_guest_example("fadd_only");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_dot3() {
        prove_guest_example("dot3_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fminmax() {
        prove_guest_example("fminmax_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fcmp() {
        prove_guest_example("fcmp_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fcvt() {
        prove_guest_example("fcvt_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fma() {
        prove_guest_example("fma_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fdiv() {
        prove_guest_example("fdiv_prove");
    }

    #[cfg(feature = "proving")]
    #[test]
    fn test_prove_fsqrt() {
        prove_guest_example("fsqrt_prove");
    }
}

/// Chip-level proving tests: hand-built programs run through air_test.
#[cfg(feature = "proving")]
#[cfg(test)]
mod chip_proving_tests {
    use openvm_circuit::utils::air_test;
    use openvm_rv32f_circuit::Rv32FCpuBuilder;

    use super::*;

    fn prove_program(instructions: Vec<Instruction<F>>, init_memory: SparseMemoryImage) {
        let program = Program::new_without_debug_infos(&instructions, 0);
        let exe = VmExe::new(program).with_init_memory(init_memory);
        air_test(Rv32FCpuBuilder, Rv32FConfig::default(), exe);
    }

    #[test]
    fn test_prove_fcvt_s_w_single() {
        let mut mem = SparseMemoryImage::new();
        write_fp_reg_bits(&mut mem, 1, (-42i32) as u32);
        prove_program(vec![fcvt_s_w(0, 1), terminate()], mem);
    }

    #[test]
    fn test_prove_fcvt_s_wu_single() {
        let mut mem = SparseMemoryImage::new();
        write_fp_reg_bits(&mut mem, 1, 42);
        prove_program(vec![fcvt_s_wu(0, 1), terminate()], mem);
    }

    #[test]
    fn test_prove_fclass_single() {
        let mut mem = SparseMemoryImage::new();
        write_fp_reg(&mut mem, 1, 1.5);
        prove_program(vec![fclass_s(0, 1), terminate()], mem);
    }

    #[test]
    fn test_prove_fcvt_w_s_single() {
        let mut mem = SparseMemoryImage::new();
        write_fp_reg(&mut mem, 1, 3.7);
        prove_program(vec![fcvt_w_s(0, 1), terminate()], mem);
    }

    #[test]
    fn test_prove_fcvt_wu_s_single() {
        let mut mem = SparseMemoryImage::new();
        write_fp_reg(&mut mem, 1, 3.7);
        prove_program(vec![fcvt_wu_s(0, 1), terminate()], mem);
    }
}
