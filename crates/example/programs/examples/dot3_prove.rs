//! Guest program with hardcoded values for STARK proof testing.
//!
//! Computes dot3([1,2,3], [4,5,6]) = 32 using FMUL.S + FADD.S.

#![no_main]
#![no_std]

openvm::entry!(main);

pub fn main() {
    // Hardcoded inputs: a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0]
    let ax: u32 = 0x3F800000; // 1.0
    let ay: u32 = 0x40000000; // 2.0
    let az: u32 = 0x40400000; // 3.0
    let bx: u32 = 0x40800000; // 4.0
    let by: u32 = 0x40A00000; // 5.0
    let bz: u32 = 0x40C00000; // 6.0

    // m1 = ax * bx
    let m1: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m1, rs1 = In ax, rs2 = In bx,
    );

    // m2 = ay * by
    let m2: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m2, rs1 = In ay, rs2 = In by,
    );

    // m3 = az * bz
    let m3: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m3, rs1 = In az, rs2 = In bz,
    );

    // s1 = m1 + m2
    let s1: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FADD_S_FUNCT7,
        rd = Out s1, rs1 = In m1, rs2 = In m2,
    );

    // result = s1 + m3 = dot3([1,2,3],[4,5,6]) = 32.0
    let result: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FADD_S_FUNCT7,
        rd = Out result, rs1 = In s1, rs2 = In m3,
    );

    openvm::io::reveal_u32(result, 0);
}
