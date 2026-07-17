//! Guest program that computes dot3 using FMUL.S and FADD.S via `custom_insn_r!`.
//!
//! Matches JoltPhysics's `Vec3::Dot` evaluation order:
//!   ((ax*bx) + (ay*by)) + (az*bz)
//!
//! In Zfinx mode these instructions operate on integer registers.

#![no_main]
#![no_std]

openvm::entry!(main);

pub fn main() {
    // Read two vec3 inputs (6 × u32 bit patterns)
    let ax: u32 = openvm::io::read();
    let ay: u32 = openvm::io::read();
    let az: u32 = openvm::io::read();
    let bx: u32 = openvm::io::read();
    let by: u32 = openvm::io::read();
    let bz: u32 = openvm::io::read();

    // m1 = ax * bx
    let m1: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m1,
        rs1 = In ax,
        rs2 = In bx,
    );

    // m2 = ay * by
    let m2: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m2,
        rs1 = In ay,
        rs2 = In by,
    );

    // m3 = az * bz
    let m3: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FMUL_S_FUNCT7,
        rd = Out m3,
        rs1 = In az,
        rs2 = In bz,
    );

    // s1 = m1 + m2
    let s1: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FADD_S_FUNCT7,
        rd = Out s1,
        rs1 = In m1,
        rs2 = In m2,
    );

    // result = s1 + m3  →  ((ax*bx) + (ay*by)) + (az*bz)
    let result: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FADD_S_FUNCT7,
        rd = Out result,
        rs1 = In s1,
        rs2 = In m3,
    );

    openvm::io::reveal_u32(result, 0);
}
