//! Minimal FADD.S test: 3.0 + 4.0 = 7.0

#![no_main]
#![no_std]

openvm::entry!(main);

pub fn main() {
    let a: u32 = 0x40400000; // 3.0
    let b: u32 = 0x40800000; // 4.0
    let result: u32;
    openvm_custom_insn::custom_insn_r!(
        opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
        funct3 = openvm_rv32f_guest::RM_RNE,
        funct7 = openvm_rv32f_guest::FADD_S_FUNCT7,
        rd = Out result, rs1 = In a, rs2 = In b,
    );
    openvm::io::reveal_u32(result, 0);
}
