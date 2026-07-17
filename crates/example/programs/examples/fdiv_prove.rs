//! FDIV.S test: normal, infinite, zero, and NaN cases.
//!
//! Reveals 8 u32 results:
//!   0: 12.0 / 4.0   = 3.0
//!   1: 1.0 / 3.0    = 0.333...
//!   2: -7.0 / 2.0   = -3.5
//!   3: 1.0 / +0.0   = +inf
//!   4: -1.0 / +0.0  = -inf
//!   5: 0.0 / 0.0    = canonical NaN
//!   6: inf / inf    = canonical NaN
//!   7: 1.0 / inf    = +0.0

#![no_main]
#![no_std]

openvm::entry!(main);

macro_rules! fdiv {
    ($a:expr, $b:expr, $slot:expr) => {{
        let result: u32;
        openvm_custom_insn::custom_insn_r!(
            opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
            funct3 = openvm_rv32f_guest::RM_RNE,
            funct7 = openvm_rv32f_guest::FDIV_S_FUNCT7,
            rd = Out result,
            rs1 = In $a,
            rs2 = In $b,
        );
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let twelve: u32 = 0x41400000; // 12.0
    let four: u32 = 0x40800000; // 4.0
    let one: u32 = 0x3F800000; // 1.0
    let three: u32 = 0x40400000; // 3.0
    let two: u32 = 0x40000000; // 2.0
    let neg7: u32 = 0xC0E00000; // -7.0
    let pos_zero: u32 = 0x00000000; // +0.0
    let inf: u32 = 0x7F800000; // +inf
    let neg1: u32 = 0xBF800000; // -1.0

    fdiv!(twelve, four, 0);
    fdiv!(one, three, 1);
    fdiv!(neg7, two, 2);
    fdiv!(one, pos_zero, 3);
    fdiv!(neg1, pos_zero, 4);
    fdiv!(pos_zero, pos_zero, 5);
    fdiv!(inf, inf, 6);
    fdiv!(one, inf, 7);
}
