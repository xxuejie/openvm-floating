//! FMIN.S / FMAX.S test: exercises normal, NaN, and signed-zero cases.
//!
//! Reveals 8 u32 results:
//!   0: fmin(3.0, 4.0)   = 3.0
//!   1: fmax(3.0, 4.0)   = 4.0
//!   2: fmin(NaN, 2.0)   = 2.0        (one NaN -> other operand)
//!   3: fmax(-1.5, NaN)  = -1.5       (one NaN -> other operand)
//!   4: fmin(NaN, NaN)   = 0x7FC00000 (both NaN -> canonical NaN)
//!   5: fmax(-0.0, +0.0) = +0.0       (signed zero)
//!   6: fmin(-0.0, +0.0) = -0.0       (signed zero)
//!   7: fmax(-2.5, -7.25) = -2.5      (both negative)

#![no_main]
#![no_std]

openvm::entry!(main);

macro_rules! fminmax {
    ($funct3:expr, $a:expr, $b:expr, $slot:expr) => {{
        let result: u32;
        openvm_custom_insn::custom_insn_r!(
            opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
            funct3 = $funct3,
            funct7 = openvm_rv32f_guest::FMIN_S_FUNCT7,
            rd = Out result,
            rs1 = In $a,
            rs2 = In $b,
        );
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let three: u32 = 0x40400000; // 3.0
    let four: u32 = 0x40800000; // 4.0
    let two: u32 = 0x40000000; // 2.0
    let neg1_5: u32 = 0xBFC00000; // -1.5
    let neg2_5: u32 = 0xC0200000; // -2.5
    let neg7_25: u32 = 0xC0E80000; // -7.25
    let nan: u32 = 0x7FC00000; // canonical NaN
    let pos_zero: u32 = 0x00000000; // +0.0
    let neg_zero: u32 = 0x80000000; // -0.0

    fminmax!(openvm_rv32f_guest::FMIN_FUNCT3, three, four, 0);
    fminmax!(openvm_rv32f_guest::FMAX_FUNCT3, three, four, 1);
    fminmax!(openvm_rv32f_guest::FMIN_FUNCT3, nan, two, 2);
    fminmax!(openvm_rv32f_guest::FMAX_FUNCT3, neg1_5, nan, 3);
    fminmax!(openvm_rv32f_guest::FMIN_FUNCT3, nan, nan, 4);
    fminmax!(openvm_rv32f_guest::FMAX_FUNCT3, neg_zero, pos_zero, 5);
    fminmax!(openvm_rv32f_guest::FMIN_FUNCT3, neg_zero, pos_zero, 6);
    fminmax!(openvm_rv32f_guest::FMAX_FUNCT3, neg2_5, neg7_25, 7);
}
