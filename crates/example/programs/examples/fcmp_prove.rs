//! FEQ.S / FLT.S / FLE.S test: exercises normal, NaN, and signed-zero cases.
//!
//! Reveals 8 u32 results (each 0 or 1):
//!   0: feq(2.0, 2.0)    = 1
//!   1: feq(2.0, 3.0)    = 0
//!   2: flt(-1.0, 1.0)   = 1
//!   3: flt(1.0, -1.0)   = 0
//!   4: fle(-0.0, +0.0)  = 1  (signed zeros compare equal)
//!   5: feq(-0.0, +0.0)  = 1
//!   6: flt(-0.0, +0.0)  = 0
//!   7: fle(NaN, 1.0)    = 0  (any NaN -> false)

#![no_main]
#![no_std]

openvm::entry!(main);

macro_rules! fcmp {
    ($funct3:expr, $a:expr, $b:expr, $slot:expr) => {{
        let result: u32;
        openvm_custom_insn::custom_insn_r!(
            opcode = openvm_rv32f_guest::RV32_FP_OPCODE,
            funct3 = $funct3,
            funct7 = openvm_rv32f_guest::FCMP_S_FUNCT7,
            rd = Out result,
            rs1 = In $a,
            rs2 = In $b,
        );
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let one: u32 = 0x3F800000; // 1.0
    let two: u32 = 0x40000000; // 2.0
    let three: u32 = 0x40400000; // 3.0
    let neg1: u32 = 0xBF800000; // -1.0
    let nan: u32 = 0x7FC00000; // canonical NaN
    let pos_zero: u32 = 0x00000000; // +0.0
    let neg_zero: u32 = 0x80000000; // -0.0

    fcmp!(openvm_rv32f_guest::FEQ_FUNCT3, two, two, 0);
    fcmp!(openvm_rv32f_guest::FEQ_FUNCT3, two, three, 1);
    fcmp!(openvm_rv32f_guest::FLT_FUNCT3, neg1, one, 2);
    fcmp!(openvm_rv32f_guest::FLT_FUNCT3, one, neg1, 3);
    fcmp!(openvm_rv32f_guest::FLE_FUNCT3, neg_zero, pos_zero, 4);
    fcmp!(openvm_rv32f_guest::FEQ_FUNCT3, neg_zero, pos_zero, 5);
    fcmp!(openvm_rv32f_guest::FLT_FUNCT3, neg_zero, pos_zero, 6);
    fcmp!(openvm_rv32f_guest::FLE_FUNCT3, nan, one, 7);
}
