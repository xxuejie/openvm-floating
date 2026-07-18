//! FSQRT.S test: normal, zero, infinite, subnormal-input, and NaN cases.
//!
//! Reveals 8 u32 results:
//!   0: sqrt(4.0)      = 2.0
//!   1: sqrt(2.0)      = 1.4142135...
//!   2: sqrt(0.25)     = 0.5
//!   3: sqrt(min subnormal) = 2^-74.5 (normal)
//!   4: sqrt(+0.0)     = +0.0
//!   5: sqrt(-0.0)     = -0.0
//!   6: sqrt(+inf)     = +inf
//!   7: sqrt(-1.0)     = canonical NaN

#![no_main]
#![no_std]

openvm::entry!(main);

/// fsqrt.s: funct7 = 0b0101100, funct3 = rm (RNE), rs2 = x0.
macro_rules! fsqrt {
    ($a:expr, $slot:expr) => {{
        let result: u32;
        unsafe {
            core::arch::asm!(
                ".insn r 0x53, {f3}, {f7}, {rd}, {rs1}, x0",
                f3 = const openvm_rv32f_guest::RM_RNE,
                f7 = const openvm_rv32f_guest::FSQRT_S_FUNCT7,
                rd = out(reg) result,
                rs1 = in(reg) $a,
            );
        }
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let four: u32 = 0x40800000; // 4.0
    let two: u32 = 0x40000000; // 2.0
    let quarter: u32 = 0x3E800000; // 0.25
    let min_subnormal: u32 = 0x00000001;
    let pos_zero: u32 = 0x00000000; // +0.0
    let neg_zero: u32 = 0x80000000; // -0.0
    let inf: u32 = 0x7F800000; // +inf
    let neg1: u32 = 0xBF800000; // -1.0

    fsqrt!(four, 0);
    fsqrt!(two, 1);
    fsqrt!(quarter, 2);
    fsqrt!(min_subnormal, 3);
    fsqrt!(pos_zero, 4);
    fsqrt!(neg_zero, 5);
    fsqrt!(inf, 6);
    fsqrt!(neg1, 7);
}
