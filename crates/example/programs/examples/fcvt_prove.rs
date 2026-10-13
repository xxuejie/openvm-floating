//! FCLASS.S / FCVT test: classification and float <-> int conversions.
//!
//! Reveals 8 u32 results:
//!   0: fclass(1.5)      = 1 << 6 (+normal)
//!   1: fclass(-inf)     = 1 << 0 (-inf)
//!   2: fclass(qNaN)     = 1 << 9
//!   3: fcvt.w.s(3.7)    = 4
//!   4: fcvt.w.s(-3.7)   = -4 = 0xFFFFFFFC
//!   5: fcvt.wu.s(NaN)   = 0xFFFFFFFF
//!   6: fcvt.s.w(-42)    = -42.0 (bits)
//!   7: fcvt.s.wu(42)    = 42.0 (bits)

#![no_main]
#![no_std]

openvm::entry!(main);

/// fclass.s: funct7 = 0b1110000, funct3 = 001, rs2 = x0.
macro_rules! fclass {
    ($a:expr, $slot:expr) => {{
        let result: u32;
        unsafe {
            core::arch::asm!(
                ".insn r 0x53, {f3}, {f7}, {rd}, {rs1}, x0",
                f3 = const openvm_rv32f_guest::FCLASS_FUNCT3,
                f7 = const openvm_rv32f_guest::FCLASS_S_FUNCT7,
                rd = out(reg) result,
                rs1 = in(reg) $a,
            );
        }
        openvm::io::reveal_u32(result, $slot);
    }};
}

/// fcvt: funct3 = rm matching the circuit (RTZ for float→int, RNE for int→float).
/// rs2 = x0 for W / x1 for WU.
macro_rules! fcvt {
    ($funct7:expr, $rm:expr, $rs2:tt, $a:expr, $slot:expr) => {{
        let result: u32;
        unsafe {
            core::arch::asm!(
                concat!(".insn r 0x53, {f3}, {f7}, {rd}, {rs1}, ", $rs2),
                f3 = const $rm,
                f7 = const $funct7,
                rd = out(reg) result,
                rs1 = in(reg) $a,
            );
        }
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let pos1_5: u32 = 0x3FC00000; // 1.5
    let neg_inf: u32 = 0xFF800000; // -inf
    let qnan: u32 = 0x7FC00000; // canonical NaN
    let pos3_7: u32 = 0x40666666; // 3.7
    let neg3_7: u32 = 0xC0666666; // -3.7
    let neg42: u32 = (-42i32) as u32;
    let pos42: u32 = 42;

    fclass!(pos1_5, 0);
    fclass!(neg_inf, 1);
    fclass!(qnan, 2);
    // float → int: circuit is RTZ-only
    fcvt!(
        openvm_rv32f_guest::FCVT_W_S_FUNCT7,
        openvm_rv32f_guest::RM_RTZ,
        "x0",
        pos3_7,
        3
    );
    fcvt!(
        openvm_rv32f_guest::FCVT_W_S_FUNCT7,
        openvm_rv32f_guest::RM_RTZ,
        "x0",
        neg3_7,
        4
    );
    fcvt!(
        openvm_rv32f_guest::FCVT_W_S_FUNCT7,
        openvm_rv32f_guest::RM_RTZ,
        "x1",
        qnan,
        5
    );
    // int → float: circuit is RNE
    fcvt!(
        openvm_rv32f_guest::FCVT_S_W_FUNCT7,
        openvm_rv32f_guest::RM_RNE,
        "x0",
        neg42,
        6
    );
    fcvt!(
        openvm_rv32f_guest::FCVT_S_W_FUNCT7,
        openvm_rv32f_guest::RM_RNE,
        "x1",
        pos42,
        7
    );
}
