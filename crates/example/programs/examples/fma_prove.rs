//! Fused multiply-add family test: FMADD.S / FMSUB.S / FNMSUB.S / FNMADD.S.
//!
//! Reveals 8 u32 results:
//!   0: fmadd(3.0, 4.0, 1.0)   = 13.0
//!   1: fmsub(3.0, 4.0, 1.0)   = 11.0
//!   2: fnmsub(3.0, 4.0, 1.0)  = -11.0
//!   3: fnmadd(3.0, 4.0, 1.0)  = -13.0
//!   4: fmadd(1+2^-12, 1+2^-12, -1.0) = 2^-11 + 2^-24 (fused-only)
//!   5: fmadd(inf, 2.0, 1.0)   = inf
//!   6: fmadd(inf, 0.0, 1.0)   = canonical NaN (0 * inf is invalid)
//!   7: fmadd(1.0, 2.0, NaN)   = canonical NaN

#![no_main]
#![no_std]

openvm::entry!(main);

/// Fused family: R4 format `.insn r4 opcode, funct3, fmt, rd, rs1, rs2, rs3`
/// with funct3 = rm (RNE) and fmt = 0 (single precision).
macro_rules! fmadd {
    ($opcode:expr, $a:expr, $b:expr, $c:expr, $slot:expr) => {{
        let result: u32;
        unsafe {
            core::arch::asm!(
                ".insn r4 {op}, {f3}, 0, {rd}, {rs1}, {rs2}, {rs3}",
                op = const $opcode,
                f3 = const openvm_rv32f_guest::RM_RNE,
                rd = out(reg) result,
                rs1 = in(reg) $a,
                rs2 = in(reg) $b,
                rs3 = in(reg) $c,
            );
        }
        openvm::io::reveal_u32(result, $slot);
    }};
}

pub fn main() {
    let three: u32 = 0x40400000; // 3.0
    let four: u32 = 0x40800000; // 4.0
    let one: u32 = 0x3F800000; // 1.0
    let two: u32 = 0x40000000; // 2.0
    let neg1: u32 = 0xBF800000; // -1.0
    let inf: u32 = 0x7F800000; // +inf
    let pos_zero: u32 = 0x00000000; // +0.0
    let nan: u32 = 0x7FC00000; // canonical NaN
    let one_plus_eps: u32 = 0x3F801000; // 1 + 2^-12

    fmadd!(openvm_rv32f_guest::FMADD_S_OPCODE, three, four, one, 0);
    fmadd!(openvm_rv32f_guest::FMSUB_S_OPCODE, three, four, one, 1);
    fmadd!(openvm_rv32f_guest::FNMSUB_S_OPCODE, three, four, one, 2);
    fmadd!(openvm_rv32f_guest::FNMADD_S_OPCODE, three, four, one, 3);
    fmadd!(openvm_rv32f_guest::FMADD_S_OPCODE, one_plus_eps, one_plus_eps, neg1, 4);
    fmadd!(openvm_rv32f_guest::FMADD_S_OPCODE, inf, two, one, 5);
    fmadd!(openvm_rv32f_guest::FMADD_S_OPCODE, inf, pos_zero, one, 6);
    fmadd!(openvm_rv32f_guest::FMADD_S_OPCODE, one, two, nan, 7);
}
