use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr};

/// Opcodes for the RISC-V F (single-precision floating-point) extension.
/// The offset 0x300 is chosen to avoid collision with rv32im opcodes
/// (which use 0x200-0x260 range).
#[allow(non_camel_case_types)]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x300]
#[repr(usize)]
pub enum FpOpcode {
    FADD_S,
    FSUB_S,
    FMUL_S,
    FDIV_S,
    FSGNJ_S,
    FSGNJN_S,
    FSGNJX_S,
    FMIN_S,
    FMAX_S,
    FLE_S,
    FLT_S,
    FEQ_S,
    FCLASS_S,
    FCVT_W_S,
    FCVT_WU_S,
    FCVT_S_W,
    FCVT_S_WU,
    FMADD_S,
    FMSUB_S,
    FNMSUB_S,
    FNMADD_S,
}
