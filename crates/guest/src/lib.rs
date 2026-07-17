#![no_std]

/// RISC-V OPC-FP major opcode (all F arithmetic instructions).
pub const RV32_FP_OPCODE: u8 = 0b1010011;

/// funct7 field values for F arithmetic instructions.
pub const FADD_S_FUNCT7: u8 = 0b0000000;
pub const FSUB_S_FUNCT7: u8 = 0b0000100;
pub const FMUL_S_FUNCT7: u8 = 0b0001000;
pub const FDIV_S_FUNCT7: u8 = 0b0001100;

/// funct7 for sign injection family (fsgnj.s / fsgnjn.s / fsgnjx.s).
pub const FSGNJ_S_FUNCT7: u8 = 0b0010000;
/// funct3 values within the sign injection family.
pub const FSGNJ_FUNCT3: u8 = 0b000;
pub const FSGNJN_FUNCT3: u8 = 0b001;
pub const FSGNJX_FUNCT3: u8 = 0b010;

/// funct7 for min/max family (fmin.s / fmax.s).
pub const FMIN_S_FUNCT7: u8 = 0b0010100;
/// funct3 values within the min/max family.
pub const FMIN_FUNCT3: u8 = 0b000;
pub const FMAX_FUNCT3: u8 = 0b001;

/// RISC-V rounding mode (rm field, funct3 bits).
pub const RM_RNE: u8 = 0b000; // Round to Nearest, ties to Even
pub const RM_RTZ: u8 = 0b001; // Round toward Zero
pub const RM_RDN: u8 = 0b010; // Round Down (toward -inf)
pub const RM_RUP: u8 = 0b011; // Round Up (toward +inf)
pub const RM_RMM: u8 = 0b100; // Round to Nearest, ties to Max Magnitude
pub const RM_DYN: u8 = 0b111; // Dynamic rounding mode (uses FCSR)

/// Address space for FP registers when using F extension mode.
/// Zfinx mode uses AS 1 (integer registers) instead.
pub const RV32_FP_REGISTER_AS: u32 = 5;

/// Canonical NaN bit pattern for f32 (RISC-V / JoltPhysics deterministic mode).
pub const CANONICAL_NAN: u32 = 0x7FC00000;
