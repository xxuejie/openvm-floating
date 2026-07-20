#![no_std]

/// RISC-V OPC-FP major opcode (all F arithmetic instructions).
pub const RV32_FP_OPCODE: u8 = 0b1010011;

/// RISC-V LOAD-FP opcode (flw).
pub const RV32_LOAD_FP_OPCODE: u8 = 0b0000111;
/// RISC-V STORE-FP opcode (fsw).
pub const RV32_STORE_FP_OPCODE: u8 = 0b0100111;
/// funct3 for single-precision load/store (flw/fsw).
pub const FLW_FSW_FUNCT3: u8 = 0b010;

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

/// funct7 for compare family (feq.s / flt.s / fle.s).
/// These write rd to the integer register file, even in F extension mode.
pub const FCMP_S_FUNCT7: u8 = 0b1010000;
/// funct3 values within the compare family.
pub const FLE_FUNCT3: u8 = 0b000;
pub const FLT_FUNCT3: u8 = 0b001;
pub const FEQ_FUNCT3: u8 = 0b010;

/// funct7 for fclass.s (funct3=001) and fmv.x.s (funct3=000).
/// Single FP source operand; rd is an integer register.
pub const FCLASS_S_FUNCT7: u8 = 0b1110000;
/// funct3 for fclass.s.
pub const FCLASS_FUNCT3: u8 = 0b001;
/// funct3 for fmv.x.s (move FP bits to integer register).
pub const FMV_X_S_FUNCT3: u8 = 0b000;

/// funct7 for fmv.s.x (move integer bits to FP register).
/// Single integer source operand; rd is an FP register.
pub const FMV_S_X_FUNCT7: u8 = 0b1111000;

/// funct7 for fsqrt.s. Single FP source operand; rs2 field must be 0.
pub const FSQRT_S_FUNCT7: u8 = 0b0101100;

/// funct7 for fcvt.w.s / fcvt.wu.s (float -> int); the rs2 field selects
/// the integer type (0 = int32, 1 = uint32). rd is an integer register.
pub const FCVT_W_S_FUNCT7: u8 = 0b1100000;
/// funct7 for fcvt.s.w / fcvt.s.wu (int -> float); the rs2 field selects
/// the integer type (0 = int32, 1 = uint32). rs1 is an integer register.
pub const FCVT_S_W_FUNCT7: u8 = 0b1101000;
/// rs2-field values selecting the conversion integer type.
pub const FCVT_RS2_W: u8 = 0b00000;
pub const FCVT_RS2_WU: u8 = 0b00001;

/// Major opcodes for the fused multiply-add family (R4 format).
/// All take a third source register rs3 in bits [31:27] and a 2-bit fmt
/// field in bits [26:25] (0b00 = single precision).
pub const FMADD_S_OPCODE: u8 = 0b1000011;
pub const FMSUB_S_OPCODE: u8 = 0b1000111;
pub const FNMSUB_S_OPCODE: u8 = 0b1001011;
pub const FNMADD_S_OPCODE: u8 = 0b1001111;

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
