use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_rv32f_guest::{
    FADD_S_FUNCT7, FDIV_S_FUNCT7, FMUL_S_FUNCT7, FSGNJN_FUNCT3, FSGNJX_FUNCT3, FSGNJ_FUNCT3,
    FSGNJ_S_FUNCT7, FSUB_S_FUNCT7, RM_DYN, RM_RNE, RV32_FP_OPCODE, RV32_FP_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};
use rrs_lib::instruction_formats::RType;

mod instructions;
pub use instructions::*;

/// Transpiler extension for RISC-V F / Zfinx single-precision floating-point instructions.
///
/// Recognizes OPC-FP (0x53) instructions and converts them to OpenVM opcodes.
/// Register operands (rd, rs1, rs2) are mapped to the configured address space:
/// - AS 1 (integer registers) for Zfinx mode — default
/// - AS 5 (FP registers) for F extension mode
///
/// Only RNE (rm=0) and DYN (rm=7, treated as RNE) rounding modes are accepted.
/// All other rounding modes are transpiled to UNIMP.
pub struct Rv32FTranspilerExtension {
    /// Address space for FP register operands.
    /// 1 = Zfinx (integer registers), 5 = F extension (FP registers).
    pub fp_register_as: u32,
}

impl Default for Rv32FTranspilerExtension {
    fn default() -> Self {
        Self {
            fp_register_as: openvm_instructions::riscv::RV32_REGISTER_AS, // Zfinx
        }
    }
}

impl Rv32FTranspilerExtension {
    /// Create a transpiler for Zfinx mode (floats in integer registers).
    pub fn new_zfinx() -> Self {
        Self {
            fp_register_as: openvm_instructions::riscv::RV32_REGISTER_AS,
        }
    }

    /// Create a transpiler for F extension mode (separate FP registers).
    pub fn new_f() -> Self {
        Self {
            fp_register_as: RV32_FP_REGISTER_AS,
        }
    }
}

impl<F: PrimeField32> TranspilerExtension<F> for Rv32FTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];

        let opcode = (instruction_u32 & 0x7f) as u8;
        if opcode != RV32_FP_OPCODE {
            return None;
        }

        let dec_insn = RType::new(instruction_u32);
        let funct7 = dec_insn.funct7 as u8;
        let rm = ((instruction_u32 >> 12) & 0b111) as u8;

        // Sign injection family: dispatch on funct3 (not rm), no rounding mode check
        if funct7 == FSGNJ_S_FUNCT7 {
            let opcode_out = match rm {
                FSGNJ_FUNCT3 => FpOpcode::FSGNJ_S,
                FSGNJN_FUNCT3 => FpOpcode::FSGNJN_S,
                FSGNJX_FUNCT3 => FpOpcode::FSGNJX_S,
                _ => return None,
            };
            return Some(TranspilerOutput::one_to_one(Instruction::new(
                openvm_instructions::VmOpcode::from_usize(opcode_out.global_opcode_usize()),
                F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
                F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
                F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
                F::from_u32(self.fp_register_as),
                F::from_u32(self.fp_register_as),
                F::ZERO,
                F::ZERO,
            )));
        }

        // Arithmetic instructions: only RNE and DYN (treated as RNE) are supported.
        if rm != RM_RNE && rm != RM_DYN {
            return Some(TranspilerOutput::one_to_one(
                openvm_transpiler::util::unimp(),
            ));
        }

        let opcode_out = match funct7 {
            FADD_S_FUNCT7 => FpOpcode::FADD_S,
            FSUB_S_FUNCT7 => FpOpcode::FSUB_S,
            FMUL_S_FUNCT7 => FpOpcode::FMUL_S,
            FDIV_S_FUNCT7 => FpOpcode::FDIV_S,
            _ => return None, // Not an F instruction we handle (e.g. FSQRT, FSGNJ, etc.)
        };

        let inst = Instruction::new(
            openvm_instructions::VmOpcode::from_usize(opcode_out.global_opcode_usize()),
            F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
            F::from_u32(self.fp_register_as), // d: rd address space
            F::from_u32(self.fp_register_as), // e: rs2 address space
            F::ZERO,
            F::ZERO,
        );

        Some(TranspilerOutput::one_to_one(inst))
    }
}
