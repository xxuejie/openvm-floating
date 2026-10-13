//! Coverage-guided fuzzing of the FP transpiler extension.
//!
//! Feeds arbitrary u32 words through `Rv32FTranspilerExtension::process_custom`
//! and cross-checks the decode against an independent re-derivation of the
//! encoding table (panics and decode mismatches are bugs).

#![no_main]

use libfuzzer_sys::fuzz_target;
use openvm_instructions::LocalOpcode;
use openvm_rv32f_guest::*;
use openvm_rv32f_transpiler::{FpOpcode, Rv32FTranspilerExtension};
use openvm_stark_sdk::{openvm_stark_backend::p3_field::PrimeField32, p3_baby_bear::BabyBear};
use openvm_transpiler::TranspilerExtension;

type F = BabyBear;

const RV32_FP_OPCODE: u8 = 0b1010011;

fn rm_ok(rm: u8) -> bool {
    rm == RM_RNE || rm == RM_DYN
}

/// Independent re-derivation of the transpiler's decode table.
/// Returns Some(Some(op)) if a valid FP instruction is expected,
/// Some(None) if UNIMP is expected (valid family, bad rounding mode),
/// None if the word should not be handled by this extension.
fn expected_decode(insn: u32) -> Option<Option<FpOpcode>> {
    let opcode = (insn & 0x7f) as u8;
    let rm = ((insn >> 12) & 0b111) as u8;

    // Fused multiply-add family (R4 format)
    let fused = match opcode {
        FMADD_S_OPCODE => Some(FpOpcode::FMADD_S),
        FMSUB_S_OPCODE => Some(FpOpcode::FMSUB_S),
        FNMSUB_S_OPCODE => Some(FpOpcode::FNMSUB_S),
        FNMADD_S_OPCODE => Some(FpOpcode::FNMADD_S),
        _ => None,
    };
    if let Some(op) = fused {
        let fmt = ((insn >> 25) & 0b11) as u8;
        if fmt != 0b00 {
            return None;
        }
        return Some(if rm_ok(rm) { Some(op) } else { None });
    }

    if opcode != RV32_FP_OPCODE {
        return None;
    }
    let funct7 = ((insn >> 25) & 0x7f) as u8;
    let rs2 = ((insn >> 20) & 0b11111) as u8;
    match funct7 {
        FADD_S_FUNCT7 => Some(if rm_ok(rm) {
            Some(FpOpcode::FADD_S)
        } else {
            None
        }),
        FSUB_S_FUNCT7 => Some(if rm_ok(rm) {
            Some(FpOpcode::FSUB_S)
        } else {
            None
        }),
        FMUL_S_FUNCT7 => Some(if rm_ok(rm) {
            Some(FpOpcode::FMUL_S)
        } else {
            None
        }),
        FDIV_S_FUNCT7 => Some(if rm_ok(rm) {
            Some(FpOpcode::FDIV_S)
        } else {
            None
        }),
        FSGNJ_S_FUNCT7 => match rm {
            FSGNJ_FUNCT3 => Some(Some(FpOpcode::FSGNJ_S)),
            FSGNJN_FUNCT3 => Some(Some(FpOpcode::FSGNJN_S)),
            FSGNJX_FUNCT3 => Some(Some(FpOpcode::FSGNJX_S)),
            _ => None,
        },
        FMIN_S_FUNCT7 => match rm {
            FMIN_FUNCT3 => Some(Some(FpOpcode::FMIN_S)),
            FMAX_FUNCT3 => Some(Some(FpOpcode::FMAX_S)),
            _ => None,
        },
        FCMP_S_FUNCT7 => match rm {
            FLE_FUNCT3 => Some(Some(FpOpcode::FLE_S)),
            FLT_FUNCT3 => Some(Some(FpOpcode::FLT_S)),
            FEQ_FUNCT3 => Some(Some(FpOpcode::FEQ_S)),
            _ => None,
        },
        FSQRT_S_FUNCT7 => {
            if rs2 != 0 {
                None
            } else {
                Some(if rm_ok(rm) {
                    Some(FpOpcode::FSQRT_S)
                } else {
                    None
                })
            }
        }
        FCLASS_S_FUNCT7 => {
            if rm == FCLASS_FUNCT3 && rs2 == 0 {
                Some(Some(FpOpcode::FCLASS_S))
            } else {
                None
            }
        }
        FCVT_W_S_FUNCT7 => {
            // AIR always truncates: RTZ only (DYN/RNE rejected).
            let ok = rm == RM_RTZ;
            match rs2 {
                FCVT_RS2_W => Some(if ok { Some(FpOpcode::FCVT_W_S) } else { None }),
                FCVT_RS2_WU => Some(if ok { Some(FpOpcode::FCVT_WU_S) } else { None }),
                _ => None,
            }
        }
        FCVT_S_W_FUNCT7 => {
            // AIR implements RNE; DYN treated as RNE.
            match rs2 {
                FCVT_RS2_W => Some(if rm_ok(rm) {
                    Some(FpOpcode::FCVT_S_W)
                } else {
                    None
                }),
                FCVT_RS2_WU => Some(if rm_ok(rm) {
                    Some(FpOpcode::FCVT_S_WU)
                } else {
                    None
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

fuzz_target!(|data: &[u8]| {
    let transpiler = Rv32FTranspilerExtension::default();
    for word in data.chunks(4) {
        if word.len() < 4 {
            break;
        }
        let insn = u32::from_le_bytes(word.try_into().unwrap());
        let expected = expected_decode(insn);
        let out: Option<openvm_transpiler::TranspilerOutput<F>> =
            transpiler.process_custom(&[insn]);
        match (expected, out) {
            (Some(Some(op)), Some(output)) => {
                assert_eq!(output.instructions.len(), 1, "insn {insn:#010x}");
                let inst = output.instructions[0].as_ref().expect("instruction");
                assert_eq!(
                    inst.opcode.as_usize(),
                    op.global_opcode_usize(),
                    "insn {insn:#010x}: opcode mismatch (expected {op:?})"
                );
                // rd / rs1 pointers
                let rd = (insn >> 7) & 0b11111;
                let rs1 = (insn >> 15) & 0b11111;
                assert_eq!(inst.a.as_canonical_u32(), 4 * rd, "insn {insn:#010x}: rd");
                assert_eq!(inst.b.as_canonical_u32(), 4 * rs1, "insn {insn:#010x}: rs1");
            }
            (Some(None), Some(output)) => {
                // UNIMP expected
                let unimp = openvm_transpiler::util::unimp::<F>();
                let inst = output.instructions[0].as_ref().expect("instruction");
                assert_eq!(
                    inst.opcode.as_usize(),
                    unimp.opcode.as_usize(),
                    "insn {insn:#010x}: expected UNIMP"
                );
            }
            (Some(Some(op)), None) => {
                panic!("insn {insn:#010x}: extension missed valid {op:?}");
            }
            _ => {}
        }
    }
});
