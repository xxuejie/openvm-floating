//! Coverage-guided fuzzing of FP instruction execution through the full VM.
//!
//! Decodes bytes into a sequence of FP instructions (all families) plus
//! initial register contents, executes them through the VM interpreter, and
//! compares the final registers against the native semantics helpers.
//! Register aliasing (rd == rs*) is modeled faithfully.

#![no_main]

use std::{collections::BTreeMap, sync::OnceLock};

use libfuzzer_sys::fuzz_target;
use openvm_circuit::arch::VmExecutor;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::RV32_REGISTER_AS,
    LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_rv32f_circuit::{
    fadd::fadd_s,
    fclass::fclass_s,
    fcmp::{fcmp_result, FcmpOp},
    fcvt_f2i::fcvt_f2i_result,
    fcvt_i2f::fcvt_i2f_result,
    fdiv::fdiv_s,
    fma::{fmadd_result, FmaddOp},
    fminmax::fminmax_result,
    fmul::fmul_s,
    fsqrt::fsqrt_s,
    fsub::fsub_s,
    Rv32FConfig,
};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::PrimeCharacteristicRing, p3_baby_bear::BabyBear,
};

type F = BabyBear;

const FP_AS: u32 = RV32_REGISTER_AS;
const NUM_VARIANTS: u8 = 22;
const MAX_OPS: usize = 24;

static EXECUTOR: OnceLock<VmExecutor<F, Rv32FConfig>> = OnceLock::new();

fn terminate() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        ..Default::default()
    }
}

fn make_instruction(sel: u8, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> Instruction<F> {
    let opcode = match sel {
        0 => FpOpcode::FADD_S,
        1 => FpOpcode::FSUB_S,
        2 => FpOpcode::FMUL_S,
        3 => FpOpcode::FDIV_S,
        4 => FpOpcode::FMIN_S,
        5 => FpOpcode::FMAX_S,
        6 => FpOpcode::FSGNJ_S,
        7 => FpOpcode::FSGNJN_S,
        8 => FpOpcode::FSGNJX_S,
        9 => FpOpcode::FEQ_S,
        10 => FpOpcode::FLT_S,
        11 => FpOpcode::FLE_S,
        12 => FpOpcode::FCLASS_S,
        13 => FpOpcode::FCVT_W_S,
        14 => FpOpcode::FCVT_WU_S,
        15 => FpOpcode::FCVT_S_W,
        16 => FpOpcode::FCVT_S_WU,
        17 => FpOpcode::FSQRT_S,
        18 => FpOpcode::FMADD_S,
        19 => FpOpcode::FMSUB_S,
        20 => FpOpcode::FNMSUB_S,
        _ => FpOpcode::FNMADD_S,
    };
    if sel >= 18 {
        // ternary: a=rd, b=rs1, c=rs2, d=rs3, e=fp_as
        Instruction::new(
            VmOpcode::from_usize(opcode.global_opcode_usize()),
            F::from_u32(4 * rd),
            F::from_u32(4 * rs1),
            F::from_u32(4 * rs2),
            F::from_u32(4 * rs3),
            F::from_u32(FP_AS),
            F::ZERO,
            F::ZERO,
        )
    } else if (12..=17).contains(&sel) {
        // unary: a=rd, b=rs1, c=0, d=e=fp_as
        Instruction::new(
            VmOpcode::from_usize(opcode.global_opcode_usize()),
            F::from_u32(4 * rd),
            F::from_u32(4 * rs1),
            F::ZERO,
            F::from_u32(FP_AS),
            F::from_u32(FP_AS),
            F::ZERO,
            F::ZERO,
        )
    } else {
        Instruction::new(
            VmOpcode::from_usize(opcode.global_opcode_usize()),
            F::from_u32(4 * rd),
            F::from_u32(4 * rs1),
            F::from_u32(4 * rs2),
            F::from_u32(FP_AS),
            F::from_u32(FP_AS),
            F::ZERO,
            F::ZERO,
        )
    }
}

fn expected(sel: u8, a: u32, b: u32, c: u32) -> u32 {
    match sel {
        0 => fadd_s(a, b),
        1 => fsub_s(a, b),
        2 => fmul_s(a, b),
        3 => fdiv_s(a, b),
        4 => fminmax_result(a, b, true),
        5 => fminmax_result(a, b, false),
        6 => (a & 0x7FFF_FFFF) | (b & 0x8000_0000),
        7 => (a & 0x7FFF_FFFF) | (!b & 0x8000_0000),
        8 => a ^ (b & 0x8000_0000),
        9 => fcmp_result(a, b, FcmpOp::Eq),
        10 => fcmp_result(a, b, FcmpOp::Lt),
        11 => fcmp_result(a, b, FcmpOp::Le),
        12 => fclass_s(a),
        13 => fcvt_f2i_result(a, false),
        14 => fcvt_f2i_result(a, true),
        15 => fcvt_i2f_result(a, false),
        16 => fcvt_i2f_result(a, true),
        17 => fsqrt_s(a),
        18 => fmadd_result(a, b, c, FmaddOp::Madd),
        19 => fmadd_result(a, b, c, FmaddOp::Msub),
        20 => fmadd_result(a, b, c, FmaddOp::Nmsub),
        _ => fmadd_result(a, b, c, FmaddOp::Nmadd),
    }
}

struct DecodedOp {
    sel: u8,
    rd: u32,
    rs1: u32,
    rs2: u32,
    rs3: u32,
    a: u32,
    b: u32,
    c: u32,
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n_ops = (u16::from_le_bytes([data[0], data[1]]) as usize % MAX_OPS) + 1;
    let mut offset = 2;
    let mut ops: Vec<DecodedOp> = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        if offset + 17 > data.len() {
            break;
        }
        ops.push(DecodedOp {
            sel: data[offset] % NUM_VARIANTS,
            rd: (data[offset + 1] % 32) as u32,
            rs1: (data[offset + 2] % 32) as u32,
            rs2: (data[offset + 3] % 32) as u32,
            rs3: (data[offset + 4] % 32) as u32,
            a: u32::from_le_bytes(data[offset + 5..offset + 9].try_into().unwrap()),
            b: u32::from_le_bytes(data[offset + 9..offset + 13].try_into().unwrap()),
            c: u32::from_le_bytes(data[offset + 13..offset + 17].try_into().unwrap()),
        });
        offset += 17;
    }
    if ops.is_empty() {
        return;
    }

    // Initial register file: operand writes applied in order (last wins),
    // exactly matching what each op will read at runtime (unless an earlier
    // op's rd aliased a source register, which we simulate below).
    let mut regs: BTreeMap<u32, u32> = BTreeMap::new();
    for op in &ops {
        regs.insert(op.rs1, op.a);
        regs.insert(op.rs2, op.b);
        if op.sel >= 18 {
            regs.insert(op.rs3, op.c);
        }
    }
    let mut init = SparseMemoryImage::new();
    for (reg, val) in &regs {
        for (i, byte) in val.to_le_bytes().iter().enumerate() {
            init.insert((FP_AS, reg * 4 + i as u32), *byte);
        }
    }

    let mut instrs: Vec<Instruction<F>> = ops
        .iter()
        .map(|op| make_instruction(op.sel, op.rd, op.rs1, op.rs2, op.rs3))
        .collect();
    instrs.push(terminate());
    let program = Program::new_without_debug_infos(&instrs, 0);
    let exe = VmExe::new(program).with_init_memory(init);

    // Simulate with full aliasing semantics. All FP results are deterministic
    // (canonical NaN everywhere), so registers can be compared bit-exactly.
    let mut model = regs.clone();
    for op in &ops {
        let a = *model.get(&op.rs1).unwrap_or(&0);
        let b = *model.get(&op.rs2).unwrap_or(&0);
        let c = *model.get(&op.rs3).unwrap_or(&0);
        model.insert(op.rd, expected(op.sel, a, b, c));
    }

    let executor =
        EXECUTOR.get_or_init(|| VmExecutor::new(Rv32FConfig::default()).expect("executor"));
    let instance = executor.instance(&exe).expect("instance");
    let state = instance.execute(vec![], None).expect("execution failed");

    for (reg, want) in &model {
        let got: [u8; 4] = unsafe { state.memory.read::<u8, 4>(FP_AS, reg * 4) };
        assert_eq!(
            u32::from_le_bytes(got),
            *want,
            "reg {reg} mismatch ({} ops, input {data:02x?})",
            ops.len()
        );
    }
});
