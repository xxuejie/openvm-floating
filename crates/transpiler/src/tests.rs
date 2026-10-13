//! Unit tests for the RISC-V F extension transpiler.
//!
//! These verify the OpenVM instruction operand layout (a..g) produced for each
//! RISC-V F/Zfinx instruction, in both Zfinx mode (fp_register_as = 1) and
//! F-extension mode (fp_register_as = 5). The cross-AS instructions (fmv.x.s,
//! fmv.s.x, flw, fsw, fclass, fcvt) encode their read/write address spaces in
//! the `d`/`e`/`f` fields, which the circuit chips rely on.

use openvm_rv32f_guest::{RM_RDN, RM_RMM, RM_RUP};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::TranspilerOutput;

use super::*;

type F = BabyBear;

fn enc(opcode: u8, rd: u32, funct3: u8, rs1: u32, rs2: u32, funct7: u8) -> u32 {
    let opcode = opcode as u32;
    let funct3 = funct3 as u32;
    let funct7 = funct7 as u32;
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn enc_i(opcode: u8, rd: u32, funct3: u8, rs1: u32, imm: i32) -> u32 {
    let opcode = opcode as u32;
    let funct3 = funct3 as u32;
    let imm = imm as u32;
    (imm & 0xFFF) << 20 | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn enc_s(opcode: u8, rs1: u32, funct3: u8, rs2: u32, imm: i32) -> u32 {
    let opcode = opcode as u32;
    let funct3 = funct3 as u32;
    let imm = imm as u32;
    let imm11_5 = (imm >> 5) & 0x7F;
    let imm4_0 = imm & 0x1F;
    (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm4_0 << 7) | opcode
}

/// Transpiles a single instruction and returns the emitted OpenVM instruction.
fn trans(ext: &Rv32FTranspilerExtension, insn: u32) -> Instruction<F> {
    let out = ext
        .process_custom(&[insn])
        .expect("expected a transpile hit");
    assert_eq!(out.used_u32s, 1);
    out.instructions[0]
        .clone()
        .expect("expected one instruction")
}

fn fields(insn: &Instruction<F>) -> (u32, u32, u32, u32, u32, u32, u32) {
    (
        insn.a.as_canonical_u32(),
        insn.b.as_canonical_u32(),
        insn.c.as_canonical_u32(),
        insn.d.as_canonical_u32(),
        insn.e.as_canonical_u32(),
        insn.f.as_canonical_u32(),
        insn.g.as_canonical_u32(),
    )
}

/// FMV.X.S encodes as (a=FP, b=int?, d=1, e=fp_as) in Zfinx and (d=1, e=5) in F mode.
#[test]
fn fmv_x_s_fields() {
    let insn = enc(RV32_FP_OPCODE, 3, FMV_X_S_FUNCT3, 5, 0, FCLASS_S_FUNCT7);
    let zfinx = fields(&trans(&Rv32FTranspilerExtension::new_zfinx(), insn));
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), insn));
    // a = int rd, b = fp rs1, c = 0, d = int AS, e = fp AS
    assert_eq!(zfinx, (4 * 3, 4 * 5, 0, 1, 1, 0, 0));
    assert_eq!(f, (4 * 3, 4 * 5, 0, 1, 5, 0, 0));
    assert_eq!(
        trans(&Rv32FTranspilerExtension::new_f(), insn).opcode,
        FpOpcode::FMV_X_S.global_opcode()
    );
}

/// FMV.S.X encodes as (a=FP rd, b=int rs1, d=fp_as, e=1).
#[test]
fn fmv_s_x_fields() {
    let insn = enc(RV32_FP_OPCODE, 4, 0, 6, 0, FMV_S_X_FUNCT7);
    let zfinx = fields(&trans(&Rv32FTranspilerExtension::new_zfinx(), insn));
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), insn));
    assert_eq!(zfinx, (4 * 4, 4 * 6, 0, 1, 1, 0, 0));
    assert_eq!(f, (4 * 4, 4 * 6, 0, 5, 1, 0, 0));
}

/// FLW: a = FP rd, b = base (int), c = imm, d = fp_as, e = 2 (memory).
#[test]
fn flw_fields() {
    let insn = enc_i(RV32_LOAD_FP_OPCODE, 2, FLW_FSW_FUNCT3, 1, 16);
    let zfinx = fields(&trans(&Rv32FTranspilerExtension::new_zfinx(), insn));
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), insn));
    assert_eq!(zfinx, (4 * 2, 4, 16, 1, 2, 0, 0));
    assert_eq!(f, (4 * 2, 4, 16, 5, 2, 0, 0));
}

/// FSW: a = base (int), b = FP src, c = imm, d = 2 (memory), e = fp_as, f = 1.
#[test]
fn fsw_fields() {
    let insn = enc_s(RV32_STORE_FP_OPCODE, 1, FLW_FSW_FUNCT3, 7, -4);
    let zfinx = fields(&trans(&Rv32FTranspilerExtension::new_zfinx(), insn));
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), insn));
    assert_eq!(zfinx, (4, 4 * 7, 0xFFFC, 2, 1, 1, 0));
    assert_eq!(f, (4, 4 * 7, 0xFFFC, 2, 5, 1, 0));
}

/// FCLASS.S: a = int rd, b = fp rs1, d = 1 (int), e = fp_as.
#[test]
fn fclass_fields() {
    let insn = enc(RV32_FP_OPCODE, 8, FCLASS_FUNCT3, 3, 0, FCLASS_S_FUNCT7);
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), insn));
    assert_eq!(f, (4 * 8, 4 * 3, 0, 1, 5, 0, 0));
}

/// fcvt.w.s: read fp, write int; fcvt.s.w: read int, write fp.
#[test]
fn fcvt_fields() {
    // fcvt.w.s requires RTZ; fcvt.s.w requires RNE (or DYN).
    let fcvt_w = enc(
        RV32_FP_OPCODE,
        9,
        RM_RTZ,
        3,
        FCVT_RS2_W as u32,
        FCVT_W_S_FUNCT7,
    );
    let fcvt_s = enc(
        RV32_FP_OPCODE,
        9,
        RM_RNE,
        3,
        FCVT_RS2_W as u32,
        FCVT_S_W_FUNCT7,
    );
    let fw = fields(&trans(&Rv32FTranspilerExtension::new_f(), fcvt_w));
    let fs = fields(&trans(&Rv32FTranspilerExtension::new_f(), fcvt_s));
    assert_eq!(fw, (4 * 9, 4 * 3, 0, 1, 5, 0, 0)); // fp -> int
    assert_eq!(fs, (4 * 9, 4 * 3, 0, 5, 1, 0, 0)); // int -> fp
}

/// Binary FP ops: all operands in fp_as, d = e = fp_as.
#[test]
fn binary_op_fields() {
    let fadd = enc(RV32_FP_OPCODE, 2, 0, 3, 4, FADD_S_FUNCT7);
    let fmul = enc(RV32_FP_OPCODE, 2, 0, 3, 4, FMUL_S_FUNCT7);
    let f = fields(&trans(&Rv32FTranspilerExtension::new_f(), fadd));
    let z = fields(&trans(&Rv32FTranspilerExtension::new_zfinx(), fmul));
    assert_eq!(f, (4 * 2, 4 * 3, 4 * 4, 5, 5, 0, 0));
    assert_eq!(z, (4 * 2, 4 * 3, 4 * 4, 1, 1, 0, 0));
    assert_eq!(
        trans(&Rv32FTranspilerExtension::new_f(), fadd).opcode,
        FpOpcode::FADD_S.global_opcode()
    );
    assert_eq!(
        trans(&Rv32FTranspilerExtension::new_zfinx(), fmul).opcode,
        FpOpcode::FMUL_S.global_opcode()
    );
}

/// fsgnj family dispatches on funct3 and keeps d = e = fp_as.
#[test]
fn fsgnj_fields() {
    for (funct3, op) in [
        (FSGNJ_FUNCT3, FpOpcode::FSGNJ_S),
        (FSGNJN_FUNCT3, FpOpcode::FSGNJN_S),
        (FSGNJX_FUNCT3, FpOpcode::FSGNJX_S),
    ] {
        let insn = enc(RV32_FP_OPCODE, 2, funct3, 3, 4, FSGNJ_S_FUNCT7);
        let out = trans(&Rv32FTranspilerExtension::new_f(), insn);
        assert_eq!(out.opcode, op.global_opcode());
        assert_eq!(fields(&out), (4 * 2, 4 * 3, 4 * 4, 5, 5, 0, 0));
    }
}

/// fmin/fmax dispatch on funct3.
#[test]
fn fminmax_fields() {
    let fmin = enc(RV32_FP_OPCODE, 2, FMIN_FUNCT3, 3, 4, FMIN_S_FUNCT7);
    let fmax = enc(RV32_FP_OPCODE, 2, FMAX_FUNCT3, 3, 4, FMIN_S_FUNCT7);
    assert_eq!(
        trans(&Rv32FTranspilerExtension::new_f(), fmin).opcode,
        FpOpcode::FMIN_S.global_opcode()
    );
    assert_eq!(
        trans(&Rv32FTranspilerExtension::new_f(), fmax).opcode,
        FpOpcode::FMAX_S.global_opcode()
    );
}

/// Non-F instructions fall through; unsupported RMs become UNIMP.
#[test]
fn falls_through() {
    // A plain integer RISC-V instruction (addi).
    let addi = enc_i(0x13, 1, 0, 2, 8);
    let ext: &dyn TranspilerExtension<F> = &Rv32FTranspilerExtension::new_f();
    assert!(ext.process_custom(&[addi]).is_none());
    // fadd.s with a non-RNE rounding mode -> UNIMP.
    let fadd_rm = enc(RV32_FP_OPCODE, 2, RM_RUP, 3, 4, FADD_S_FUNCT7);
    let out = ext.process_custom(&[fadd_rm]).expect("hit");
    let inst = out.instructions[0].as_ref().expect("insn");
    assert_eq!(inst.opcode, openvm_transpiler::util::unimp::<F>().opcode);
}

fn is_unimp(insn: u32) -> bool {
    let ext = Rv32FTranspilerExtension::new_f();
    let out: TranspilerOutput<F> = ext
        .process_custom(&[insn])
        .expect("expected a transpile hit");
    let inst = out.instructions[0].as_ref().expect("insn");
    inst.opcode == openvm_transpiler::util::unimp::<F>().opcode
}

fn is_opcode(insn: u32, op: FpOpcode) -> bool {
    let out = trans(&Rv32FTranspilerExtension::new_f(), insn);
    out.opcode == op.global_opcode()
}

/// Arithmetic / FMA / fsqrt: RNE and DYN accepted; other RMs → UNIMP.
#[test]
fn rm_policy_rne_ops() {
    // fadd.s
    assert!(is_opcode(
        enc(RV32_FP_OPCODE, 2, RM_RNE, 3, 4, FADD_S_FUNCT7),
        FpOpcode::FADD_S
    ));
    assert!(is_opcode(
        enc(RV32_FP_OPCODE, 2, RM_DYN, 3, 4, FADD_S_FUNCT7),
        FpOpcode::FADD_S
    ));
    for bad in [RM_RTZ, RM_RDN, RM_RUP, RM_RMM] {
        assert!(
            is_unimp(enc(RV32_FP_OPCODE, 2, bad, 3, 4, FADD_S_FUNCT7)),
            "fadd rm={bad} should UNIMP"
        );
    }

    // fsqrt.s
    assert!(is_opcode(
        enc(RV32_FP_OPCODE, 2, RM_RNE, 3, 0, FSQRT_S_FUNCT7),
        FpOpcode::FSQRT_S
    ));
    assert!(is_opcode(
        enc(RV32_FP_OPCODE, 2, RM_DYN, 3, 0, FSQRT_S_FUNCT7),
        FpOpcode::FSQRT_S
    ));
    assert!(is_unimp(enc(
        RV32_FP_OPCODE,
        2,
        RM_RTZ,
        3,
        0,
        FSQRT_S_FUNCT7
    )));

    // fmadd.s (R4): rs3=0, fmt=single, rs2=4, rs1=3, rd=2
    let fmadd = |rm: u8| {
        (4u32 << 20) | (3u32 << 15) | ((rm as u32) << 12) | (2u32 << 7) | (FMADD_S_OPCODE as u32)
    };
    assert!(is_opcode(fmadd(RM_RNE), FpOpcode::FMADD_S));
    assert!(is_unimp(fmadd(RM_RTZ)));
}

/// fcvt.w[u].s: RTZ only (AIR truncates). RNE/DYN/other → UNIMP.
#[test]
fn rm_policy_fcvt_f2i() {
    assert!(is_opcode(
        enc(
            RV32_FP_OPCODE,
            9,
            RM_RTZ,
            3,
            FCVT_RS2_W as u32,
            FCVT_W_S_FUNCT7
        ),
        FpOpcode::FCVT_W_S
    ));
    assert!(is_opcode(
        enc(
            RV32_FP_OPCODE,
            9,
            RM_RTZ,
            3,
            FCVT_RS2_WU as u32,
            FCVT_W_S_FUNCT7
        ),
        FpOpcode::FCVT_WU_S
    ));
    for bad in [RM_RNE, RM_DYN, RM_RDN, RM_RUP, RM_RMM] {
        assert!(
            is_unimp(enc(
                RV32_FP_OPCODE,
                9,
                bad,
                3,
                FCVT_RS2_W as u32,
                FCVT_W_S_FUNCT7
            )),
            "fcvt.w.s rm={bad} should UNIMP"
        );
        assert!(
            is_unimp(enc(
                RV32_FP_OPCODE,
                9,
                bad,
                3,
                FCVT_RS2_WU as u32,
                FCVT_W_S_FUNCT7
            )),
            "fcvt.wu.s rm={bad} should UNIMP"
        );
    }
}

/// fcvt.s.w[u]: RNE or DYN only (AIR is RNE). Other RMs → UNIMP.
#[test]
fn rm_policy_fcvt_i2f() {
    assert!(is_opcode(
        enc(
            RV32_FP_OPCODE,
            9,
            RM_RNE,
            3,
            FCVT_RS2_W as u32,
            FCVT_S_W_FUNCT7
        ),
        FpOpcode::FCVT_S_W
    ));
    assert!(is_opcode(
        enc(
            RV32_FP_OPCODE,
            9,
            RM_DYN,
            3,
            FCVT_RS2_WU as u32,
            FCVT_S_W_FUNCT7
        ),
        FpOpcode::FCVT_S_WU
    ));
    for bad in [RM_RTZ, RM_RDN, RM_RUP, RM_RMM] {
        assert!(
            is_unimp(enc(
                RV32_FP_OPCODE,
                9,
                bad,
                3,
                FCVT_RS2_W as u32,
                FCVT_S_W_FUNCT7
            )),
            "fcvt.s.w rm={bad} should UNIMP"
        );
    }
}
