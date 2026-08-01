use std::borrow::BorrowMut;

use openvm_circuit::arch::{BasicAdapterInterface, MinimalInstruction, VmCoreAir};
use openvm_instructions::LocalOpcode;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::*;
use crate::test_utils::NumericBuilder;

type F = BabyBear;
type TestI = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 4, 4>;

fn build_row(data: [u8; 4], opcode: FpOpcode, is_valid: bool) -> Vec<F> {
    let mut buf = vec![F::ZERO; FxferCoreCols::<F>::width()];
    let row: &mut FxferCoreCols<F> = buf.as_mut_slice().borrow_mut();
    for (i, &b) in data.iter().enumerate() {
        row.data[i] = F::from_u8(b);
        row.result[i] = F::from_u8(b);
    }
    match opcode {
        FpOpcode::FMV_X_S => {
            row.fmv_x_flag = F::ONE;
            row.fmv_s_flag = F::ZERO;
        }
        FpOpcode::FMV_S_X => {
            row.fmv_x_flag = F::ZERO;
            row.fmv_s_flag = F::ONE;
        }
        _ => unreachable!(),
    }
    row.is_valid = F::from_bool(is_valid);
    buf
}

fn eval_row(buf: &[F]) -> NumericBuilder<F> {
    let air = FxferCoreAir::new(FpOpcode::CLASS_OFFSET);
    let mut b = NumericBuilder::new(vec![]);
    VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
    b
}

#[test]
fn honest_fmv_x_s_passes() {
    let buf = build_row(3.5f32.to_bits().to_le_bytes(), FpOpcode::FMV_X_S, true);
    assert!(eval_row(&buf).ok(), "fmv.x.s honest row should pass");
}

#[test]
fn honest_fmv_s_x_passes() {
    let buf = build_row((-2.25f32).to_bits().to_le_bytes(), FpOpcode::FMV_S_X, true);
    assert!(eval_row(&buf).ok(), "fmv.s.x honest row should pass");
}

#[test]
fn zero_padding_row_passes() {
    let mut buf = vec![F::ZERO; FxferCoreCols::<F>::width()];
    let row: &mut FxferCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.is_valid = F::ZERO;
    assert!(eval_row(&buf).ok(), "padding row should pass");
}

#[test]
fn mismatched_result_is_rejected() {
    let mut buf = build_row(1.0f32.to_bits().to_le_bytes(), FpOpcode::FMV_X_S, true);
    let row: &mut FxferCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.result[0] = F::from_u8(0x7F);
    assert!(!eval_row(&buf).ok(), "result != data must be rejected");
}

#[test]
fn both_flags_set_is_rejected() {
    let mut buf = build_row(1.0f32.to_bits().to_le_bytes(), FpOpcode::FMV_X_S, true);
    let row: &mut FxferCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.fmv_s_flag = F::ONE;
    assert!(
        !eval_row(&buf).ok(),
        "both opcode flags set must be rejected"
    );
}

#[test]
fn no_flag_set_is_rejected() {
    let mut buf = build_row(1.0f32.to_bits().to_le_bytes(), FpOpcode::FMV_X_S, true);
    let row: &mut FxferCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.fmv_x_flag = F::ZERO;
    assert!(
        !eval_row(&buf).ok(),
        "zero opcode flags on a valid row must be rejected"
    );
}
