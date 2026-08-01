use std::borrow::BorrowMut;

use openvm_circuit::arch::{VmAdapterInterface, VmCoreAir};
use openvm_instructions::LocalOpcode;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::*;
use crate::adapters::FpLoadStoreInstruction;
use crate::test_utils::NumericBuilder;

#[cfg(test)]
mod harness_tests {
    use openvm_circuit::arch::testing::{TestBuilder, TestChipHarness, VmChipTestBuilder};
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_rv32f_transpiler::FpOpcode;
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use crate::{
        adapters::{Rv32FpMemAdapterAir, Rv32FpMemAdapterExecutor, Rv32FpMemAdapterFiller},
        fmem::{FmemCoreAir, FmemExecutor, FmemFiller, Rv32FmemAir, Rv32FmemChip},
    };

    type F = BabyBear;
    type Harness =
        TestChipHarness<F, FmemExecutor<Rv32FpMemAdapterExecutor>, Rv32FmemAir, Rv32FmemChip<F>>;

    const FP_AS: u32 = 1;
    const MEM_AS: u32 = 2;
    const POINTER_MAX_BITS: usize = 32;

    fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
        let range_checker = tester.range_checker();
        let air = Rv32FmemAir::new(
            Rv32FpMemAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
                FP_AS,
                POINTER_MAX_BITS,
            ),
            FmemCoreAir::new(FpOpcode::CLASS_OFFSET),
        );
        let executor = FmemExecutor::new(
            Rv32FpMemAdapterExecutor::new(FP_AS, POINTER_MAX_BITS),
            FpOpcode::CLASS_OFFSET,
        );
        let chip = Rv32FmemChip::<F>::new(
            FmemFiller::new(
                Rv32FpMemAdapterFiller::new(range_checker, FP_AS, POINTER_MAX_BITS),
                FpOpcode::CLASS_OFFSET,
            ),
            tester.memory_helper(),
        );
        Harness::with_capacity(executor, air, chip, 16)
    }

    fn flw_instruction(a: usize, b: usize, c: usize) -> Instruction<F> {
        Instruction::from_usize(
            FpOpcode::FLW.global_opcode(),
            [a, b, c, FP_AS as usize, MEM_AS as usize, 0, 0],
        )
    }

    #[test]
    fn single_flw_proves() {
        let mut tester = VmChipTestBuilder::default();
        let mut harness = create_harness(&mut tester);
        tester.write(1, 4, [F::ZERO, F::from_u32(0x10), F::ZERO, F::ZERO]); // x1 = 0x1000
        tester.write(
            MEM_AS as usize,
            0x1000,
            [F::ZERO, F::ZERO, F::from_u32(0x60), F::from_u32(0x40)], // 3.5
        );
        let instr = flw_instruction(4 * 5, 4, 0);
        tester.execute(&mut harness.executor, &mut harness.arena, &instr);
        // After the load, FP f5 (= x5, AS1@20) must hold 3.5.
        let rd: [F; 4] = tester.read(1, 4 * 5);
        assert_eq!(
            rd,
            [F::ZERO, F::ZERO, F::from_u32(0x60), F::from_u32(0x40)],
            "FLW did not write 3.5 to f5"
        );
        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    }
}

type F = BabyBear;

struct TestI;
impl VmAdapterInterface<F> for TestI {
    type Reads = [[F; 4]; 1];
    type Writes = [[F; 4]; 1];
    type ProcessedInstruction = FpLoadStoreInstruction<F>;
}

fn build_row(data: [u8; 4], opcode: FpOpcode, is_valid: bool) -> Vec<F> {
    let mut buf = vec![F::ZERO; FmemCoreCols::<F>::width()];
    let row: &mut FmemCoreCols<F> = buf.as_mut_slice().borrow_mut();
    for (i, &b) in data.iter().enumerate() {
        row.data[i] = F::from_u8(b);
    }
    match opcode {
        FpOpcode::FLW => {
            row.flw_flag = F::ONE;
            row.fsw_flag = F::ZERO;
        }
        FpOpcode::FSW => {
            row.flw_flag = F::ZERO;
            row.fsw_flag = F::ONE;
        }
        _ => unreachable!(),
    }
    row.is_valid = F::from_bool(is_valid);
    buf
}

fn eval_row(buf: &[F]) -> NumericBuilder<F> {
    let air = FmemCoreAir::new(FpOpcode::CLASS_OFFSET);
    let mut b = NumericBuilder::new(vec![]);
    VmCoreAir::<NumericBuilder<F>, TestI>::eval(&air, &mut b, buf, F::ZERO);
    b
}

#[test]
fn honest_flw_passes() {
    let buf = build_row(3.5f32.to_bits().to_le_bytes(), FpOpcode::FLW, true);
    assert!(eval_row(&buf).ok(), "flw honest row should pass");
}

#[test]
fn honest_fsw_passes() {
    let buf = build_row((-2.25f32).to_bits().to_le_bytes(), FpOpcode::FSW, true);
    assert!(eval_row(&buf).ok(), "fsw honest row should pass");
}

#[test]
fn zero_padding_row_passes() {
    let mut buf = vec![F::ZERO; FmemCoreCols::<F>::width()];
    let row: &mut FmemCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.is_valid = F::ZERO;
    assert!(eval_row(&buf).ok(), "padding row should pass");
}

#[test]
fn both_flags_set_is_rejected() {
    let mut buf = build_row(1.0f32.to_bits().to_le_bytes(), FpOpcode::FLW, true);
    let row: &mut FmemCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.fsw_flag = F::ONE;
    assert!(
        !eval_row(&buf).ok(),
        "both opcode flags set must be rejected"
    );
}

#[test]
fn no_flag_set_is_rejected() {
    let mut buf = build_row(1.0f32.to_bits().to_le_bytes(), FpOpcode::FLW, true);
    let row: &mut FmemCoreCols<F> = buf.as_mut_slice().borrow_mut();
    row.flw_flag = F::ZERO;
    assert!(
        !eval_row(&buf).ok(),
        "zero opcode flags on a valid row must be rejected"
    );
}

#[test]
fn corruption_sweep_finds_no_unconstrained_witness() {
    use crate::test_utils::corruption_sweep;
    use openvm_circuit_primitives::StructReflectionHelper;

    let names = FmemCoreCols::<F>::struct_reflection().unwrap_or_default();
    // `data` is the value the adapter moves between AS 2 and the FP register
    // file: it is pinned by the memory bus (the adapter's value read and write
    // both claim it), which NumericBuilder ignores.
    let whitelist: std::collections::HashSet<usize> = (0..4)
        .map(|i| {
            names
                .iter()
                .position(|n| n == &format!("data__{i}"))
                .unwrap()
        })
        .collect();
    for (data, opcode) in [
        (3.5f32.to_bits().to_le_bytes(), FpOpcode::FLW),
        ((-2.25f32).to_bits().to_le_bytes(), FpOpcode::FSW),
        ([0x00, 0xFF, 0x80, 0x7F], FpOpcode::FLW),
    ] {
        let row = build_row(data, opcode, true);
        let accepted = corruption_sweep(&row, |builder, cols| {
            VmCoreAir::<NumericBuilder<F>, TestI>::eval(
                &FmemCoreAir::new(FpOpcode::CLASS_OFFSET),
                builder,
                cols,
                F::ZERO,
            );
        });
        let unexpected: Vec<usize> = accepted
            .into_iter()
            .filter(|i| !whitelist.contains(i))
            .collect();
        assert!(
            unexpected.is_empty(),
            "fmem opcode={opcode:?}: unconstrained columns {:?}",
            unexpected
                .iter()
                .map(|&i| (i, names.get(i).cloned().unwrap_or_default()))
                .collect::<Vec<_>>()
        );
    }
}
