use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::AnyEnum;
use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::LocalOpcode;
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{adapters::*, fadd::*, fmul::*};

/// RISC-V F (single-precision floating-point) extension config.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv32F {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

impl Default for Rv32F {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

/// F-extension executor enum.
#[derive(
    Clone, From, AnyEnum, openvm_circuit_derive::Executor, openvm_circuit_derive::PreflightExecutor,
)]
pub enum Rv32FExecutor {
    Fmul(Rv32FmulExecutor),
    Fadd(Rv32FaddExecutor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Rv32F {
    type Executor = Rv32FExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32FExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let fmul = Rv32FmulExecutor::new(Rv32FpAdapterExecutor, FpOpcode::CLASS_OFFSET);
        inventory.add_executor(fmul, [FpOpcode::FMUL_S.global_opcode()])?;

        let fadd = Rv32FaddExecutor::new(Rv32FpAdapterExecutor, FpOpcode::CLASS_OFFSET);
        inventory.add_executor(fadd, [FpOpcode::FADD_S.global_opcode()])?;
        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv32F {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);

        let range_tuple_checker = {
            let existing_air = inventory.find_air::<RangeTupleCheckerAir<2>>().find(|c| {
                c.bus.sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus.sizes[1] >= self.range_tuple_checker_sizes[1]
            });
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = RangeTupleCheckerBus::new(
                    inventory.new_bus_idx(),
                    self.range_tuple_checker_sizes,
                );
                let air = RangeTupleCheckerAir { bus };
                inventory.add_air(air);
                air.bus
            }
        };

        let fmul = Rv32FmulAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge),
            FmulCoreAir::new(range_tuple_checker, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fmul);

        let fadd = Rv32FaddAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge),
            FaddCoreAir::new(range_tuple_checker, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fadd);

        Ok(())
    }
}

/// CPU prover extension for F extension.
pub struct Rv32FCpuProverExt;

impl<SC, E, RA> VmProverExtension<E, RA, Rv32F> for Rv32FCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: openvm_circuit::arch::RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &Rv32F,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<SharedRangeTupleCheckerChip<2>>()
                .find(|c| {
                    c.bus().sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.bus().sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &RangeTupleCheckerAir<2> = inventory.next_air()?;
                let chip = SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv32FmulAir>()?;
        let fmul = Rv32FmulChip::new(
            FmulFiller::new(
                Rv32FpAdapterFiller,
                range_tuple_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fmul);

        inventory.next_air::<Rv32FaddAir>()?;
        let fadd = Rv32FaddChip::new(
            FaddFiller::new(
                Rv32FpAdapterFiller,
                range_tuple_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fadd);

        Ok(())
    }
}
