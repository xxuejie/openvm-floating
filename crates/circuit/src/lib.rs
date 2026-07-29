#![allow(unexpected_cfgs)]

use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmField, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::VmConfig;
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_rv32f_guest::RV32_FP_REGISTER_AS;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_sha2_circuit::{Sha2, Sha2CpuProverExt, Sha2Executor};
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

pub mod adapters;
pub mod extension;
pub mod fadd;
pub mod fclass;
pub mod fcmp;
pub mod fcvt_f2i;
pub mod fcvt_i2f;
pub mod fdiv;
pub mod fma;
pub mod fminmax;
pub mod fmul;
pub mod fsgnj;
pub mod fsqrt;
pub mod fsub;
pub mod fxfer;

#[cfg(test)]
mod test_utils;

pub use extension::*;

/// VM config for rv32imf (base + mul + io + F extension).
#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32FConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub f: Rv32F,
    #[extension]
    pub sha2: Sha2,
}

impl InitFileGenerator for Rv32FConfig {}

impl Default for Rv32FConfig {
    fn default() -> Self {
        Self::new_zfinx()
    }
}

impl Rv32FConfig {
    /// Zfinx mode: floats in integer registers (AS 1). No FP register file needed.
    pub fn new_zfinx() -> Self {
        Self {
            system: SystemConfig::default(),
            base: Default::default(),
            io: Default::default(),
            mul: Default::default(),
            f: Rv32F::default(),
            sha2: Sha2,
        }
    }

    /// F extension mode: separate FP register file in AS 5.
    pub fn new_f() -> Self {
        use openvm_circuit::arch::MemoryCellType;
        let mut system = SystemConfig::default();
        // Allocate FP register file: 32 registers × 4 bytes = 128 cells
        system.memory_config.addr_spaces[RV32_FP_REGISTER_AS as usize].num_cells = 32 * 4;
        system.memory_config.addr_spaces[RV32_FP_REGISTER_AS as usize].layout = MemoryCellType::U8;
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
            mul: Default::default(),
            f: Rv32F {
                range_tuple_checker_sizes: Default::default(),
                fp_register_as: RV32_FP_REGISTER_AS,
            },
            sha2: Sha2,
        }
    }
}

/// CPU builder for rv32imf — chains system + base + io + mul + F prover extensions.
#[derive(Clone)]
pub struct Rv32FCpuBuilder;

impl<SC, E> VmBuilder<E> for Rv32FCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv32FConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Rv32FConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.mul, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32FCpuProverExt, &config.f, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, &config.sha2, inventory)?;
        Ok(chip_complex)
    }
}
