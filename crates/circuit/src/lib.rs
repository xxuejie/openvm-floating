#![allow(unexpected_cfgs)]

use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig},
    system::SystemExecutor,
};
use openvm_circuit_derive::VmConfig;
use openvm_rv32im_circuit::{Rv32I, Rv32IExecutor, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor};
use serde::{Deserialize, Serialize};

pub mod adapters;
pub mod extension;
pub mod fadd;
pub mod fmul;

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
}

impl InitFileGenerator for Rv32FConfig {}

impl Default for Rv32FConfig {
    fn default() -> Self {
        let f = Rv32F::default();
        let mut system = SystemConfig::default();
        // For F extension mode (AS 5), allocate the FP register file.
        // Zfinx mode (AS 1) reuses the existing integer register file — no allocation needed.
        if f.fp_register_as != 1 {
            use openvm_circuit::arch::MemoryCellType;
            system.memory_config.addr_spaces[f.fp_register_as as usize].num_cells = 32 * 4;
            system.memory_config.addr_spaces[f.fp_register_as as usize].layout = MemoryCellType::U8;
        }
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
            mul: Default::default(),
            f,
        }
    }
}
