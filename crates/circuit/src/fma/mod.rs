use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32Fp3AdapterAir, Rv32Fp3AdapterExecutor, Rv32Fp3AdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FmaAir = VmAirWrapper<Rv32Fp3AdapterAir, FmaCoreAir>;
pub type Rv32FmaExecutor = FmaExecutor<Rv32Fp3AdapterExecutor>;
pub type Rv32FmaChip<F> = VmChipWrapper<F, FmaFiller<Rv32Fp3AdapterFiller>>;

#[cfg(test)]
mod tests;
