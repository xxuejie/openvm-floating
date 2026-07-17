use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FsubAir = VmAirWrapper<Rv32FpAdapterAir, FsubCoreAir>;
pub type Rv32FsubExecutor = FsubExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FsubChip<F> = VmChipWrapper<F, FsubFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
