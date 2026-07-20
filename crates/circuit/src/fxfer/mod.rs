use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FxferAir = VmAirWrapper<Rv32FpAdapterAir, FxferCoreAir>;
pub type Rv32FxferExecutor = FxferExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FxferChip<F> = VmChipWrapper<F, FxferFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
