use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FaddAir = VmAirWrapper<Rv32FpAdapterAir, FaddCoreAir>;
pub type Rv32FaddExecutor = FaddExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FaddChip<F> = VmChipWrapper<F, FaddFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
