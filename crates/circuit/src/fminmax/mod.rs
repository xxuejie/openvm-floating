use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FminmaxAir = VmAirWrapper<Rv32FpAdapterAir, FminmaxCoreAir>;
pub type Rv32FminmaxExecutor = FminmaxExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FminmaxChip<F> = VmChipWrapper<F, FminmaxFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
