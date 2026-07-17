use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FsgnjAir = VmAirWrapper<Rv32FpAdapterAir, FsgnjCoreAir>;
pub type Rv32FsgnjExecutor = FsgnjExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FsgnjChip<F> = VmChipWrapper<F, FsgnjFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
