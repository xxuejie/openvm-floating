use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
pub use core::*;

pub type Rv32FmulAir = VmAirWrapper<Rv32FpAdapterAir, FmulCoreAir>;
pub type Rv32FmulExecutor = FmulExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FmulChip<F> = VmChipWrapper<F, FmulFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
