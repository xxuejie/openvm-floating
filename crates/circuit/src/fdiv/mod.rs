use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpAdapterAir, Rv32FpAdapterExecutor, Rv32FpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FdivAir = VmAirWrapper<Rv32FpAdapterAir, FdivCoreAir>;
pub type Rv32FdivExecutor = FdivExecutor<Rv32FpAdapterExecutor>;
pub type Rv32FdivChip<F> = VmChipWrapper<F, FdivFiller<Rv32FpAdapterFiller>>;

#[cfg(test)]
mod tests;
