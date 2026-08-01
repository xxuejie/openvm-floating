use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpMemAdapterAir, Rv32FpMemAdapterExecutor, Rv32FpMemAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FmemAir = VmAirWrapper<Rv32FpMemAdapterAir, FmemCoreAir>;
pub type Rv32FmemExecutor = FmemExecutor<Rv32FpMemAdapterExecutor>;
pub type Rv32FmemChip<F> = VmChipWrapper<F, FmemFiller<Rv32FpMemAdapterFiller>>;

#[cfg(test)]
mod tests;
