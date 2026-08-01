use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FpMoveAdapterAir, Rv32FpMoveAdapterExecutor, Rv32FpMoveAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FxferAir = VmAirWrapper<Rv32FpMoveAdapterAir, FxferCoreAir>;
pub type Rv32FxferExecutor = FxferExecutor<Rv32FpMoveAdapterExecutor>;
pub type Rv32FxferChip<F> = VmChipWrapper<F, FxferFiller<Rv32FpMoveAdapterFiller>>;

#[cfg(test)]
mod tests;
