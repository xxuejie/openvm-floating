use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv32FcmpAdapterAir, Rv32FcmpAdapterExecutor, Rv32FcmpAdapterFiller};

mod core;
mod execution;
pub use core::*;

pub type Rv32FcmpAir = VmAirWrapper<Rv32FcmpAdapterAir, FcmpCoreAir>;
pub type Rv32FcmpExecutor = FcmpExecutor<Rv32FcmpAdapterExecutor>;
pub type Rv32FcmpChip<F> = VmChipWrapper<F, FcmpFiller<Rv32FcmpAdapterFiller>>;

#[cfg(test)]
mod tests;
