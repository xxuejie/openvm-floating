use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    Rv32FpUnaryAdapterAir, Rv32FpUnaryAdapterExecutor, Rv32FpUnaryAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

pub type Rv32FclassAir = VmAirWrapper<Rv32FpUnaryAdapterAir, FclassCoreAir>;
pub type Rv32FclassExecutor = FclassExecutor<Rv32FpUnaryAdapterExecutor>;
pub type Rv32FclassChip<F> = VmChipWrapper<F, FclassFiller<Rv32FpUnaryAdapterFiller>>;

#[cfg(test)]
mod tests;
