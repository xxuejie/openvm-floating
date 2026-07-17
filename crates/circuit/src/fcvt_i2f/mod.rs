use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    Rv32FpUnaryAdapterAir, Rv32FpUnaryAdapterExecutor, Rv32FpUnaryAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

pub type Rv32FcvtI2FAir = VmAirWrapper<Rv32FpUnaryAdapterAir, FcvtI2FCoreAir>;
pub type Rv32FcvtI2FExecutor = FcvtI2FExecutor<Rv32FpUnaryAdapterExecutor>;
pub type Rv32FcvtI2FChip<F> = VmChipWrapper<F, FcvtI2FFiller<Rv32FpUnaryAdapterFiller>>;

#[cfg(test)]
mod tests;
