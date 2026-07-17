use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    Rv32FpUnaryAdapterAir, Rv32FpUnaryAdapterExecutor, Rv32FpUnaryAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

pub type Rv32FcvtF2IAir = VmAirWrapper<Rv32FpUnaryAdapterAir, FcvtF2ICoreAir>;
pub type Rv32FcvtF2IExecutor = FcvtF2IExecutor<Rv32FpUnaryAdapterExecutor>;
pub type Rv32FcvtF2IChip<F> = VmChipWrapper<F, FcvtF2IFiller<Rv32FpUnaryAdapterFiller>>;

#[cfg(test)]
mod tests;
