use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    Rv32FpUnaryAdapterAir, Rv32FpUnaryAdapterExecutor, Rv32FpUnaryAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

pub type Rv32FsqrtAir = VmAirWrapper<Rv32FpUnaryAdapterAir, FsqrtCoreAir>;
pub type Rv32FsqrtExecutor = FsqrtExecutor<Rv32FpUnaryAdapterExecutor>;
pub type Rv32FsqrtChip<F> = VmChipWrapper<F, FsqrtFiller<Rv32FpUnaryAdapterFiller>>;

#[cfg(test)]
mod tests;
