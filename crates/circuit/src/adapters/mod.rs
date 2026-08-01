mod fcmp;
mod fp;
mod fp3;
mod fp_move;
mod fp_unary;
pub use fcmp::*;
pub use fp::*;
pub use fp3::*;
pub use fp_move::*;
pub use fp_unary::*;

pub const RV32_FP_REGISTER_AS: u32 = 5;

pub const RV32_REGISTER_NUM_LIMBS: usize = 4;
pub const RV32_CELL_BITS: usize = 8;

pub const CANONICAL_NAN: u32 = 0x7FC00000;
pub const F32_EXP_BITS: usize = 8;
pub const F32_MANT_BITS: usize = 23;
pub const F32_BIAS: u32 = 127;
