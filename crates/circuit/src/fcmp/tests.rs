//! Tests for the feq.s / flt.s / fle.s chip (compare family).
//!
//! `fcmp_result` uses native f32 comparisons; the AIR instead proves the
//! result via monotone comparison keys and a both-zero special case
//! (-0.0 == +0.0). These tests cross-validate the two formulations.

#![allow(dead_code)]

use openvm_rv32f_guest::CANONICAL_NAN;

use super::{fcmp_result, FcmpOp};
use crate::fminmax::fp_key;

fn is_nan_bits(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0
}

/// Reference mirroring the AIR constraint logic: key comparison + zero detect.
fn ref_fcmp_keybased(a: u32, b: u32, op: FcmpOp) -> u32 {
    if is_nan_bits(a) || is_nan_bits(b) {
        return 0;
    }
    let ka = fp_key(a);
    let kb = fp_key(b);
    let lt_ab = ka < kb;
    let lt_ba = kb < ka;
    let eq = !lt_ab && !lt_ba;
    let both_zero = (a & 0x7FFF_FFFF == 0) && (b & 0x7FFF_FFFF == 0);
    match op {
        FcmpOp::Eq => (eq || both_zero) as u32,
        FcmpOp::Lt => (lt_ab && !both_zero) as u32,
        FcmpOp::Le => (!lt_ba || both_zero) as u32,
    }
}

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;

#[test]
fn test_feq_basic() {
    let two = 2.0f32.to_bits();
    let three = 3.0f32.to_bits();
    assert_eq!(fcmp_result(two, two, FcmpOp::Eq), 1);
    assert_eq!(fcmp_result(two, three, FcmpOp::Eq), 0);
    assert_eq!(fcmp_result(three, two, FcmpOp::Eq), 0);
}

#[test]
fn test_flt_basic() {
    let neg1 = (-1.0f32).to_bits();
    let one = 1.0f32.to_bits();
    assert_eq!(fcmp_result(neg1, one, FcmpOp::Lt), 1);
    assert_eq!(fcmp_result(one, neg1, FcmpOp::Lt), 0);
    assert_eq!(fcmp_result(one, one, FcmpOp::Lt), 0);
}

#[test]
fn test_fle_basic() {
    let neg1 = (-1.0f32).to_bits();
    let one = 1.0f32.to_bits();
    assert_eq!(fcmp_result(neg1, one, FcmpOp::Le), 1);
    assert_eq!(fcmp_result(one, one, FcmpOp::Le), 1);
    assert_eq!(fcmp_result(one, neg1, FcmpOp::Le), 0);
}

#[test]
fn test_fcmp_signed_zero() {
    // -0.0 == +0.0 for feq/flt/fle
    assert_eq!(fcmp_result(NEG_ZERO, POS_ZERO, FcmpOp::Eq), 1);
    assert_eq!(fcmp_result(POS_ZERO, NEG_ZERO, FcmpOp::Eq), 1);
    assert_eq!(fcmp_result(NEG_ZERO, POS_ZERO, FcmpOp::Lt), 0);
    assert_eq!(fcmp_result(POS_ZERO, NEG_ZERO, FcmpOp::Lt), 0);
    assert_eq!(fcmp_result(NEG_ZERO, POS_ZERO, FcmpOp::Le), 1);
    assert_eq!(fcmp_result(POS_ZERO, NEG_ZERO, FcmpOp::Le), 1);
}

#[test]
fn test_fcmp_nan() {
    let nan = CANONICAL_NAN;
    let snan = 0x7FA0_0001; // signaling NaN
    let one = 1.0f32.to_bits();
    for op in [FcmpOp::Eq, FcmpOp::Lt, FcmpOp::Le] {
        assert_eq!(fcmp_result(nan, one, op), 0);
        assert_eq!(fcmp_result(one, nan, op), 0);
        assert_eq!(fcmp_result(nan, nan, op), 0);
        assert_eq!(fcmp_result(snan, one, op), 0);
        assert_eq!(fcmp_result(one, snan, op), 0);
    }
}

#[test]
fn test_fcmp_infinity() {
    let inf = f32::INFINITY.to_bits();
    let neg_inf = f32::NEG_INFINITY.to_bits();
    let one = 1.0f32.to_bits();
    assert_eq!(fcmp_result(inf, one, FcmpOp::Lt), 0);
    assert_eq!(fcmp_result(one, inf, FcmpOp::Lt), 1);
    assert_eq!(fcmp_result(neg_inf, one, FcmpOp::Le), 1);
    assert_eq!(fcmp_result(inf, neg_inf, FcmpOp::Le), 0);
    assert_eq!(fcmp_result(inf, inf, FcmpOp::Eq), 1);
    assert_eq!(fcmp_result(neg_inf, inf, FcmpOp::Lt), 1);
}

struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x as u32
    }
}

#[test]
fn fuzz_fcmp() {
    let mut rng = XorShift64::new(0xCC0E_2025_0009);
    // Interesting patterns mixed in with random bits: NaNs, infs, zeros, subnormals.
    let specials = [
        POS_ZERO,
        NEG_ZERO,
        CANONICAL_NAN,
        0x7FA0_0001,
        0xFFC0_0000,
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x0000_0001,
        0x8000_0001,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
    ];
    for i in 0..200_000 {
        let a = if i % 4 == 0 {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        let b = if i % 3 == 0 {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        for op in [FcmpOp::Eq, FcmpOp::Lt, FcmpOp::Le] {
            assert_eq!(
                fcmp_result(a, b, op),
                ref_fcmp_keybased(a, b, op),
                "{op:?}({a:#010x}, {b:#010x})"
            );
        }
    }
}
