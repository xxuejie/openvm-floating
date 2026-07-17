//! Tests for the fcvt.s.w / fcvt.s.wu chip (int -> float conversions).

#![allow(dead_code)]

use super::*;

#[test]
fn test_fcvt_s_w_basic() {
    assert_eq!(fcvt_s_w(0), 0.0f32.to_bits());
    assert_eq!(fcvt_s_w(42), 42.0f32.to_bits());
    assert_eq!(fcvt_s_w((-42i32) as u32), (-42.0f32).to_bits());
    assert_eq!(fcvt_s_w(1), 1.0f32.to_bits());
    assert_eq!(fcvt_s_w(i32::MAX as u32), 2147483648.0f32.to_bits());
    assert_eq!(fcvt_s_w(i32::MIN as u32), (-2147483648.0f32).to_bits());
}

#[test]
fn test_fcvt_s_w_rne_rounding() {
    // 2^24 + 1 is exactly between 2^24 and 2^24 + 2; RNE ties-to-even -> 2^24
    assert_eq!(fcvt_s_w(0x0100_0001), 16777216.0f32.to_bits());
    // 2^24 + 3 rounds to 2^24 + 4 (2^24+3 is the midpoint of +2 and +4)
    assert_eq!(fcvt_s_w(0x0100_0003), 16777220.0f32.to_bits());
    // negative: -(2^24 + 1) -> -(2^24) (ties to even)
    assert_eq!(fcvt_s_w((-16777217i32) as u32), (-16777216.0f32).to_bits());
}

#[test]
fn test_fcvt_s_wu_basic() {
    assert_eq!(fcvt_s_wu(0), 0.0f32.to_bits());
    assert_eq!(fcvt_s_wu(42), 42.0f32.to_bits());
    // 2^32 - 1 rounds up to 2^32
    assert_eq!(fcvt_s_wu(u32::MAX), 4294967296.0f32.to_bits());
    assert_eq!(fcvt_s_wu(0x8000_0000), 2147483648.0f32.to_bits());
}

#[test]
fn test_fcvt_s_wu_rne_rounding() {
    assert_eq!(fcvt_s_wu(0x0100_0001), 16777216.0f32.to_bits());
    assert_eq!(fcvt_s_wu(0x0100_0003), 16777220.0f32.to_bits());
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
fn fuzz_fcvt_i2f() {
    let mut rng = XorShift64::new(0x12F0_2025_000C);
    for _ in 0..200_000 {
        let x = rng.next_u32();
        // reference: round via i64/f64 intermediate (f64 is exact for i32/u32)
        let ref_s = ((x as i32) as f64).round_ties_even() as f32;
        let ref_su = (x as f64).round_ties_even() as f32;
        assert_eq!(f32::from_bits(fcvt_s_w(x)), ref_s, "fcvt.s.w({x:#010x})");
        assert_eq!(f32::from_bits(fcvt_s_wu(x)), ref_su, "fcvt.s.wu({x:#010x})");
    }
}
