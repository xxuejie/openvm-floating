//! Tests for the fcvt.w.s / fcvt.wu.s chip (float -> int conversions).

#![allow(dead_code)]

use super::*;

const CANON: u32 = 0x7FC0_0000;

#[test]
fn test_fcvt_w_s_basic() {
    assert_eq!(fcvt_w_s(3.7f32.to_bits()), 4);
    assert_eq!(fcvt_w_s((-3.7f32).to_bits()), (-4i32) as u32);
    assert_eq!(fcvt_w_s(0.0f32.to_bits()), 0);
    assert_eq!(fcvt_w_s((-0.0f32).to_bits()), 0);
}

#[test]
fn test_fcvt_w_s_rne_rounding() {
    // ties to even: 2.5 -> 2, 3.5 -> 4, -2.5 -> -2, -3.5 -> -4
    assert_eq!(fcvt_w_s(2.5f32.to_bits()), 2);
    assert_eq!(fcvt_w_s(3.5f32.to_bits()), 4);
    assert_eq!(fcvt_w_s((-2.5f32).to_bits()), (-2i32) as u32);
    assert_eq!(fcvt_w_s((-3.5f32).to_bits()), (-4i32) as u32);
    // non-ties round to nearest
    assert_eq!(fcvt_w_s(2.6f32.to_bits()), 3);
    assert_eq!(fcvt_w_s(2.4f32.to_bits()), 2);
    assert_eq!(fcvt_w_s(0.4f32.to_bits()), 0);
}

#[test]
fn test_fcvt_w_s_special() {
    assert_eq!(fcvt_w_s(CANON), i32::MAX as u32);
    assert_eq!(fcvt_w_s(0x7FA0_0001), i32::MAX as u32); // sNaN
    assert_eq!(fcvt_w_s(f32::INFINITY.to_bits()), i32::MAX as u32);
    assert_eq!(fcvt_w_s(f32::NEG_INFINITY.to_bits()), i32::MIN as u32);
    // out of range saturates
    assert_eq!(fcvt_w_s(3e9f32.to_bits()), i32::MAX as u32);
    assert_eq!(fcvt_w_s((-3e9f32).to_bits()), i32::MIN as u32);
    // boundaries: 2^31 saturates, -2^31 is in range
    assert_eq!(fcvt_w_s(2147483648.0f32.to_bits()), i32::MAX as u32);
    assert_eq!(fcvt_w_s((-2147483648.0f32).to_bits()), i32::MIN as u32);
    // just inside the boundary
    assert_eq!(
        fcvt_w_s((-2147483520.0f32).to_bits()),
        (-2147483520i32) as u32
    );
}

#[test]
fn test_fcvt_wu_s_basic() {
    assert_eq!(fcvt_wu_s(3.7f32.to_bits()), 4);
    assert_eq!(fcvt_wu_s(0.0f32.to_bits()), 0);
    assert_eq!(fcvt_wu_s(4294967040.0f32.to_bits()), 4294967040);
}

#[test]
fn test_fcvt_wu_s_negative_is_zero() {
    assert_eq!(fcvt_wu_s((-0.4f32).to_bits()), 0);
    assert_eq!(fcvt_wu_s((-0.5f32).to_bits()), 0); // RNE -> -0 -> 0
    assert_eq!(fcvt_wu_s((-3.0f32).to_bits()), 0);
    assert_eq!(fcvt_wu_s(f32::NEG_INFINITY.to_bits()), 0);
    assert_eq!(fcvt_wu_s((-3e9f32).to_bits()), 0);
}

#[test]
fn test_fcvt_wu_s_special() {
    assert_eq!(fcvt_wu_s(CANON), u32::MAX);
    assert_eq!(fcvt_wu_s(f32::INFINITY.to_bits()), u32::MAX);
    // 3e9 is in range for u32
    assert_eq!(fcvt_wu_s(3e9f32.to_bits()), 3000000000);
    // 5e9 saturates
    assert_eq!(fcvt_wu_s(5e9f32.to_bits()), u32::MAX);
}

#[test]
fn test_fcvt_f2i_result_dispatch() {
    assert_eq!(fcvt_f2i_result(3.7f32.to_bits(), false), 4);
    assert_eq!(fcvt_f2i_result((-3.7f32).to_bits(), true), 0);
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
fn fuzz_fcvt_f2i() {
    let mut rng = XorShift64::new(0xFC77_2025_000B);
    for _ in 0..200_000 {
        let bits = rng.next_u32();
        let f = f32::from_bits(bits);
        // independent reference via explicit comparisons
        let (ws, wus) = if f.is_nan() {
            (i32::MAX as u32, u32::MAX)
        } else {
            let r = f.round_ties_even();
            let ws = if r >= 2147483648.0 {
                i32::MAX as u32
            } else if r < -2147483648.0 {
                i32::MIN as u32
            } else {
                (r as i64) as i32 as u32
            };
            let wus = if r >= 4294967296.0 {
                u32::MAX
            } else if r <= 0.0 {
                0
            } else {
                r as u64 as u32
            };
            (ws, wus)
        };
        assert_eq!(fcvt_w_s(bits), ws, "fcvt.w.s({bits:#010x} = {f})");
        assert_eq!(fcvt_wu_s(bits), wus, "fcvt.wu.s({bits:#010x} = {f})");
    }
}
