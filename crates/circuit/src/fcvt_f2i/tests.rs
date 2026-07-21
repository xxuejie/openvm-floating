//! Tests for the fcvt.w.s / fcvt.wu.s chip (float -> int conversions).

#![allow(dead_code)]

use super::*;

const CANON: u32 = 0x7FC0_0000;

#[test]
fn test_fcvt_w_s_basic() {
    assert_eq!(fcvt_w_s(3.7f32.to_bits()), 3); // truncation toward zero
    assert_eq!(fcvt_w_s((-3.7f32).to_bits()), (-3i32) as u32);
    assert_eq!(fcvt_w_s(0.0f32.to_bits()), 0);
    assert_eq!(fcvt_w_s((-0.0f32).to_bits()), 0);
}

#[test]
fn test_fcvt_w_s_rtz_rounding() {
    // truncation: 2.5 -> 2, 3.5 -> 3, -2.5 -> -2, -3.5 -> -3
    assert_eq!(fcvt_w_s(2.5f32.to_bits()), 2);
    assert_eq!(fcvt_w_s(3.5f32.to_bits()), 3);
    assert_eq!(fcvt_w_s((-2.5f32).to_bits()), (-2i32) as u32);
    assert_eq!(fcvt_w_s((-3.5f32).to_bits()), (-3i32) as u32);
    // non-integers truncate toward zero
    assert_eq!(fcvt_w_s(2.6f32.to_bits()), 2);
    assert_eq!(fcvt_w_s(2.4f32.to_bits()), 2);
    assert_eq!(fcvt_w_s(0.4f32.to_bits()), 0);
    assert_eq!(fcvt_w_s((-0.4f32).to_bits()), 0);
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
    assert_eq!(fcvt_wu_s(3.7f32.to_bits()), 3); // truncation toward zero
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
    assert_eq!(fcvt_f2i_result(3.7f32.to_bits(), false), 3); // truncation toward zero
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

/// Property check: verifies truncation (RTZ) semantics for fcvt.w.s.
/// NaN -> INT32_MAX, out-of-range saturates, in-range truncates toward zero.
fn check_fcvt_w_s(bits: u32, result: u32) {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        assert_eq!(result, i32::MAX as u32, "fcvt.w.s({bits:#010x}) is NaN");
        return;
    }
    let fd = f as f64;
    let r = result as i32;
    if r == i32::MAX {
        assert!(fd >= 2147483648.0, "fcvt.w.s({f}) = MAX but in range");
    } else if r == i32::MIN {
        assert!(fd <= -2147483648.0, "fcvt.w.s({f}) = MIN but in range");
    } else {
        // Truncation toward zero: |result| <= |f| and |f - result| < 1
        let expected = fd.trunc() as i32;
        assert_eq!(
            r, expected,
            "fcvt.w.s({f}) = {r}, expected trunc = {expected}"
        );
    }
}

/// Property check: verifies truncation (RTZ) semantics for fcvt.wu.s.
/// NaN -> UINT32_MAX, negative -> 0, saturates at UINT32_MAX.
fn check_fcvt_wu_s(bits: u32, result: u32) {
    let f = f32::from_bits(bits);
    if f.is_nan() {
        assert_eq!(result, u32::MAX, "fcvt.wu.s({bits:#010x}) is NaN");
        return;
    }
    let fd = f as f64;
    if result == u32::MAX {
        assert!(fd >= 4294967296.0, "fcvt.wu.s({f}) = MAX but in range");
    } else if result == 0 {
        assert!(fd < 1.0, "fcvt.wu.s({f}) = 0 but trunc >= 1");
    } else {
        let expected = fd.trunc() as u32;
        assert_eq!(
            result, expected,
            "fcvt.wu.s({f}) = {result}, expected trunc = {expected}"
        );
    }
}

#[test]
fn fuzz_fcvt_f2i() {
    let mut rng = XorShift64::new(0xFC77_2025_000B);
    // Interesting boundaries: saturation thresholds, tie points, small values.
    let specials = [
        0x0000_0000,
        0x8000_0000,
        CANON,
        0x7FA0_0001,
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        2147483648.0f32.to_bits(),
        (-2147483648.0f32).to_bits(),
        2147483520.0f32.to_bits(),
        4294967040.0f32.to_bits(),
        4294967296.0f32.to_bits(),
        0.5f32.to_bits(),
        (-0.5f32).to_bits(),
        1.5f32.to_bits(),
        2.5f32.to_bits(),
        3.5f32.to_bits(),
        3e9f32.to_bits(),
        (-3e9f32).to_bits(),
        5e9f32.to_bits(),
    ];
    for i in 0..200_000 {
        let bits = if i % 4 == 0 {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        check_fcvt_w_s(bits, fcvt_w_s(bits));
        check_fcvt_wu_s(bits, fcvt_wu_s(bits));
    }
}
