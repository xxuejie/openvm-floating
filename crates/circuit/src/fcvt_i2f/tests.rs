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

/// Neighbors of a nonzero finite f32 in value order (toward +inf / toward -inf).
fn f32_neighbors(bits: u32) -> (f32, f32) {
    debug_assert!(bits != 0);
    let (up, down) = if bits & 0x8000_0000 == 0 {
        (bits + 1, bits - 1)
    } else {
        (bits - 1, bits + 1)
    };
    (f32::from_bits(up), f32::from_bits(down))
}

/// Property check (independent of the implementation): the result must be a
/// nearest f32 to the exact integer value, with ties resolved to even
/// mantissa. All comparisons are done in f64, which is exact here.
fn check_fcvt_i2f(x: u32, is_wu: bool, result_bits: u32) {
    let xd = if is_wu { x as f64 } else { (x as i32) as f64 };
    let y = f32::from_bits(result_bits);
    assert!(!y.is_nan() && !y.is_infinite(), "fcvt({x:#010x}) -> {y}");
    if xd == 0.0 {
        assert_eq!(result_bits, 0, "fcvt(0) must be +0.0");
        return;
    }
    assert_eq!(y.is_sign_negative(), xd < 0.0, "fcvt({x:#010x}) sign");
    let err = (y as f64 - xd).abs();
    let (up, down) = f32_neighbors(result_bits);
    let up_err = (up as f64 - xd).abs();
    let down_err = (xd - down as f64).abs();
    assert!(
        err <= up_err && err <= down_err,
        "fcvt({x:#010x}) = {y}: not a nearest f32 (up {up}, down {down})"
    );
    if err == up_err || err == down_err {
        assert_eq!(
            result_bits & 1,
            0,
            "fcvt({x:#010x}) = {y}: midpoint tie not to even mantissa"
        );
    }
}

#[test]
fn fuzz_fcvt_i2f() {
    let mut rng = XorShift64::new(0x12F0_2025_000C);
    let specials = [
        0,
        1,
        u32::MAX,
        i32::MIN as u32,
        i32::MAX as u32,
        0x0100_0001, // 2^24 + 1 (tie)
        0x0100_0003, // 2^24 + 3 (tie)
        (-16777217i32) as u32,
        (-16777219i32) as u32,
    ];
    for i in 0..200_000 {
        let x = if i % 4 == 0 {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        check_fcvt_i2f(x, false, fcvt_s_w(x));
        check_fcvt_i2f(x, true, fcvt_s_wu(x));
    }
}
