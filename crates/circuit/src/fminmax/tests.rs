//! Tests for the fmin.s / fmax.s chip (min/max family).
//!
//! Verifies the key-mapping based implementation (`fminmax_result`) against an
//! independent reference written with native f32 comparisons.

#![allow(dead_code)]

use openvm_rv32f_guest::CANONICAL_NAN;

use super::{fminmax_result, fp_key};

/// Independent reference: explicit f32 comparison with RISC-V NaN and ±0 rules.
fn ref_minmax(a: u32, b: u32, is_min: bool) -> u32 {
    let a_f = f32::from_bits(a);
    let b_f = f32::from_bits(b);
    let a_nan = a_f.is_nan();
    let b_nan = b_f.is_nan();
    if a_nan && b_nan {
        CANONICAL_NAN
    } else if a_nan {
        b
    } else if b_nan {
        a
    } else {
        // For non-NaN floats, a_f == b_f only when the values compare equal;
        // the only distinct bit patterns with equal values are +0.0 / -0.0.
        let a_less = a_f < b_f || (a_f == b_f && a_f.is_sign_negative() && b_f.is_sign_positive());
        let b_less = b_f < a_f || (a_f == b_f && b_f.is_sign_negative() && a_f.is_sign_positive());
        if is_min {
            if b_less {
                b
            } else {
                a // a_less or equal
            }
        } else if a_less {
            b
        } else {
            a // b_less or equal
        }
    }
}

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;

#[test]
fn test_fmin_basic() {
    let a = 3.0f32.to_bits();
    let b = 4.0f32.to_bits();
    assert_eq!(fminmax_result(a, b, true), a);
    assert_eq!(fminmax_result(a, b, false), b);
    // symmetric
    assert_eq!(fminmax_result(b, a, true), a);
    assert_eq!(fminmax_result(b, a, false), b);
}

#[test]
fn test_fminmax_negatives() {
    let neg3 = (-3.0f32).to_bits();
    let neg4 = (-4.0f32).to_bits();
    let pos4 = 4.0f32.to_bits();
    assert_eq!(fminmax_result(neg3, pos4, true), neg3);
    assert_eq!(fminmax_result(neg3, pos4, false), pos4);
    assert_eq!(fminmax_result(neg3, neg4, true), neg4);
    assert_eq!(fminmax_result(neg3, neg4, false), neg3);
}

#[test]
fn test_fminmax_signed_zero() {
    // RISC-V: fmin(-0.0, +0.0) = -0.0, fmax(-0.0, +0.0) = +0.0
    assert_eq!(fminmax_result(NEG_ZERO, POS_ZERO, true), NEG_ZERO);
    assert_eq!(fminmax_result(POS_ZERO, NEG_ZERO, true), NEG_ZERO);
    assert_eq!(fminmax_result(NEG_ZERO, POS_ZERO, false), POS_ZERO);
    assert_eq!(fminmax_result(POS_ZERO, NEG_ZERO, false), POS_ZERO);
}

#[test]
fn test_fminmax_infinity() {
    let inf = f32::INFINITY.to_bits();
    let neg_inf = f32::NEG_INFINITY.to_bits();
    let one = 1.0f32.to_bits();
    assert_eq!(fminmax_result(inf, one, true), one);
    assert_eq!(fminmax_result(inf, one, false), inf);
    assert_eq!(fminmax_result(neg_inf, one, true), neg_inf);
    assert_eq!(fminmax_result(neg_inf, one, false), one);
    assert_eq!(fminmax_result(neg_inf, inf, true), neg_inf);
    assert_eq!(fminmax_result(neg_inf, inf, false), inf);
}

#[test]
fn test_fminmax_nan() {
    let nan = CANONICAL_NAN;
    let snan = 0x7FA0_0000; // signaling NaN
    let one = 1.0f32.to_bits();
    // exactly one NaN -> the other operand (bit-exact)
    assert_eq!(fminmax_result(nan, one, true), one);
    assert_eq!(fminmax_result(one, nan, true), one);
    assert_eq!(fminmax_result(nan, one, false), one);
    assert_eq!(fminmax_result(one, nan, false), one);
    assert_eq!(fminmax_result(snan, one, true), one);
    assert_eq!(fminmax_result(one, snan, false), one);
    // both NaN -> canonical NaN
    assert_eq!(fminmax_result(nan, nan, true), CANONICAL_NAN);
    assert_eq!(fminmax_result(nan, snan, false), CANONICAL_NAN);
    assert_eq!(fminmax_result(snan, nan, true), CANONICAL_NAN);
}

#[test]
fn test_fminmax_equal() {
    let a = 2.5f32.to_bits();
    assert_eq!(fminmax_result(a, a, true), a);
    assert_eq!(fminmax_result(a, a, false), a);
}

#[test]
fn test_fp_key_monotone() {
    // -inf < -1 < -0 < +0 < 1 < +inf in key order
    let mut keys = [
        f32::NEG_INFINITY.to_bits(),
        (-1.0f32).to_bits(),
        NEG_ZERO,
        POS_ZERO,
        1.0f32.to_bits(),
        f32::INFINITY.to_bits(),
    ]
    .map(fp_key);
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            fp_key(f32::NEG_INFINITY.to_bits()),
            fp_key((-1.0f32).to_bits()),
            fp_key(NEG_ZERO),
            fp_key(POS_ZERO),
            fp_key(1.0f32.to_bits()),
            fp_key(f32::INFINITY.to_bits()),
        ]
    );
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
fn fuzz_fminmax() {
    let mut rng = XorShift64::new(0xF11D_2025_0008);
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
        assert_eq!(
            fminmax_result(a, b, true),
            ref_minmax(a, b, true),
            "fmin({a:#010x}, {b:#010x})"
        );
        assert_eq!(
            fminmax_result(a, b, false),
            ref_minmax(a, b, false),
            "fmax({a:#010x}, {b:#010x})"
        );
    }
}
