//! Tests for the fdiv.s chip.
//!
//! The execution semantics (`fdiv_s`) use native IEEE division. The fuzz
//! reference below is an independent correctly-rounded division implemented
//! with integer long division (u128 quotient + remainder-based RNE),
//! cross-validating both.

#![allow(dead_code)]

use openvm_rv32f_guest::CANONICAL_NAN;

use super::*;

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;
const POS_INF: u32 = 0x7F80_0000;
const NEG_INF: u32 = 0xFF80_0000;

// ════════════════════════════════════════════════════════════════════════════
// Exact integer-division reference (independent of native `/`)
// ════════════════════════════════════════════════════════════════════════════

fn decompose(bits: u32) -> (u32, u32, u32) {
    ((bits >> 31) & 1, (bits >> 23) & 0xFF, bits & 0x7FFFFF)
}

/// Normalize a significand to 24 bits [2^23, 2^24); returns (sig, exponent
/// adjustment).
fn norm_sig(exp: u32, mant: u32) -> (u64, i32) {
    if exp == 0 {
        // subnormal: value = mant * 2^-149
        let mut sig = mant as u64;
        let mut adj = -126 - 23;
        while sig != 0 && sig < (1 << 23) {
            sig <<= 1;
            adj -= 1;
        }
        (sig, adj)
    } else {
        ((mant | 0x80_0000) as u64, exp as i32 - 127 - 23)
    }
}

/// Exact correctly-rounded (RNE) division for finite nonzero inputs.
fn soft_div_finite(a: u32, b: u32) -> u32 {
    let (sa, ea, ma) = decompose(a);
    let (sb, eb, mb) = decompose(b);
    let (sig_a, exp_a) = norm_sig(ea, ma);
    let (sig_b, exp_b) = norm_sig(eb, mb);
    // value = (sig_a / sig_b) * 2^(exp_a - exp_b)
    let num = (sig_a as u128) << 64;
    let q = num / (sig_b as u128);
    let rem = num % (sig_b as u128);
    let mut sticky = rem != 0;
    // mag = q at scale exp_a - exp_b - 64
    let mag = q;
    let scale = exp_a - exp_b - 64;
    let sign = sa ^ sb;

    // Normalize to 24-bit significand, RNE with remainder sticky.
    let msb = 127 - mag.leading_zeros();
    let mut exp = msb as i32 + scale + 127;
    if exp <= 0 {
        // Subnormal: value = mant * 2^-149, single rounding from mag.
        let t = -scale - 149;
        let (mut mant, guard, lost) = if t < 0 {
            (mag << (-t) as u32, false, false)
        } else if t >= 128 {
            (0u128, false, true)
        } else {
            let t = t as u32;
            (
                mag >> t,
                t > 0 && ((mag >> (t - 1)) & 1) == 1,
                t > 1 && (mag & ((1u128 << (t - 1)) - 1)) != 0,
            )
        };
        let sticky2 = sticky || lost;
        if guard && (sticky2 || (mant & 1) == 1) {
            mant += 1;
        }
        return (sign << 31) | mant as u32;
    }
    let shift = msb as i32 - 23;
    let mut sig24;
    if shift >= 0 {
        let sh = shift as u32;
        sig24 = (mag >> sh) as u64;
        let guard = sh > 0 && ((mag >> (sh - 1)) & 1) == 1;
        let lost = sh > 1 && (mag & ((1u128 << (sh - 1)) - 1)) != 0;
        sticky |= lost;
        if guard && (sticky || (sig24 & 1) == 1) {
            sig24 += 1;
        }
    } else {
        sig24 = (mag << (-shift) as u32) as u64;
    }
    if sig24 >= (1 << 24) {
        sig24 >>= 1;
        exp += 1;
    }
    if exp >= 255 {
        return (sign << 31) | 0x7F80_0000;
    }
    (sign << 31) | ((exp as u32) << 23) | (sig24 as u32 & 0x7F_FFFF)
}

/// Full division reference with RISC-V special-case rules.
fn soft_div(a: u32, b: u32) -> u32 {
    let (sa, ea, ma) = decompose(a);
    let (sb, eb, mb) = decompose(b);
    let _ = (sa, sb);
    let a_nan = ea == 0xFF && ma != 0;
    let b_nan = eb == 0xFF && mb != 0;
    let a_inf = ea == 0xFF && ma == 0;
    let b_inf = eb == 0xFF && mb == 0;
    let a_zero = ea == 0 && ma == 0;
    let b_zero = eb == 0 && mb == 0;
    if a_nan || b_nan || (a_inf && b_inf) || (a_zero && b_zero) {
        return CANONICAL_NAN;
    }
    let sign = sa ^ sb;
    if a_inf {
        return (sign << 31) | POS_INF;
    }
    if b_inf {
        return sign << 31;
    }
    if b_zero {
        return (sign << 31) | POS_INF;
    }
    if a_zero {
        return sign << 31;
    }
    soft_div_finite(a, b)
}

// ════════════════════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fdiv_basic() {
    assert_eq!(
        fdiv_s(12.0f32.to_bits(), 4.0f32.to_bits()),
        3.0f32.to_bits()
    );
    assert_eq!(
        fdiv_s(1.0f32.to_bits(), 3.0f32.to_bits()),
        (1.0f32 / 3.0f32).to_bits()
    );
    assert_eq!(
        fdiv_s((-7.0f32).to_bits(), 2.0f32.to_bits()),
        (-3.5f32).to_bits()
    );
    assert_eq!(
        fdiv_s(7.0f32.to_bits(), (-2.0f32).to_bits()),
        (-3.5f32).to_bits()
    );
}

#[test]
fn test_fdiv_special() {
    let one = 1.0f32.to_bits();
    // x / 0 = +-inf (sign of x), x / +-0 sign = sa^sb
    assert_eq!(fdiv_s(one, POS_ZERO), POS_INF);
    assert_eq!(fdiv_s(one, NEG_ZERO), NEG_INF);
    assert_eq!(fdiv_s((-1.0f32).to_bits(), POS_ZERO), NEG_INF);
    // inf / 0 = inf (not an invalid operation)
    assert_eq!(fdiv_s(POS_INF, POS_ZERO), POS_INF);
    assert_eq!(fdiv_s(POS_INF, NEG_ZERO), NEG_INF);
    // 0 / x = +-0
    assert_eq!(fdiv_s(POS_ZERO, one), POS_ZERO);
    assert_eq!(fdiv_s(POS_ZERO, (-1.0f32).to_bits()), NEG_ZERO);
    // 0 / 0 = canonical NaN (must NOT be classified as inf)
    assert_eq!(fdiv_s(POS_ZERO, POS_ZERO), CANONICAL_NAN);
    assert_eq!(fdiv_s(NEG_ZERO, NEG_ZERO), CANONICAL_NAN);
    assert_eq!(fdiv_s(POS_ZERO, NEG_ZERO), CANONICAL_NAN);
    // inf / x = +-inf, x / inf = +-0
    assert_eq!(fdiv_s(POS_INF, one), POS_INF);
    assert_eq!(fdiv_s(POS_INF, (-1.0f32).to_bits()), NEG_INF);
    assert_eq!(fdiv_s(NEG_INF, POS_ZERO), NEG_INF);
    assert_eq!(fdiv_s(one, POS_INF), POS_ZERO);
    assert_eq!(fdiv_s(one, NEG_INF), NEG_ZERO);
    // inf / inf = canonical NaN
    assert_eq!(fdiv_s(POS_INF, NEG_INF), CANONICAL_NAN);
    // NaN input = canonical NaN
    assert_eq!(fdiv_s(CANONICAL_NAN, one), CANONICAL_NAN);
    assert_eq!(fdiv_s(one, 0x7FA0_0001), CANONICAL_NAN);
}

#[test]
fn test_soft_div_unit() {
    assert_eq!(
        soft_div(12.0f32.to_bits(), 4.0f32.to_bits()),
        3.0f32.to_bits()
    );
    assert_eq!(
        soft_div(1.0f32.to_bits(), 3.0f32.to_bits()),
        (1.0f32 / 3.0f32).to_bits()
    );
    // subnormal result: min_normal / 2^30 -> ~2^-156 -> +0
    assert_eq!(soft_div(0x0080_0000, 0x4D00_0000), POS_ZERO);
    // min_normal / 2 -> subnormal 2^-127
    assert_eq!(soft_div(0x0080_0000, 2.0f32.to_bits()), 0x0040_0000);
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
fn fuzz_fdiv() {
    let mut rng = XorShift64::new(0xD1D1_2025_000E);
    let specials = [
        POS_ZERO,
        NEG_ZERO,
        CANONICAL_NAN,
        0x7FA0_0001,
        POS_INF,
        NEG_INF,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x0080_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x3F800000, // 1.0
        0xBF800000, // -1.0
    ];
    for i in 0..200_000u32 {
        let a = if i.is_multiple_of(3) {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        let b = if i.is_multiple_of(4) {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        assert_eq!(fdiv_s(a, b), soft_div(a, b), "fdiv({a:#010x}, {b:#010x})");
    }
}
