//! Tests for the fsqrt.s chip.
//!
//! `fsqrt_s` uses native `f32::sqrt` (x86 `sqrtss`), which is the unique
//! IEEE correctly-rounded result for all valid inputs. The fuzz reference is
//! an independent software sqrt using exact integer isqrt + RNE rounding,
//! cross-validating bit-for-bit.

#![allow(dead_code)]

use openvm_rv32f_guest::CANONICAL_NAN;

use super::*;

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;
const POS_INF: u32 = 0x7F80_0000;
const NEG_INF: u32 = 0xFF80_0000;

// ════════════════════════════════════════════════════════════════════════════
// Exact integer sqrt reference (independent of native f32::sqrt)
// ════════════════════════════════════════════════════════════════════════════

fn decompose(bits: u32) -> (u32, u32, u32) {
    ((bits >> 31) & 1, (bits >> 23) & 0xFF, bits & 0x7FFFFF)
}

/// floor(sqrt(n)) for n < 2^128.
fn isqrt_u128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // Over-estimate: a power of two >= sqrt(n).
    let mut x = 1u128 << ((128 - n.leading_zeros() as usize).div_ceil(2));
    // Newton's method converges from above to floor(sqrt(n)).
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            break;
        }
        x = y;
    }
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
}

/// Normalize a significand to 24 bits [2^23, 2^24); returns (sig, exponent
/// of the LSB).
fn norm_sig(exp: u32, mant: u32) -> (u64, i32) {
    if exp == 0 {
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

/// Exact correctly-rounded (RNE) square root for positive finite inputs.
fn soft_sqrt_finite(bits: u32) -> u32 {
    let (_, exp, mant) = decompose(bits);
    let (mut sig, mut e) = norm_sig(exp, mant);
    // value = sig * 2^e; make e even so the root exponent is exact
    if e % 2 != 0 {
        sig <<= 1;
        e -= 1;
    }
    let t = e / 2;
    // root value = sqrt(sig) * 2^t = q * 2^(t - 24), q = floor(sqrt(sig*2^48))
    let n = (sig as u128) << 48;
    let q = isqrt_u128(n);
    let sticky = q * q != n;
    let mag = q;
    let scale = t - 24;

    let msb = 127 - mag.leading_zeros();
    let mut res_exp = msb as i32 + scale + 127;
    debug_assert!((1..255).contains(&res_exp), "sqrt result must be normal");
    let shift = msb as i32 - 23;
    let mut sig24;
    if shift >= 0 {
        let sh = shift as u32;
        sig24 = (mag >> sh) as u64;
        let guard = sh > 0 && ((mag >> (sh - 1)) & 1) == 1;
        let lost = sh > 1 && (mag & ((1u128 << (sh - 1)) - 1)) != 0;
        let sticky2 = sticky || lost;
        if guard && (sticky2 || (sig24 & 1) == 1) {
            sig24 += 1;
        }
    } else {
        sig24 = (mag << (-shift) as u32) as u64;
    }
    if sig24 >= (1 << 24) {
        sig24 >>= 1;
        res_exp += 1;
    }
    debug_assert!(res_exp < 255);
    ((res_exp as u32) << 23) | (sig24 as u32 & 0x7F_FFFF)
}

fn soft_sqrt(bits: u32) -> u32 {
    let (sign, exp, mant) = decompose(bits);
    let nan = exp == 0xFF && mant != 0;
    let inf = exp == 0xFF && mant == 0;
    let zero = exp == 0 && mant == 0;
    if nan || (sign == 1 && !zero) {
        return CANONICAL_NAN;
    }
    if inf {
        return POS_INF;
    }
    if zero {
        return bits; // +-0
    }
    soft_sqrt_finite(bits)
}

// ════════════════════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fsqrt_basic() {
    assert_eq!(fsqrt_s(4.0f32.to_bits()), 2.0f32.to_bits());
    assert_eq!(fsqrt_s(2.0f32.to_bits()), f32::sqrt(2.0).to_bits());
    assert_eq!(fsqrt_s(0.25f32.to_bits()), 0.5f32.to_bits());
    assert_eq!(fsqrt_s(1e10f32.to_bits()), 1e5f32.to_bits());
    assert_eq!(fsqrt_s(1.0f32.to_bits()), 1.0f32.to_bits());
}

#[test]
fn test_fsqrt_special() {
    assert_eq!(fsqrt_s(POS_ZERO), POS_ZERO);
    assert_eq!(fsqrt_s(NEG_ZERO), NEG_ZERO); // sqrt(-0.0) = -0.0
    assert_eq!(fsqrt_s(POS_INF), POS_INF);
    // negative nonzero (incl. -inf and negative subnormals) -> canonical NaN
    assert_eq!(fsqrt_s((-1.0f32).to_bits()), CANONICAL_NAN);
    assert_eq!(fsqrt_s(NEG_INF), CANONICAL_NAN);
    assert_eq!(fsqrt_s(0x8000_0001), CANONICAL_NAN);
    assert_eq!(fsqrt_s(CANONICAL_NAN), CANONICAL_NAN);
    assert_eq!(fsqrt_s(0x7FA0_0001), CANONICAL_NAN);
}

#[test]
fn test_fsqrt_subnormal_input() {
    // sqrt(min subnormal = 2^-149) = 2^-74.5 (normal result)
    let r = fsqrt_s(0x0000_0001);
    assert_eq!(r, f32::from_bits(0x0000_0001).sqrt().to_bits());
    assert_eq!(r, soft_sqrt(0x0000_0001));
    // sqrt(min normal / 2^24-ish subnormal)
    assert_eq!(fsqrt_s(0x007F_FFFF), soft_sqrt(0x007F_FFFF));
}

#[test]
fn test_soft_sqrt_unit() {
    assert_eq!(soft_sqrt(4.0f32.to_bits()), 2.0f32.to_bits());
    assert_eq!(soft_sqrt(2.0f32.to_bits()), f32::sqrt(2.0).to_bits());
    assert_eq!(soft_sqrt(0.25f32.to_bits()), 0.5f32.to_bits());
    // perfect squares near the 24-bit significand boundary
    assert_eq!(soft_sqrt((16777216.0f32).to_bits()), 4096.0f32.to_bits());
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
fn fuzz_fsqrt() {
    let mut rng = XorShift64::new(0x5E17_2025_000F);
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
    for i in 0..300_000u32 {
        let bits = if i.is_multiple_of(3) {
            specials[(rng.next_u32() as usize) % specials.len()]
        } else {
            rng.next_u32()
        };
        assert_eq!(
            fsqrt_s(bits),
            soft_sqrt(bits),
            "fsqrt({bits:#010x} = {})",
            f32::from_bits(bits)
        );
    }
}

/// Additionally verify against perfect squares and their neighbors, where
/// RNE boundary behavior is the trickiest.
#[test]
fn fuzz_fsqrt_perfect_squares() {
    let mut rng = XorShift64::new(0x5E17_2025_0010);
    for _ in 0..50_000 {
        // random 12-bit root -> exact 24-bit square
        let r = (rng.next_u32() & 0xFFF) as u64 | 0x800;
        let sq = (r * r) as u32;
        // place the square as a 24-bit significand
        for shift in 0..2u32 {
            let mant = sq << shift;
            if !(1 << 23..1 << 24).contains(&mant) {
                continue;
            }
            let bits = (126u32 << 23) | (mant & 0x7FFFFF);
            assert_eq!(fsqrt_s(bits), soft_sqrt(bits), "fsqrt({bits:#010x})");
        }
    }
}
