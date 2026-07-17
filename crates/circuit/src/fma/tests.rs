//! Tests for the fused multiply-add chip (fmadd/fmsub/fnmsub/fnmadd).
//!
//! The execution semantics (`fmadd_result`) use the host's `f32::mul_add`
//! (a true fused operation with one rounding). The fuzz reference below is an
//! independent exact software FMA implemented with integer arithmetic,
//! cross-validating both.

#![allow(dead_code)]

use openvm_rv32f_guest::CANONICAL_NAN;

use super::*;

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;
const POS_INF: u32 = 0x7F80_0000;
const NEG_INF: u32 = 0xFF80_0000;

// ════════════════════════════════════════════════════════════════════════════
// Exact software FMA reference (integer arithmetic, independent of mul_add)
// ════════════════════════════════════════════════════════════════════════════

fn decompose(bits: u32) -> (u32, u32, u32) {
    ((bits >> 31) & 1, (bits >> 23) & 0xFF, bits & 0x7FFFFF)
}

/// Exact fused multiply-add: RNE(a * b + c), RISC-V NaN rules.
fn soft_fma(a: u32, b: u32, c: u32) -> u32 {
    let (sa, ea, ma) = decompose(a);
    let (sb, eb, mb) = decompose(b);
    let (sc, ec, mc) = decompose(c);

    let a_nan = ea == 0xFF && ma != 0;
    let b_nan = eb == 0xFF && mb != 0;
    let c_nan = ec == 0xFF && mc != 0;
    let a_inf = ea == 0xFF && ma == 0;
    let b_inf = eb == 0xFF && mb == 0;
    let c_inf = ec == 0xFF && mc == 0;
    let a_zero = ea == 0 && ma == 0;
    let b_zero = eb == 0 && mb == 0;

    if a_nan || b_nan || c_nan {
        return CANONICAL_NAN;
    }
    if (a_inf && b_zero) || (b_inf && a_zero) {
        return CANONICAL_NAN; // 0 * inf
    }
    let prod_sign = sa ^ sb;
    if a_inf || b_inf {
        if c_inf && sc != prod_sign {
            return CANONICAL_NAN; // inf + (-inf)
        }
        return (prod_sign << 31) | 0x7F80_0000;
    }
    if c_inf {
        return c;
    }

    let sig_a = if ea == 0 {
        ma as u64
    } else {
        (ma | 0x80_0000) as u64
    };
    let sig_b = if eb == 0 {
        mb as u64
    } else {
        (mb | 0x80_0000) as u64
    };
    let sig_c = if ec == 0 {
        mc as u64
    } else {
        (mc | 0x80_0000) as u64
    };

    let prod = sig_a * sig_b; // exact, <= 48 bits
    let exp_of = |e: u32| -> i32 {
        if e == 0 {
            -126 - 23
        } else {
            e as i32 - 127 - 23
        }
    };
    let e_prod = exp_of(ea) + exp_of(eb); // prod value = prod * 2^e_prod
    let e_c = exp_of(ec); // sig_c value = sig_c * 2^e_c

    if prod == 0 {
        // product is +-0
        if sig_c == 0 {
            return if prod_sign == 1 && sc == 1 {
                NEG_ZERO
            } else {
                POS_ZERO
            };
        }
        return c;
    }

    let sp = if prod_sign == 1 {
        -(prod as i128)
    } else {
        prod as i128
    };
    let scv = if sc == 1 {
        -(sig_c as i128)
    } else {
        sig_c as i128
    };

    // Align the two addends at a common scale, tracking a single sticky bit.
    let (sum, scale, mut sticky) = if sig_c == 0 {
        // c is +-0: the sum is exactly the product (possibly tiny)
        (sp, e_prod, false)
    } else if e_prod >= e_c {
        let d = (e_prod - e_c) as u32;
        if d <= 70 {
            // shift product left to c's scale, full precision
            ((sp << d) + scv, e_c, false)
        } else {
            // c is negligible at the product's scale
            (sp, e_prod, sig_c != 0)
        }
    } else {
        let d = (e_c - e_prod) as u32;
        if d <= 70 {
            // shift c left to the product's scale, full precision
            (sp + (scv << d), e_prod, false)
        } else {
            // product is negligible at c's scale
            (scv, e_c, true)
        }
    };

    if sum == 0 {
        return POS_ZERO; // exact cancellation -> +0 in RNE
    }
    let neg = sum < 0;
    let mag = if neg { -sum } else { sum } as u128;
    let msb = 127 - mag.leading_zeros(); // index of the highest set bit
                                         // value ~= mag * 2^scale
    let mut exp = msb as i32 + scale + 127;
    let sign = if neg { 1u32 } else { 0 };
    if exp <= 0 {
        // Subnormal: mant satisfies value = mant * 2^-149, so round mag down
        // by t = -scale - 149 bits in a single rounding (no double rounding!).
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
        // If the round carries to 2^23, the bits already encode min-normal.
        return (sign << 31) | mant as u32;
    }
    // Normal: round mag to a 24-bit significand.
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

// ════════════════════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fmadd_basic() {
    let (a, b, c) = (3.0f32.to_bits(), 4.0f32.to_bits(), 1.0f32.to_bits());
    assert_eq!(fmadd_result(a, b, c, FmaddOp::Madd), 13.0f32.to_bits());
    assert_eq!(fmadd_result(a, b, c, FmaddOp::Msub), 11.0f32.to_bits());
    assert_eq!(fmadd_result(a, b, c, FmaddOp::Nmsub), (-11.0f32).to_bits());
    assert_eq!(fmadd_result(a, b, c, FmaddOp::Nmadd), (-13.0f32).to_bits());
}

#[test]
fn test_fmadd_fused_single_rounding() {
    // (1 + 2^-12)^2 - 1 = 2^-11 + 2^-24: the 2^-24 bit survives only with
    // fused (single) rounding; an unfused mul-then-add loses it.
    let a = (1.0f32 + 2f32.powi(-12)).to_bits(); // 1 + 2^-12
    let c = (-1.0f32).to_bits();
    let fused = fmadd_result(a, a, c, FmaddOp::Madd);
    let unfused = ((f32::from_bits(a) * f32::from_bits(a)) + f32::from_bits(c)).to_bits();
    assert_eq!(fused, (2f32.powi(-11) + 2f32.powi(-24)).to_bits());
    assert_ne!(
        fused, unfused,
        "this case must distinguish fused from unfused"
    );
    assert_eq!(unfused, 2f32.powi(-11).to_bits());
}

#[test]
fn test_fmadd_nan() {
    let one = 1.0f32.to_bits();
    for op in [FmaddOp::Madd, FmaddOp::Msub, FmaddOp::Nmsub, FmaddOp::Nmadd] {
        assert_eq!(fmadd_result(CANONICAL_NAN, one, one, op), CANONICAL_NAN);
        assert_eq!(fmadd_result(one, CANONICAL_NAN, one, op), CANONICAL_NAN);
        assert_eq!(fmadd_result(one, one, CANONICAL_NAN, op), CANONICAL_NAN);
        assert_eq!(fmadd_result(0x7FA0_0001, one, one, op), CANONICAL_NAN);
    }
}

#[test]
fn test_fmadd_inf() {
    let (one, two) = (1.0f32.to_bits(), 2.0f32.to_bits());
    assert_eq!(fmadd_result(POS_INF, two, one, FmaddOp::Madd), POS_INF);
    assert_eq!(fmadd_result(NEG_INF, two, one, FmaddOp::Madd), NEG_INF);
    // -(inf * 2) + 1 = -inf
    assert_eq!(fmadd_result(POS_INF, two, one, FmaddOp::Nmsub), NEG_INF);
    assert_eq!(fmadd_result(POS_INF, two, one, FmaddOp::Nmadd), NEG_INF);
    // 0 * inf is invalid
    assert_eq!(
        fmadd_result(POS_ZERO, POS_INF, one, FmaddOp::Madd),
        CANONICAL_NAN
    );
    assert_eq!(
        fmadd_result(POS_INF, NEG_ZERO, one, FmaddOp::Madd),
        CANONICAL_NAN
    );
    // inf + (-inf) is invalid
    assert_eq!(
        fmadd_result(POS_INF, one, NEG_INF, FmaddOp::Madd),
        CANONICAL_NAN
    );
    // -(inf * 1) + (+inf) = -inf + inf is invalid
    assert_eq!(
        fmadd_result(POS_INF, one, POS_INF, FmaddOp::Nmsub),
        CANONICAL_NAN
    );
    // (inf * 1) + (+inf) = inf
    assert_eq!(fmadd_result(POS_INF, one, POS_INF, FmaddOp::Madd), POS_INF);
    // -( -inf * 1) - (+inf) = inf - inf is invalid
    assert_eq!(
        fmadd_result(NEG_INF, one, POS_INF, FmaddOp::Nmadd),
        CANONICAL_NAN
    );
}

#[test]
fn test_fmadd_zero() {
    let (one, two) = (1.0f32.to_bits(), 2.0f32.to_bits());
    // exact cancellation -> +0
    assert_eq!(
        fmadd_result(one, two, (-2.0f32).to_bits(), FmaddOp::Madd),
        POS_ZERO
    );
    assert_eq!(
        fmadd_result(one, two, 2.0f32.to_bits(), FmaddOp::Msub),
        POS_ZERO
    );
    // -0 * 2 + -0 = -0
    assert_eq!(
        fmadd_result(NEG_ZERO, two, NEG_ZERO, FmaddOp::Madd),
        NEG_ZERO
    );
    // -0 * 2 + +0 = +0
    assert_eq!(
        fmadd_result(NEG_ZERO, two, POS_ZERO, FmaddOp::Madd),
        POS_ZERO
    );
}

#[test]
fn test_soft_fma_unit() {
    // spot-check the reference itself
    let (a, b, c) = (3.0f32.to_bits(), 4.0f32.to_bits(), 1.0f32.to_bits());
    assert_eq!(soft_fma(a, b, c), 13.0f32.to_bits());
    // min_normal^2 underflows to +0, so the result is min_normal
    assert_eq!(soft_fma(0x0080_0000, 0x0080_0000, 0x0080_0000), 0x0080_0000);
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
fn fuzz_fmadd() {
    let mut rng = XorShift64::new(0xF4A4_2025_000D);
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
    for i in 0..200_000 {
        let pick = |rng: &mut XorShift64, i: usize| {
            if (i + rng.next_u32() as usize).is_multiple_of(3) {
                specials[(rng.next_u32() as usize) % specials.len()]
            } else {
                rng.next_u32()
            }
        };
        let a = pick(&mut rng, i);
        let b = pick(&mut rng, i + 1);
        let c = pick(&mut rng, i + 2);
        let expected = soft_fma(a, b, c);
        assert_eq!(
            fmadd_result(a, b, c, FmaddOp::Madd),
            expected,
            "fmadd({a:#010x}, {b:#010x}, {c:#010x})"
        );
        // -(a * b) - c == -(soft_fma with flipped signs)
        let flipped = soft_fma(a ^ 0x8000_0000, b, c ^ 0x8000_0000);
        assert_eq!(
            fmadd_result(a, b, c, FmaddOp::Nmadd),
            flipped,
            "fnmadd({a:#010x}, {b:#010x}, {c:#010x})"
        );
        let flipped = soft_fma(a ^ 0x8000_0000, b, c);
        assert_eq!(
            fmadd_result(a, b, c, FmaddOp::Nmsub),
            flipped,
            "fnmsub({a:#010x}, {b:#010x}, {c:#010x})"
        );
        let flipped = soft_fma(a, b, c ^ 0x8000_0000);
        assert_eq!(
            fmadd_result(a, b, c, FmaddOp::Msub),
            flipped,
            "fmsub({a:#010x}, {b:#010x}, {c:#010x})"
        );
    }
}
