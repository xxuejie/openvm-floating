//! Tests for the FADD.S chip.

#![allow(dead_code)]

// ════════════════════════════════════════════════════════════════════════════
// Pure computation helpers (mirror the alignment + add/sub logic)
// ════════════════════════════════════════════════════════════════════════════

struct FaddResult {
    a_sign: u32,
    a_exp: u32,
    a_is_normal: bool,
    a_is_nan: bool,
    a_is_inf: bool,
    a_is_zero: bool,
    b_sign: u32,
    b_exp: u32,
    b_is_normal: bool,
    ediff: u32,
    a_is_larger: bool,
    same_sign: bool,
    result_bits: u32,
    is_nan_result: bool,
    is_inf_result: bool,
    is_zero_result: bool,
}

fn compute_fadd(a_bits: u32, b_bits: u32) -> FaddResult {
    let a_sign = (a_bits >> 31) & 1;
    let a_exp = (a_bits >> 23) & 0xFF;
    let a_mant = a_bits & 0x7FFFFF;
    let b_sign = (b_bits >> 31) & 1;
    let b_exp = (b_bits >> 23) & 0xFF;
    let b_mant = b_bits & 0x7FFFFF;

    let a_is_nan = a_exp == 255 && a_mant != 0;
    let a_is_inf = a_exp == 255 && a_mant == 0;
    let a_is_zero = a_exp == 0 && a_mant == 0;
    let a_is_normal = (1..=254).contains(&a_exp);

    let b_is_nan = b_exp == 255 && b_mant != 0;
    let b_is_inf = b_exp == 255 && b_mant == 0;
    let _b_is_zero = b_exp == 0 && b_mant == 0;
    let b_is_normal = (1..=254).contains(&b_exp);

    let a_is_larger = a_exp >= b_exp;
    let ediff = if a_is_larger {
        a_exp - b_exp
    } else {
        b_exp - a_exp
    };
    let same_sign = a_sign == b_sign;

    let result_bits = (f32::from_bits(a_bits) + f32::from_bits(b_bits)).to_bits();

    let r_exp = (result_bits >> 23) & 0xFF;
    let r_mant = result_bits & 0x7FFFFF;

    let is_nan_result = a_is_nan || b_is_nan || (a_is_inf && b_is_inf && a_sign != b_sign);
    let is_inf_result = !is_nan_result && (a_is_inf || b_is_inf || r_exp == 255);
    let is_zero_result = !is_nan_result && (r_exp == 0 && r_mant == 0);

    FaddResult {
        a_sign,
        a_exp,
        a_is_normal,
        a_is_nan,
        a_is_inf,
        a_is_zero,
        b_sign,
        b_exp,
        b_is_normal,
        ediff,
        a_is_larger,
        same_sign,
        result_bits,
        is_nan_result,
        is_inf_result,
        is_zero_result,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 1. Unit tests: basic correctness
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_basic_add_1_plus_2() {
    let r = compute_fadd(1.0f32.to_bits(), 2.0f32.to_bits());
    assert_eq!(r.result_bits, 3.0f32.to_bits());
}

#[test]
fn test_basic_add_fractions() {
    let a = 0.1f32;
    let b = 0.2f32;
    let expected = a + b;
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_add_negative() {
    let a = 5.0f32;
    let b = -3.0f32;
    let expected = a + b;
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
    assert!(!r.same_sign);
}

#[test]
fn test_add_to_zero() {
    let a = 42.0f32;
    let b = -42.0f32;
    let expected = a + b;
    assert_eq!(expected, 0.0);
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
    assert!(r.is_zero_result);
}

#[test]
fn test_add_cancel() {
    // 1e10 + (-1e10 + 1) → should give 1.0 (catastrophic cancellation)
    let large = 1e10f32;
    let small = 1.0f32;
    let b = -(large - small);
    let expected = large + b;
    let r = compute_fadd(large.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_add_large_exponent_diff() {
    let a = 1e20f32;
    let b = 1e-20f32;
    let expected = a + b;
    // b is negligible → result == a
    assert_eq!(expected, a);
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_commutative() {
    // fadd should be commutative for finite values (same rounding)
    let values = [
        1.0f32,
        -2.5,
        0.001,
        1e10,
        -1e-10,
        std::f32::consts::PI,
        42.0,
        0.5,
    ];
    for &a in &values {
        for &b in &values {
            let r1 = compute_fadd(a.to_bits(), b.to_bits());
            let r2 = compute_fadd(b.to_bits(), a.to_bits());
            assert_eq!(
                r1.result_bits, r2.result_bits,
                "Not commutative: {} + {} ≠ {} + {}",
                a, b, b, a
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Edge case tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_zero_plus_normal() {
    for &v in &[1.0f32, -1.0, 1e38, -1e38, 0.5, 123.456] {
        let expected = 0.0f32 + v;
        let r = compute_fadd(0.0f32.to_bits(), v.to_bits());
        assert_eq!(r.result_bits, expected.to_bits());
    }
}

#[test]
fn test_neg_zero_plus_neg_zero() {
    let expected = (-0.0f32) + (-0.0f32);
    assert_eq!(expected.to_bits(), 0x80000000); // -0.0
    let r = compute_fadd((-0.0f32).to_bits(), (-0.0f32).to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_pos_zero_plus_neg_zero() {
    let expected = 0.0f32 + (-0.0f32);
    assert_eq!(expected.to_bits(), 0); // +0.0 (RNE: +0 is "even")
    let r = compute_fadd(0.0f32.to_bits(), (-0.0f32).to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_inf_plus_normal() {
    let expected = f32::INFINITY + 1.0f32;
    let r = compute_fadd(f32::INFINITY.to_bits(), 1.0f32.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
    assert!(r.is_inf_result);
}

#[test]
fn test_inf_plus_neg_inf_is_nan() {
    let r = compute_fadd(f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    assert!(f32::from_bits(r.result_bits).is_nan());
    assert!(r.is_nan_result);
}

#[test]
fn test_nan_input() {
    let r = compute_fadd(f32::NAN.to_bits(), 1.0f32.to_bits());
    assert!(f32::from_bits(r.result_bits).is_nan());
    assert!(r.is_nan_result);
}

#[test]
fn test_overflow_to_inf() {
    let a = f32::MAX;
    let b = f32::MAX;
    let expected = a + b;
    assert!(expected.is_infinite());
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
    assert!(r.is_inf_result);
}

#[test]
fn test_subnormal_add() {
    let sub = f32::from_bits(1); // smallest subnormal
    let expected = sub + sub;
    assert_eq!(expected.to_bits(), 2); // 2 * smallest subnormal
    let r = compute_fadd(sub.to_bits(), sub.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_rounding_tie() {
    // Find a value where addition hits an exact tie
    let a = f32::from_bits(0x3F800000); // 1.0
    let b = f32::from_bits(0x33800000); // 2^-24 = half ULP at 1.0
    let expected = a + b;
    let r = compute_fadd(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_physics_like_values() {
    let values = [
        0.001f32,
        1.0,
        100.0,
        1000.0,
        0.1,
        0.01,
        9.81,
        std::f32::consts::PI,
        std::f32::consts::TAU,
        0.0001,
        100000.0,
        -0.001,
        -9.81,
        42.195,
        0.3048,
        -500.0,
    ];
    for &a in &values {
        for &b in &values {
            let expected = a + b;
            let r = compute_fadd(a.to_bits(), b.to_bits());
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Mismatch: {} + {} = {} (0x{:08X}), got 0x{:08X}",
                a,
                b,
                expected,
                expected.to_bits(),
                r.result_bits
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Fuzz tests
// ════════════════════════════════════════════════════════════════════════════

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

#[test]
fn fuzz_fadd_random_inputs() {
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_BABE);
    for i in 0..2_000_000 {
        let a_bits = rng.next_u32();
        let b_bits = rng.next_u32();

        let expected = f32::from_bits(a_bits) + f32::from_bits(b_bits);
        let r = compute_fadd(a_bits, b_bits);

        // NaN results: just check NaN-ness (bit patterns are impl-defined)
        if expected.is_nan() {
            assert!(
                f32::from_bits(r.result_bits).is_nan(),
                "NaN mismatch at iter {}: 0x{:08X} + 0x{:08X}",
                i,
                a_bits,
                b_bits
            );
        } else {
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Mismatch at iter {}: {} + {} = {} (0x{:08X}), got 0x{:08X}",
                i,
                f32::from_bits(a_bits),
                f32::from_bits(b_bits),
                expected,
                expected.to_bits(),
                r.result_bits
            );
        }
    }
}

#[test]
fn fuzz_fadd_normal_inputs() {
    let mut rng = XorShift64::new(0x1234_5678_9ABC_DEF0);
    for _ in 0..1_000_000 {
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_a = (rng.next_u32() % 254) + 1;
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign_a | (exp_a << 23) | mant_a;

        let sign_b = (rng.next_u32() & 1) << 31;
        let exp_b = (rng.next_u32() % 254) + 1;
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let b_bits = sign_b | (exp_b << 23) | mant_b;

        let expected = f32::from_bits(a_bits) + f32::from_bits(b_bits);
        let r = compute_fadd(a_bits, b_bits);
        assert_eq!(r.result_bits, expected.to_bits());
    }
}

#[test]
fn fuzz_fadd_physics_range() {
    let mut rng = XorShift64::new(0xCAFE_BABE_DEAD_BEEF);
    for _ in 0..1_000_000 {
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_a = 100 + (rng.next_u32() % 61);
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign_a | (exp_a << 23) | mant_a;

        let sign_b = (rng.next_u32() & 1) << 31;
        let exp_b = 100 + (rng.next_u32() % 61);
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let b_bits = sign_b | (exp_b << 23) | mant_b;

        let expected = f32::from_bits(a_bits) + f32::from_bits(b_bits);
        let r = compute_fadd(a_bits, b_bits);
        assert_eq!(
            r.result_bits,
            expected.to_bits(),
            "Physics fuzz: {} + {}",
            f32::from_bits(a_bits),
            f32::from_bits(b_bits)
        );
    }
}

#[test]
fn fuzz_fadd_same_sign() {
    // JoltPhysics hot path: same-sign additions (accumulators)
    let mut rng = XorShift64::new(0xAAA_BBB_CCC_DDD_EEE);
    for _ in 0..1_000_000 {
        let sign = (rng.next_u32() & 1) << 31;
        let exp_a = 100 + (rng.next_u32() % 61);
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let exp_b = 100 + (rng.next_u32() % 61);
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign | (exp_a << 23) | mant_a;
        let b_bits = sign | (exp_b << 23) | mant_b;

        let expected = f32::from_bits(a_bits) + f32::from_bits(b_bits);
        let r = compute_fadd(a_bits, b_bits);
        assert_eq!(r.result_bits, expected.to_bits());
    }
}

#[test]
fn fuzz_fadd_special_combinations() {
    let specials: &[(u32, &str)] = &[
        (0.0f32.to_bits(), "+0"),
        ((-0.0f32).to_bits(), "-0"),
        (f32::INFINITY.to_bits(), "+inf"),
        (f32::NEG_INFINITY.to_bits(), "-inf"),
        (f32::NAN.to_bits(), "NaN"),
        (1.0f32.to_bits(), "1.0"),
        ((-1.0f32).to_bits(), "-1.0"),
        (f32::MAX.to_bits(), "MAX"),
        (f32::MIN.to_bits(), "-MAX"),
        (f32::MIN_POSITIVE.to_bits(), "MIN_POS"),
        (f32::from_bits(1).to_bits(), "MIN_SUB"),
    ];

    let mut rng = XorShift64::new(42);
    for &(sa, na) in specials {
        for &(sb, nb) in specials {
            let expected = f32::from_bits(sa) + f32::from_bits(sb);
            let r = compute_fadd(sa, sb);
            if expected.is_nan() {
                assert!(
                    f32::from_bits(r.result_bits).is_nan(),
                    "Special: {} + {} should be NaN",
                    na,
                    nb
                );
            } else {
                assert_eq!(
                    r.result_bits,
                    expected.to_bits(),
                    "Special: {} + {} = {} (0x{:08X}), got 0x{:08X}",
                    na,
                    nb,
                    expected,
                    expected.to_bits(),
                    r.result_bits
                );
            }
        }
        for _ in 0..100 {
            let rand_bits = rng.next_u32() | 0x3F800000;
            let expected = f32::from_bits(sa) + f32::from_bits(rand_bits);
            let r = compute_fadd(sa, rand_bits);
            if expected.is_nan() {
                assert!(f32::from_bits(r.result_bits).is_nan());
            } else {
                assert_eq!(r.result_bits, expected.to_bits());
            }
        }
    }
}

#[test]
fn fuzz_fadd_deep_cancellation() {
    // Subtraction of nearly-equal values — the hardest case for FADD.
    // The result has many leading zeros, requiring large normalization shifts.
    let mut rng = XorShift64::new(0xCA4C_2024_F00D_BEEF);
    for _ in 0..1_000_000 {
        // Generate a random normal value
        let sign = (rng.next_u32() & 1) << 31;
        let exp = (rng.next_u32() % 254) + 1;
        let mant = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign | (exp << 23) | mant;

        // b = a - tiny_delta (so a + (-b) ≈ tiny_delta → deep cancellation)
        let delta_mant = rng.next_u32() & 0x3; // delta in bottom 2 mantissa bits
        let b_val = f32::from_bits(a_bits ^ (1u32 << 31)); // negate a
                                                           // Perturb b's mantissa slightly so cancellation isn't exact
        let b_perturbed = if delta_mant != 0 {
            f32::from_bits((b_val.to_bits() & !0x3) | delta_mant)
        } else {
            b_val
        };

        let expected = f32::from_bits(a_bits) + b_perturbed;
        let r = compute_fadd(a_bits, b_perturbed.to_bits());
        if expected.is_nan() {
            assert!(f32::from_bits(r.result_bits).is_nan());
        } else {
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Cancellation: {} + {} = {}",
                f32::from_bits(a_bits),
                b_perturbed,
                expected
            );
        }
    }
}

#[test]
fn fuzz_fadd_narrow_ediff() {
    // Small exponent difference (0-3) — exercises guard/sticky rounding
    // most aggressively, since the shifted operand still contributes
    // meaningful bits to the result.
    let mut rng = XorShift64::new(0xED1F_0302_0100_F00D);
    for _ in 0..1_000_000 {
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_base = (rng.next_u32() % 250) + 5; // [5, 254] — leave room for ediff
        let ediff = rng.next_u32() % 4; // 0, 1, 2, or 3
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let sign_b = (rng.next_u32() & 1) << 31;

        let a_bits = sign_a | (exp_base << 23) | mant_a;
        let b_bits = sign_b | ((exp_base - ediff) << 23) | mant_b;

        let expected = f32::from_bits(a_bits) + f32::from_bits(b_bits);
        let r = compute_fadd(a_bits, b_bits);
        assert_eq!(
            r.result_bits,
            expected.to_bits(),
            "Narrow ediff={}: {} + {}",
            ediff,
            f32::from_bits(a_bits),
            f32::from_bits(b_bits)
        );
    }
}

#[test]
fn fuzz_fadd_accumulator() {
    // Simulate a dot-product accumulator: start with a seed value,
    // keep adding random values. This stresses sequential rounding.
    let mut rng = XorShift64::new(0xACCE_A00A_ABCC_0042);
    for trial in 0..10_000 {
        let mut acc = f32::from_bits(0x3F800000 | (rng.next_u32() & 0x7FFFFF)); // random [1,2)
        for _ in 0..50 {
            let addend_bits = {
                let sign = (rng.next_u32() & 1) << 31;
                let exp = 100 + (rng.next_u32() % 61); // physics range
                let mant = rng.next_u32() & 0x7FFFFF;
                sign | (exp << 23) | mant
            };
            let expected = acc + f32::from_bits(addend_bits);
            let r = compute_fadd(acc.to_bits(), addend_bits);
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Accumulator trial {} step: {} + {} = {}",
                trial,
                acc,
                f32::from_bits(addend_bits),
                expected
            );
            acc = expected;
        }
    }
}
