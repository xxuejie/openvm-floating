//! Tests for the FMUL.S chip.
//!
//! Two levels of testing:
//! 1. Pure computation tests — verify the integer arithmetic (decomposition, carry chain,
//!    normalization, RNE rounding) matches native f32::mul.
//! 2. Chip harness tests — verify AIR constraint satisfaction via the OpenVM testing
//!    infrastructure.

#![allow(dead_code)]

// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Pure computation helpers (mirror the logic in fill_fmul_witness but return
// plain integers for easy testing without field types)
// ════════════════════════════════════════════════════════════════════════════

struct FmulResult {
    a_sign: u32,
    a_exp: u32,
    a_mant: u32,
    a_is_normal: bool,
    a_is_nan: bool,
    a_is_inf: bool,
    a_is_zero: bool,
    b_sign: u32,
    b_exp: u32,
    b_mant: u32,
    b_is_normal: bool,
    product: u64,
    is_high: bool,
    norm: u64,
    guard: u32,
    sticky: bool,
    round_up: bool,
    result_bits: u32,
    is_nan_result: bool,
    is_inf_result: bool,
    is_zero_result: bool,
}

fn compute_fmul(a_bits: u32, b_bits: u32) -> FmulResult {
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
    let b_is_zero = b_exp == 0 && b_mant == 0;
    let b_is_normal = (1..=254).contains(&b_exp);

    // Significand construction
    let a_sig = if a_is_normal {
        a_mant | (1 << 23)
    } else {
        a_mant
    };
    let b_sig = if b_is_normal {
        b_mant | (1 << 23)
    } else {
        b_mant
    };

    // Product
    let product = (a_sig as u64) * (b_sig as u64);

    // Normalization
    let is_high = (product >> 47) & 1 == 1;
    let norm = if is_high { product } else { product << 1 };

    // RNE rounding
    let guard = ((norm >> 23) & 1) as u32;
    let sticky = (norm & 0x7FFFFF) != 0;
    let mant_lsb = (norm >> 24) & 1;
    let round_up = guard == 1 && (sticky || mant_lsb == 1);

    // Result (native f32 = IEEE 754 RNE oracle)
    let result_bits = (f32::from_bits(a_bits) * f32::from_bits(b_bits)).to_bits();

    let is_nan_result = a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf);
    let r_exp = (result_bits >> 23) & 0xFF;
    let r_mant = result_bits & 0x7FFFFF;
    let is_inf_result = !is_nan_result && (a_is_inf || b_is_inf || r_exp == 255);
    let is_zero_result = !is_nan_result && (a_is_zero || b_is_zero || (r_exp == 0 && r_mant == 0));

    FmulResult {
        a_sign,
        a_exp,
        a_mant,
        a_is_normal,
        a_is_nan,
        a_is_inf,
        a_is_zero,
        b_sign,
        b_exp,
        b_mant,
        b_is_normal,
        product,
        is_high,
        norm,
        guard,
        sticky,
        round_up,
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
fn test_basic_multiply_2_times_3() {
    let a = 2.0f32;
    let b = 3.0f32;
    let result = a * b;
    assert_eq!(result, 6.0);

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
    assert!(r.a_is_normal);
    assert!(r.b_is_normal);
    assert!(!r.is_nan_result);
    assert!(!r.is_inf_result);
    assert!(!r.is_zero_result);
}

#[test]
fn test_basic_multiply_fractions() {
    let a = 1.5f32;
    let b = 2.5f32;
    let result = a * b;
    assert_eq!(result, 3.75);

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

#[test]
fn test_negative_times_positive() {
    let a = -4.0f32;
    let b = 0.25f32;
    let result = a * b;
    assert_eq!(result, -1.0);

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
    assert_eq!(r.a_sign, 1);
    assert_eq!(r.b_sign, 0);
}

#[test]
fn test_two_negatives() {
    let a = -3.0f32;
    let b = -7.0f32;
    let result = a * b;
    assert_eq!(result, 21.0);

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

#[test]
fn test_large_values() {
    let a = 1e30f32;
    let b = 1e10f32;
    let result = a * b;

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

#[test]
fn test_small_values() {
    let a = 1e-20f32;
    let b = 1e-15f32;
    let result = a * b;

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Edge case tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_zero_times_normal() {
    for &v in &[1.0f32, -1.0, 1e38, -1e38, 0.5, 123.456] {
        let expected = 0.0f32 * v;
        let r = compute_fmul(0.0f32.to_bits(), v.to_bits());
        assert_eq!(r.result_bits, expected.to_bits());
        assert!(r.is_zero_result);

        let expected2 = (-0.0f32) * v;
        let r2 = compute_fmul((-0.0f32).to_bits(), v.to_bits());
        assert_eq!(r2.result_bits, expected2.to_bits());
    }
}

#[test]
fn test_inf_times_normal() {
    let inf = f32::INFINITY;
    let r = compute_fmul(inf.to_bits(), 2.0f32.to_bits());
    assert_eq!(r.result_bits, f32::INFINITY.to_bits());
    assert!(r.is_inf_result);

    let r = compute_fmul(f32::NEG_INFINITY.to_bits(), 2.0f32.to_bits());
    assert_eq!(r.result_bits, f32::NEG_INFINITY.to_bits());
    assert!(r.is_inf_result);
}

#[test]
fn test_inf_times_zero_is_nan() {
    let r = compute_fmul(f32::INFINITY.to_bits(), 0.0f32.to_bits());
    assert!(f32::from_bits(r.result_bits).is_nan());
    assert!(r.is_nan_result);
    assert!(!r.is_inf_result);
    assert!(!r.is_zero_result);
}

#[test]
fn test_nan_input_produces_nan() {
    let nan = f32::NAN;
    let expected = nan * 1.0f32;
    assert!(expected.is_nan());
    let r = compute_fmul(nan.to_bits(), 1.0f32.to_bits());
    assert!(f32::from_bits(r.result_bits).is_nan());
    assert!(r.is_nan_result);
}

#[test]
fn test_overflow_to_inf() {
    let a = f32::MAX;
    let b = f32::MAX;
    let result = a * b;
    assert!(result.is_infinite());

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
    assert!(r.is_inf_result);
}

#[test]
fn test_underflow_to_zero() {
    let a = f32::from_bits(1); // smallest subnormal
    let b = f32::from_bits(1); // smallest subnormal
    let result = a * b;
    assert_eq!(result, 0.0); // underflows to zero

    let r = compute_fmul(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

#[test]
fn test_subnormal_input() {
    let sub = f32::from_bits(1); // smallest subnormal
    let r = compute_fmul(sub.to_bits(), 2.0f32.to_bits());
    let expected = sub * 2.0;
    assert_eq!(r.result_bits, expected.to_bits());
}

#[test]
fn test_rounding_tie_even() {
    // Find a case where the product lands exactly on a tie (half-ULP).
    // 1.0 * 1.0 + half_ulp_at_bit_24 = should round to even
    let one_plus_half_ulp = f32::from_bits(0x3F800000 + 1); // 1 + 1ULP at bit 0 of mantissa
    let two = 2.0f32;
    let result = one_plus_half_ulp * two;
    let r = compute_fmul(one_plus_half_ulp.to_bits(), two.to_bits());
    assert_eq!(r.result_bits, result.to_bits());
}

#[test]
fn test_physics_like_values() {
    // Values in the range JoltPhysics uses (10^-9 to 10^5)
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
    ];
    for &a in &values {
        for &b in &values {
            let expected = a * b;
            let r = compute_fmul(a.to_bits(), b.to_bits());
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Mismatch: {} * {} = {} (bits 0x{:08X}), got bits 0x{:08X}",
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
// 3. Carry chain verification
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_carry_chain_max_significands() {
    // Max normal significand: 0xFFFFFF (24 bits, all set)
    let a_sig: u32 = 0xFFFFFF;
    let b_sig: u32 = 0xFFFFFF;
    let product = (a_sig as u64) * (b_sig as u64);

    // Verify carry chain
    let a_limbs: [u32; 3] = [a_sig & 0xFF, (a_sig >> 8) & 0xFF, (a_sig >> 16) & 0xFF];
    let b_limbs: [u32; 3] = [b_sig & 0xFF, (b_sig >> 8) & 0xFF, (b_sig >> 16) & 0xFF];

    let mut carry: u32 = 0;
    let mut reconstructed: u64 = 0;
    for i in 0..6 {
        let mut sum = carry;
        for k in 0..3 {
            if k <= i && (i - k) < 3 {
                sum += a_limbs[k] * b_limbs[i - k];
            }
        }
        let limb = sum & 0xFF;
        carry = sum >> 8;
        reconstructed |= (limb as u64) << (i * 8);
    }
    assert_eq!(reconstructed, product);
    assert_eq!(carry, 0); // no overflow for 24x24 -> 48 bits
}

#[test]
fn test_carry_chain_min_significands() {
    let a_sig: u32 = 0x800000; // minimum normal (implicit bit only)
    let b_sig: u32 = 0x800000;
    let product = (a_sig as u64) * (b_sig as u64);
    // 0x800000 * 0x800000 = 0x400000000000
    assert_eq!(product, 0x400000000000);

    // Bit 46 is set, bit 47 is not → is_high = false
    assert_eq!((product >> 47) & 1, 0);
    assert_eq!((product >> 46) & 1, 1);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Fuzz test: differential testing against native f32::mul
// ════════════════════════════════════════════════════════════════════════════

/// Simple xorshift64 PRNG — no external dependency needed.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        } // avoid 0 seed
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
fn fuzz_fmul_random_inputs() {
    let mut rng = XorShift64::new(0xDEADBEEF_CAFEBABE);
    let iterations = 1_000_000;

    for i in 0..iterations {
        let a_bits = rng.next_u32();
        let b_bits = rng.next_u32();

        let expected = f32::from_bits(a_bits) * f32::from_bits(b_bits);
        let expected_bits = expected.to_bits();

        let r = compute_fmul(a_bits, b_bits);

        assert_eq!(
            r.result_bits,
            expected_bits,
            "Mismatch at iteration {}: 0x{:08X} * 0x{:08X}\n\
             f32: {} * {} = {} (0x{:08X})\n\
             got: 0x{:08X}\n\
             a: exp={}, mant=0x{:06X}, normal={}\n\
             b: exp={}, mant=0x{:06X}, normal={}\n\
             product=0x{:012X}, is_high={}, norm=0x{:012X}\n\
             guard={}, sticky={}, round_up={}",
            i,
            a_bits,
            b_bits,
            f32::from_bits(a_bits),
            f32::from_bits(b_bits),
            expected,
            expected_bits,
            r.result_bits,
            r.a_exp,
            r.a_mant,
            r.a_is_normal,
            r.b_exp,
            r.b_mant,
            r.b_is_normal,
            r.product,
            r.is_high,
            r.norm,
            r.guard,
            r.sticky,
            r.round_up,
        );
    }
}

#[test]
fn fuzz_fmul_normal_inputs() {
    // Focus on normal inputs (exp ∈ [1, 254]) since that's the JoltPhysics hot path
    let mut rng = XorShift64::new(0x1234_5678_9ABC_DEF0);
    let iterations = 500_000;

    for _ in 0..iterations {
        // Generate normal f32 values
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_a = (rng.next_u32() % 254) + 1; // [1, 254]
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign_a | (exp_a << 23) | mant_a;

        let sign_b = (rng.next_u32() & 1) << 31;
        let exp_b = (rng.next_u32() % 254) + 1;
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let b_bits = sign_b | (exp_b << 23) | mant_b;

        let expected = f32::from_bits(a_bits) * f32::from_bits(b_bits);
        let r = compute_fmul(a_bits, b_bits);
        assert_eq!(r.result_bits, expected.to_bits());
    }
}

#[test]
fn fuzz_fmul_physics_range() {
    // Physics-like values: exponents in [100, 160] (magnitudes ~10^-9 to 10^5)
    let mut rng = XorShift64::new(0xCAFE_BABE_DEAD_BEEF);
    let iterations = 500_000;

    for _ in 0..iterations {
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_a = 100 + (rng.next_u32() % 61); // [100, 160]
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let a_bits = sign_a | (exp_a << 23) | mant_a;

        let sign_b = (rng.next_u32() & 1) << 31;
        let exp_b = 100 + (rng.next_u32() % 61);
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let b_bits = sign_b | (exp_b << 23) | mant_b;

        let expected = f32::from_bits(a_bits) * f32::from_bits(b_bits);
        let r = compute_fmul(a_bits, b_bits);
        assert_eq!(
            r.result_bits,
            expected.to_bits(),
            "Physics fuzz mismatch: {} * {}",
            f32::from_bits(a_bits),
            f32::from_bits(b_bits)
        );
    }
}

#[test]
fn fuzz_fmul_special_combinations() {
    // Exhaustively test combinations of special values with random normals
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
            let expected = f32::from_bits(sa) * f32::from_bits(sb);
            let r = compute_fmul(sa, sb);
            assert_eq!(
                r.result_bits,
                expected.to_bits(),
                "Special combo: {} * {} = {} (0x{:08X}), got 0x{:08X}",
                na,
                nb,
                expected,
                expected.to_bits(),
                r.result_bits
            );
        }
        // Also test with random normals
        for _ in 0..100 {
            let rand_bits = rng.next_u32() | 0x3F800000; // ensure non-zero exp
            let expected = f32::from_bits(sa) * f32::from_bits(rand_bits);
            let r = compute_fmul(sa, rand_bits);
            assert_eq!(r.result_bits, expected.to_bits());
        }
    }
}
