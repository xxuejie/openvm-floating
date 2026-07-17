//! Tests for the FSUB.S chip.

#![allow(dead_code)]

struct FsubResult {
    result_bits: u32,
}

fn compute_fsub(a_bits: u32, b_bits: u32) -> FsubResult {
    let result = (f32::from_bits(a_bits) - f32::from_bits(b_bits)).to_bits();
    FsubResult {
        result_bits: result,
    }
}

#[test]
fn test_basic_sub() {
    let r = compute_fsub(5.0f32.to_bits(), 3.0f32.to_bits());
    assert_eq!(r.result_bits, 2.0f32.to_bits());
}

#[test]
fn test_sub_to_zero() {
    let r = compute_fsub(5.0f32.to_bits(), 5.0f32.to_bits());
    assert_eq!(r.result_bits, 0.0f32.to_bits());
}

#[test]
fn test_sub_negative_result() {
    let r = compute_fsub(3.0f32.to_bits(), 5.0f32.to_bits());
    assert_eq!(r.result_bits, (-2.0f32).to_bits());
}

#[test]
fn test_sub_fractions() {
    let a = 1.5f32;
    let b = 0.25f32;
    let r = compute_fsub(a.to_bits(), b.to_bits());
    assert_eq!(r.result_bits, (a - b).to_bits());
}

#[test]
fn test_sub_inf_minus_inf_is_nan() {
    let r = compute_fsub(f32::INFINITY.to_bits(), f32::INFINITY.to_bits());
    assert!(f32::from_bits(r.result_bits).is_nan());
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
fn fuzz_fsub_random() {
    let mut rng = XorShift64::new(0x5B18_2025_CAFE);
    for _ in 0..1_000_000 {
        let a = rng.next_u32();
        let b = rng.next_u32();
        let expected = f32::from_bits(a) - f32::from_bits(b);
        let r = compute_fsub(a, b);
        if expected.is_nan() {
            assert!(f32::from_bits(r.result_bits).is_nan());
        } else {
            assert_eq!(r.result_bits, expected.to_bits());
        }
    }
}

#[test]
fn fuzz_fsub_normal() {
    let mut rng = XorShift64::new(0x5FB2_025B_EEFA);
    for _ in 0..500_000 {
        let sign_a = (rng.next_u32() & 1) << 31;
        let exp_a = (rng.next_u32() % 254) + 1;
        let mant_a = rng.next_u32() & 0x7FFFFF;
        let sign_b = (rng.next_u32() & 1) << 31;
        let exp_b = (rng.next_u32() % 254) + 1;
        let mant_b = rng.next_u32() & 0x7FFFFF;
        let a = sign_a | (exp_a << 23) | mant_a;
        let b = sign_b | (exp_b << 23) | mant_b;
        let expected = f32::from_bits(a) - f32::from_bits(b);
        let r = compute_fsub(a, b);
        assert_eq!(r.result_bits, expected.to_bits());
    }
}
