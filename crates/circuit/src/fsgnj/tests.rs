//! Tests for the fsgnj.s chip (sign injection family).

#![allow(dead_code)]

struct FsgnjResult {
    result_bits: u32,
}

fn compute_fsgnj(a_bits: u32, b_bits: u32, op: u8) -> FsgnjResult {
    let result = match op {
        0 => (a_bits & 0x7FFFFFFF) | (b_bits & 0x80000000), // fsgnj
        1 => (a_bits & 0x7FFFFFFF) | (!b_bits & 0x80000000), // fsgnjn
        2 => a_bits ^ (b_bits & 0x80000000),                // fsgnjx
        _ => unreachable!(),
    };
    FsgnjResult {
        result_bits: result,
    }
}

#[test]
fn test_fsgnj_copy_sign() {
    let pos = 1.0f32.to_bits();
    let neg = (-1.0f32).to_bits();
    let r = compute_fsgnj(pos, neg, 0);
    assert_eq!(r.result_bits, neg); // sign from b
}

#[test]
fn test_fsgnjn_negate_sign() {
    let pos = 1.0f32.to_bits();
    let neg = (-1.0f32).to_bits();
    let r = compute_fsgnj(pos, neg, 1);
    assert_eq!(r.result_bits, pos); // negated sign from b
}

#[test]
fn test_fsgnjx_xor_sign() {
    let pos = 1.0f32.to_bits();
    let neg = (-1.0f32).to_bits();
    let r = compute_fsgnj(pos, neg, 2);
    assert_eq!(r.result_bits, neg); // XOR: pos XOR neg = neg
    let r2 = compute_fsgnj(neg, neg, 2);
    assert_eq!(r2.result_bits, pos); // XOR: neg XOR neg = pos (absolute value)
}

#[test]
fn test_fsgnj_preserves_magnitude() {
    let a = 3.5f32.to_bits();
    let b = (-2.71f32).to_bits();
    let r = compute_fsgnj(a, b, 0);
    assert_eq!(r.result_bits & 0x7FFFFFFF, a & 0x7FFFFFFF);
    assert_eq!(r.result_bits & 0x80000000, b & 0x80000000);
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
fn fuzz_fsgnj_all_ops() {
    let mut rng = XorShift64::new(0xCAFE_2025_0007);
    for _ in 0..500_000 {
        let a = rng.next_u32();
        let b = rng.next_u32();
        let (a_sign, b_sign) = (a >> 31, b >> 31);
        for op in 0..3u8 {
            let r = compute_fsgnj(a, b, op);
            // Magnitude always comes from a.
            assert_eq!(r.result_bits & 0x7FFF_FFFF, a & 0x7FFF_FFFF);
            // Sign is determined by b (and a, for fsgnjx).
            let expected_sign = match op {
                0 => b_sign,
                1 => b_sign ^ 1,
                2 => a_sign ^ b_sign,
                _ => unreachable!(),
            };
            assert_eq!(
                r.result_bits >> 31,
                expected_sign,
                "op {op} a={a:#010x} b={b:#010x}"
            );
        }
    }
}
