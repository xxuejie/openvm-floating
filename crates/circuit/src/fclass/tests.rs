//! Tests for the fclass.s chip.

#![allow(dead_code)]

use super::*;

const POS_ZERO: u32 = 0x0000_0000;
const NEG_ZERO: u32 = 0x8000_0000;

#[test]
fn test_fclass_all_categories() {
    assert_eq!(fclass_s(f32::NEG_INFINITY.to_bits()), FCLASS_NEG_INF);
    assert_eq!(fclass_s((-1.5f32).to_bits()), FCLASS_NEG_NORM);
    assert_eq!(fclass_s(NEG_ZERO | 1), FCLASS_NEG_SUBNORM);
    assert_eq!(fclass_s(NEG_ZERO), FCLASS_NEG_ZERO);
    assert_eq!(fclass_s(POS_ZERO), FCLASS_POS_ZERO);
    assert_eq!(fclass_s(1), FCLASS_POS_SUBNORM);
    assert_eq!(fclass_s(1.5f32.to_bits()), FCLASS_POS_NORM);
    assert_eq!(fclass_s(f32::INFINITY.to_bits()), FCLASS_POS_INF);
    assert_eq!(fclass_s(0x7FA0_0001), FCLASS_SNAN);
    assert_eq!(fclass_s(0x7FC0_0000), FCLASS_QNAN);
    // negative NaN payload / negative quiet NaN
    assert_eq!(fclass_s(0xFFB0_0000), FCLASS_SNAN);
    assert_eq!(fclass_s(0xFFC0_0000), FCLASS_QNAN);
}

#[test]
fn test_fclass_boundaries() {
    // largest subnormal / smallest normal
    assert_eq!(fclass_s(0x007F_FFFF), FCLASS_POS_SUBNORM);
    assert_eq!(fclass_s(0x0080_0000), FCLASS_POS_NORM);
    // largest finite
    assert_eq!(fclass_s(0x7F7F_FFFF), FCLASS_POS_NORM);
    assert_eq!(fclass_s(0xFF7F_FFFF), FCLASS_NEG_NORM);
    // smallest sNaN has only the LSB set (quiet bit clear)
    assert_eq!(fclass_s(0x7F80_0001), FCLASS_SNAN);
    // mantissa with quiet bit and payload bits set is still qNaN
    assert_eq!(fclass_s(0x7FFF_FFFF), FCLASS_QNAN);
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
fn fuzz_fclass() {
    let mut rng = XorShift64::new(0xC1A5_2025_000A);
    for _ in 0..500_000 {
        let bits = rng.next_u32();
        let mask = fclass_s(bits);
        // exactly one bit set
        assert_eq!(mask.count_ones(), 1, "fclass({bits:#010x}) = {mask:#x}");
        // consistent with a direct re-derivation
        let sign = (bits >> 31) & 1;
        let exp = (bits >> 23) & 0xFF;
        let mant = bits & 0x7FFFFF;
        let expected = match (exp, mant, sign) {
            (0xFF, 0, 1) => FCLASS_NEG_INF,
            (0xFF, 0, _) => FCLASS_POS_INF,
            (0xFF, m, _) if m & 0x400000 != 0 => FCLASS_QNAN,
            (0xFF, _, _) => FCLASS_SNAN,
            (0, 0, 1) => FCLASS_NEG_ZERO,
            (0, 0, _) => FCLASS_POS_ZERO,
            (0, _, 1) => FCLASS_NEG_SUBNORM,
            (0, _, _) => FCLASS_POS_SUBNORM,
            (_, _, 1) => FCLASS_NEG_NORM,
            _ => FCLASS_POS_NORM,
        };
        assert_eq!(mask, expected, "fclass({bits:#010x})");
    }
}
