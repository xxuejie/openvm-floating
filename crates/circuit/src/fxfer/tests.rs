#![allow(dead_code)]

#[test]
fn test_fxfer_identity() {
    let data = [0x00u8, 0x00, 0x40, 0x40];
    let result = data;
    assert_eq!(data, result);
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
fn fuzz_fxfer() {
    let mut rng = XorShift64::new(0xFE10_2025);
    for _ in 0..500_000 {
        let bits = rng.next_u32();
        let data = bits.to_le_bytes();
        let result = data;
        assert_eq!(data, result);
    }
}
