//! Variable-width logical right shift of a byte-decomposed value, with RNE
//! rounding metadata (guard / sticky) for floating-point use.
//!
//! Mirrors the approach used by the rv32im `SRL` instruction: decompose the
//! shift `sh = 8*q + r` into a byte-granular shift `q` (one-hot) and a sub-byte
//! shift `r` (one-hot). The low `r` bits of each byte are provided as a
//! range-checked witness `bit_shift_carry[i] = x[i] mod 2^r`, which lets the
//! shifted byte be reconstructed *without field division* by multiplying both
//! sides of the byte relation by `bit_mult = 2^r`.

use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_air::AirBuilder, p3_field::PrimeCharacteristicRing,
};

/// Bits per limb (bytes).
pub const SHLIMB_BITS: usize = 8;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct ShrnCols<T, const N: usize> {
    // === IO (provided by the embedding chip) ===
    pub x: [T; N], // value to shift, little-endian bytes
    pub shift: T,  // shift amount in [0, 8 * N - 1]
    pub enable: T, // gating flag

    // === One-hot selectors ===
    pub limb_shift_marker: [T; N], // q = shift / 8, in [0, N - 1]
    pub bit_shift_marker: [T; SHLIMB_BITS], // r = shift % 8, in [0, 7]
    pub bit_mult: T,               // 2^r

    // === Per-byte low-part carries: bit_shift_carry[i] = x[i] mod 2^r ===
    pub bit_shift_carry: [T; N],

    // === Output ===
    pub y: [T; N], // shifted value = x >> shift, little-endian bytes
}

/// Variable right-shift gadget (shift value; guard/sticky to be added).
#[derive(Copy, Clone, Debug)]
pub struct ShrnGadget {
    pub range_bus: VariableRangeCheckerBus,
}

impl ShrnGadget {
    pub fn new(range_bus: VariableRangeCheckerBus) -> Self {
        Self { range_bus }
    }

    pub fn eval_shrn<AB, const N: usize>(&self, builder: &mut AB, cols: &ShrnCols<AB::Var, N>)
    where
        AB: InteractionBuilder,
    {
        let f8: AB::F = AB::F::from_u32(SHLIMB_BITS as u32);
        let f256: AB::F = AB::F::from_u32(1 << SHLIMB_BITS);

        // ── One-hot selectors and bit_mult ───────────────────────────────
        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for i in 0..N {
            builder.assert_bool(cols.limb_shift_marker[i]);
            limb_marker_sum += cols.limb_shift_marker[i].into();
            limb_shift += AB::Expr::from_usize(i) * cols.limb_shift_marker[i];
        }
        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        for j in 0..SHLIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[j]);
            bit_marker_sum += cols.bit_shift_marker[j].into();
            bit_shift += AB::Expr::from_usize(j) * cols.bit_shift_marker[j];
            builder
                .when(cols.bit_shift_marker[j])
                .assert_eq(cols.bit_mult, AB::F::from_usize(1 << j));
        }
        // shift = 8 * limb_shift + bit_shift
        builder.when(cols.enable).assert_eq(
            cols.shift,
            limb_shift.clone() * AB::Expr::from(f8) + bit_shift.clone(),
        );
        builder.when(cols.enable).assert_one(limb_marker_sum);
        builder.when(cols.enable).assert_one(bit_marker_sum);

        // ── Range-check each bit_shift_carry[i] to a dynamic `bit_shift` bits ─
        for i in 0..N {
            self.range_bus
                .send(cols.bit_shift_carry[i].into(), bit_shift.clone())
                .eval(builder, cols.enable);
        }

        // ── Shifted byte reconstruction ──────────────────────────────────
        // For limb_shift = q, bit_shift = r:
        //   y[k] * 2^r = bit_shift_carry[k+q+1] * 256 + (x[k+q] - bit_shift_carry[k+q])
        // with the next-byte carry = 0 when k+q+1 >= N (logical fill 0), and
        // y[k] = 0 when k+q >= N (shifted out entirely).
        let f256e = AB::Expr::from(f256);
        for k in 0..N {
            for q in 0..N {
                if k + q >= N {
                    builder
                        .when(cols.enable)
                        .when(cols.limb_shift_marker[q])
                        .assert_zero(cols.y[k]);
                    continue;
                }
                let cur = cols.x[k + q] - cols.bit_shift_carry[k + q];
                let next_carry = if k + q + 1 < N {
                    cols.bit_shift_carry[k + q + 1].into()
                } else {
                    AB::Expr::ZERO
                };
                builder
                    .when(cols.enable)
                    .when(cols.limb_shift_marker[q])
                    .assert_eq(cols.y[k] * cols.bit_mult, next_carry * f256e.clone() + cur);
            }
        }

        // Range-check output bytes to 8 bits.
        for k in 0..N {
            self.range_bus
                .range_check(cols.y[k].into(), SHLIMB_BITS)
                .eval(builder, cols.enable);
        }
    }
}

// === Host-side computation (for witnesses + tests) ===

/// Compute (y = x >> shift, limb_shift, bit_shift) for little-endian byte array `x`.
pub fn run_shrn<const N: usize>(x: &[u32; N], shift: usize) -> ([u32; N], usize, usize) {
    let shift = shift.min(N * SHLIMB_BITS);
    let limb_shift = shift / SHLIMB_BITS;
    let bit_shift = shift % SHLIMB_BITS;
    let val: u128 = (0..N).fold(0u128, |acc, i| acc | ((x[i] as u128) << (i * SHLIMB_BITS)));
    let shifted = val >> shift;
    let y: [u32; N] = core::array::from_fn(|i| ((shifted >> (i * SHLIMB_BITS)) & 0xFF) as u32);
    (y, limb_shift, bit_shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_stark_backend::p3_field::PrimeField32;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    type F = BabyBear;
    const N: usize = 6;

    fn make_cols(x: [u32; N], shift: usize) -> ShrnCols<F, N> {
        let shift = shift.min(N * SHLIMB_BITS - 1);
        let (y, limb_shift, bit_shift) = run_shrn(&x, shift);
        let mut limb_shift_marker = [F::ZERO; N];
        let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
        limb_shift_marker[limb_shift] = F::ONE;
        bit_shift_marker[bit_shift] = F::ONE;
        let bit_shift_carry =
            core::array::from_fn(|i| F::from_u32(x[i] & ((1u32 << bit_shift) - 1)));
        ShrnCols {
            x: core::array::from_fn(|i| F::from_u32(x[i])),
            shift: F::from_u32(shift as u32),
            enable: F::ONE,
            limb_shift_marker,
            bit_shift_marker,
            bit_mult: F::from_u32(1u32 << bit_shift),
            bit_shift_carry,
            y: core::array::from_fn(|i| F::from_u32(y[i])),
        }
    }

    fn run(c: &ShrnCols<F, N>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        ShrnGadget::new(VariableRangeCheckerBus::new(0, 16)).eval_shrn(&mut b, c);
        b
    }

    #[test]
    fn shrn_honest_witness_passes_all_shifts() {
        let mut rng = 0x5A5A_5A5A_1234_5678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..4000 {
            // random x
            let x: [u32; N] = core::array::from_fn(|_| next() & 0xFF);
            let shift = (next() as usize) % (N * SHLIMB_BITS);
            let c = make_cols(x, shift);
            let b = run(&c);
            assert!(b.ok(), "shift {} of {:?}: {:?}", shift, x, b.failures);

            // spot-check against host computation
            let (y_exp, _, _) = run_shrn(&x, shift);
            for (i, &ye) in y_exp.iter().enumerate() {
                assert_eq!(c.y[i].as_canonical_u32(), ye);
            }
        }
    }

    #[test]
    fn shrn_zero_and_max_value() {
        for &shift in &[0usize, 1, 7, 8, 9, 15, 16, 23, 24, 31, 40, 47] {
            let zero = [0u32; N];
            let maxv = [0xFFu32; N];
            let b = run(&make_cols(zero, shift));
            assert!(b.ok(), "zero shift {}: {:?}", shift, b.failures);
            let b = run(&make_cols(maxv, shift));
            assert!(b.ok(), "max shift {}: {:?}", shift, b.failures);
        }
    }

    #[test]
    fn shrn_corrupt_output_rejected() {
        let mut c = make_cols([0xFF, 0x80, 0x01, 0x00, 0xAB, 0xCD], 13);
        c.y[0] = F::from_u32(c.y[0].as_canonical_u32() ^ 1);
        let b = run(&c);
        assert!(!b.ok(), "corrupted output must be rejected");
    }

    #[test]
    fn shrn_disabled_does_not_constrain() {
        // enable = 0: no constraint fires, so any y passes.
        let mut c = make_cols([1, 2, 3, 4, 5, 6], 5);
        c.enable = F::ZERO;
        c.y[0] = F::from_u32(222);
        let b = run(&c);
        assert!(b.ok(), "disabled gadget must not constrain");
    }
}
