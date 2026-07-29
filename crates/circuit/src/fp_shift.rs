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

impl<T: Default, const N: usize> Default for ShrnCols<T, N> {
    fn default() -> Self {
        Self {
            x: core::array::from_fn(|_| T::default()),
            shift: T::default(),
            enable: T::default(),
            limb_shift_marker: core::array::from_fn(|_| T::default()),
            bit_shift_marker: core::array::from_fn(|_| T::default()),
            bit_mult: T::default(),
            bit_shift_carry: core::array::from_fn(|_| T::default()),
            y: core::array::from_fn(|_| T::default()),
            guard: T::default(),
            sticky: T::default(),
            pmsb: T::default(),
            prest: T::default(),
            hmsb: T::default(),
            hrest: T::default(),
            lost_combined: T::default(),
            combined_inv: T::default(),
        }
    }
}

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

    // === Rounding metadata (guard / sticky) ===
    pub guard: T,         // bit (shift-1) of x (0 when shift == 0)
    pub sticky: T,        // 1 iff any bit of x strictly below `guard` is set
    pub pmsb: T,          // MSB of partial_carry = guard when r >= 1
    pub prest: T,         // partial_carry minus its MSB (bits below guard in boundary)
    pub hmsb: T,          // bit 7 of x[q-1] = guard when r == 0
    pub hrest: T,         // x[q-1] minus its top bit
    pub lost_combined: T, // sum of all bits strictly below the guard (nonzero-ness = sticky)
    pub combined_inv: T,  // inverse for the sticky nonzero check
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

        self.eval_round(builder, cols);
    }

    /// Constrain the RNE rounding metadata (`guard`, `sticky`) for the shift.
    ///
    /// Lost bits = `x mod 2^shift`. `guard` = bit `(shift-1)`, `sticky` = any bit
    /// below it. With `shift = 8q + r`:
    /// - `r >= 1`: guard is bit `(r-1)` of `x[q]` = MSB of `bit_shift_carry[q]`;
    ///   the fully-lost bytes `x[0..q-1]` and the low `(r-1)` bits of `x[q]` feed sticky.
    /// - `r == 0`: guard is bit 7 of `x[q-1]`; sticky is the rest of `x[0..q-1]`.
    pub fn eval_round<AB, const N: usize>(&self, builder: &mut AB, cols: &ShrnCols<AB::Var, N>)
    where
        AB: InteractionBuilder,
    {
        let f128: AB::F = AB::F::from_u32(128);
        let one = AB::Expr::ONE;
        let enable = cols.enable;

        builder.assert_bool(cols.guard);
        builder.assert_bool(cols.sticky);
        builder.assert_bool(cols.pmsb);
        builder.assert_bool(cols.hmsb);

        // partial_carry = bit_shift_carry[q] = x[q] mod 2^r  (0 when r == 0).
        let partial_carry: AB::Expr = (0..N).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.limb_shift_marker[i] * cols.bit_shift_carry[i]
        });
        // highest_full = x[q-1]  (0 when q == 0).
        let highest_full: AB::Expr = (1..N).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.limb_shift_marker[i] * cols.x[i - 1]
        });

        // prefix[i] = x[0]+...+x[i-1]
        let mut prefix: Vec<AB::Expr> = Vec::with_capacity(N + 1);
        prefix.push(AB::Expr::ZERO);
        for i in 0..N {
            prefix.push(prefix[i].clone() + cols.x[i]);
        }
        // Sum of fully-lost bytes:
        //   r >= 1: x[0..q-1]  = prefix[q]
        //   r == 0: x[0..q-2]  = prefix[q-1]
        let full_rge1: AB::Expr = (0..N).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.limb_shift_marker[i] * prefix[i].clone()
        });
        let full_r0: AB::Expr = (1..N).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.limb_shift_marker[i] * prefix[i - 1].clone()
        });

        // Decompose partial_carry = prest + pmsb * 2^(r-1) for r >= 1; range-check prest.
        for j in 1..SHLIMB_BITS {
            let cnt = enable * cols.bit_shift_marker[j];
            builder
                .when(enable)
                .when(cols.bit_shift_marker[j])
                .assert_eq(
                    partial_carry.clone(),
                    cols.prest + cols.pmsb * AB::F::from_usize(1 << (j - 1)),
                );
            self.range_bus
                .send(cols.prest, AB::Expr::from_usize(j - 1))
                .eval(builder, cnt);
        }
        // Decompose highest_full = hrest + hmsb * 128 for r == 0; range-check hrest.
        builder
            .when(enable)
            .when(cols.bit_shift_marker[0])
            .assert_eq(highest_full.clone(), cols.hrest + cols.hmsb * f128);
        self.range_bus
            .range_check(cols.hrest, 7)
            .eval(builder, enable * cols.bit_shift_marker[0]);

        // guard = rsl[0]*hmsb + (1 - rsl[0])*pmsb
        let not_r0 = one.clone() - cols.bit_shift_marker[0];
        builder.when(enable).assert_eq(
            cols.guard,
            cols.bit_shift_marker[0] * cols.hmsb + not_r0.clone() * cols.pmsb,
        );

        // lost_combined = sum of all bits strictly below the guard.
        //   r >= 1: prest + full_rge1
        //   r == 0: hrest + full_r0
        builder.when(enable).assert_eq(
            cols.lost_combined,
            not_r0.clone() * (cols.prest + full_rge1)
                + cols.bit_shift_marker[0] * (cols.hrest + full_r0),
        );

        // sticky = (lost_combined != 0):  sticky = lost_combined * inv,
        // and lost_combined * sticky == lost_combined (forces sticky=1 when nonzero).
        builder
            .when(enable)
            .assert_eq(cols.sticky, cols.lost_combined * cols.combined_inv);
        builder
            .when(enable)
            .assert_zero(cols.lost_combined * cols.sticky - cols.lost_combined);
    }
}

// === Variable left-shift gadget (for FP cancellation normalization) ===========

/// Variable left shift of a byte-decomposed value. Mirrors `ShrnGadget` but shifts
/// left, and enforces that **no bits are lost** (the bits shifted above the top
/// byte must be zero) — required for normalization, where the shift amount is
/// chosen so the leading set bit lands exactly at the top of the field.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct ShlnCols<T, const N: usize> {
    // === IO ===
    pub x: [T; N], // value to shift, little-endian bytes
    pub shift: T,  // shift amount in [0, 8 * N - 1]
    pub enable: T,

    // === One-hot selectors ===
    pub limb_shift_marker: [T; N],          // q = shift / 8
    pub bit_shift_marker: [T; SHLIMB_BITS], // r = shift % 8
    pub bit_mult: T,                        // 2^r

    // === Per-byte high-part carries: bit_shift_carry[i] = x[i] >> (8 - r) ===
    pub bit_shift_carry: [T; N],

    // === Output ===
    pub y: [T; N], // x << shift, little-endian bytes
}

impl<T: Default, const N: usize> Default for ShlnCols<T, N> {
    fn default() -> Self {
        Self {
            x: core::array::from_fn(|_| T::default()),
            shift: T::default(),
            enable: T::default(),
            limb_shift_marker: core::array::from_fn(|_| T::default()),
            bit_shift_marker: core::array::from_fn(|_| T::default()),
            bit_mult: T::default(),
            bit_shift_carry: core::array::from_fn(|_| T::default()),
            y: core::array::from_fn(|_| T::default()),
        }
    }
}

/// Variable left-shift gadget (no precision loss).
#[derive(Copy, Clone, Debug)]
pub struct ShlnGadget {
    pub range_bus: VariableRangeCheckerBus,
}

impl ShlnGadget {
    pub fn new(range_bus: VariableRangeCheckerBus) -> Self {
        Self { range_bus }
    }

    pub fn eval_shln<AB, const N: usize>(&self, builder: &mut AB, cols: &ShlnCols<AB::Var, N>)
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
        builder.when(cols.enable).assert_eq(
            cols.shift,
            limb_shift.clone() * AB::Expr::from(f8) + bit_shift.clone(),
        );
        builder.when(cols.enable).assert_one(limb_marker_sum);
        builder.when(cols.enable).assert_one(bit_marker_sum);

        // ── Range-check each bit_shift_carry[i] to r bits ────────────────
        // bit_shift_carry[i] = x[i] >> (8 - r), the high r bits that carry up.
        for i in 0..N {
            self.range_bus
                .send(cols.bit_shift_carry[i].into(), bit_shift.clone())
                .eval(builder, cols.enable);
        }

        // ── Shifted byte reconstruction ──────────────────────────────────
        // For limb_shift = i, bit_shift = r:
        //   j < i:                 y[j] = 0
        //   j >= i:                y[j] = carry_in + x[j-i]*2^r - 256*bit_shift_carry[j-i]
        //     where carry_in = bit_shift_carry[j-i-1] (0 when j == i)
        let f256e = AB::Expr::from(f256);
        for i in 0..N {
            for j in 0..N {
                if j < i {
                    builder
                        .when(cols.enable)
                        .when(cols.limb_shift_marker[i])
                        .assert_zero(cols.y[j]);
                } else {
                    let s = j - i;
                    let carry_in = if s == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[s - 1].into()
                    };
                    builder
                        .when(cols.enable)
                        .when(cols.limb_shift_marker[i])
                        .assert_eq(
                            cols.y[j],
                            carry_in + cols.x[s] * cols.bit_mult
                                - f256e.clone() * cols.bit_shift_carry[s],
                        );
                }
            }
        }

        // Range-check output bytes to 8 bits.
        for k in 0..N {
            self.range_bus
                .range_check(cols.y[k].into(), SHLIMB_BITS)
                .eval(builder, cols.enable);
        }

        // ── No-overflow: bits shifted above byte N-1 must be zero ─────────
        // Source byte x[k] lands at byte k + limb_shift; if that is >= N it must be 0.
        for k in 0..N {
            let overflow_byte = (0..N).fold(AB::Expr::ZERO, |acc, i| {
                if k + i >= N {
                    acc + cols.limb_shift_marker[i]
                } else {
                    acc
                }
            });
            builder
                .when(cols.enable)
                .assert_zero(cols.x[k] * overflow_byte);
        }
        // Sub-byte overflow: the carry out of the top in-range source byte
        // (bit_shift_carry[N-1-i] for active limb shift i) must be 0.
        let top_carry = (0..N).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.limb_shift_marker[i] * cols.bit_shift_carry[N - 1 - i]
        });
        builder.when(cols.enable).assert_zero(top_carry);
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

/// Host-side guard/sticky for rounding `x >> shift`: guard = bit (shift-1),
/// sticky = any bit strictly below the guard is set.
pub fn run_shrn_round<const N: usize>(x: &[u32; N], shift: usize) -> (u32, bool) {
    let shift = shift.min(N * SHLIMB_BITS - 1);
    let val: u128 = (0..N).fold(0u128, |acc, i| acc | ((x[i] as u128) << (i * SHLIMB_BITS)));
    let guard = if shift >= 1 {
        ((val >> (shift - 1)) & 1) as u32
    } else {
        0
    };
    let sticky = if shift >= 1 {
        (val & ((1u128 << (shift - 1)) - 1)) != 0
    } else {
        false
    };
    (guard, sticky)
}

/// Compute (y = x << shift, limb_shift, bit_shift) for little-endian byte array `x`.
/// Bits shifted above the field are discarded (caller must ensure they are zero for
/// normalization). Returns the in-field shifted bytes.
pub fn run_shln<const N: usize>(x: &[u32; N], shift: usize) -> ([u32; N], usize, usize) {
    let shift = shift.min(N * SHLIMB_BITS);
    let limb_shift = shift / SHLIMB_BITS;
    let bit_shift = shift % SHLIMB_BITS;
    let val: u128 = (0..N).fold(0u128, |acc, i| acc | ((x[i] as u128) << (i * SHLIMB_BITS)));
    let shifted = val << shift;
    let y: [u32; N] = core::array::from_fn(|i| ((shifted >> (i * SHLIMB_BITS)) & 0xFF) as u32);
    (y, limb_shift, bit_shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::NumericBuilder;
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    type F = BabyBear;
    const N: usize = 6;

    fn make_cols(x: [u32; N], shift: usize) -> ShrnCols<F, N> {
        let shift = shift.min(N * SHLIMB_BITS - 1);
        let (y, q, r) = run_shrn(&x, shift);
        let (guard, sticky) = run_shrn_round(&x, shift);
        let mut limb_shift_marker = [F::ZERO; N];
        let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
        limb_shift_marker[q] = F::ONE;
        bit_shift_marker[r] = F::ONE;

        // partial-carry decomposition (r >= 1)
        let pc = x[q] & ((1u32 << r) - 1);
        let pmsb = if r >= 1 { (pc >> (r - 1)) & 1 } else { 0 };
        let prest = if r >= 1 {
            pc - pmsb * (1u32 << (r - 1))
        } else {
            0
        };
        // highest fully-lost byte decomposition (r == 0)
        let hf = if q >= 1 { x[q - 1] } else { 0 };
        let hmsb = if r == 0 { (hf >> 7) & 1 } else { 0 };
        let hrest = if r == 0 { hf - hmsb * 128 } else { 0 };
        // lost-combined (sum of bits strictly below the guard)
        let full_rge1: u32 = (0..q).map(|k| x[k]).sum();
        let full_r0: u32 = (0..q.saturating_sub(1)).map(|k| x[k]).sum();
        let lost_combined = if r >= 1 {
            prest + full_rge1
        } else {
            hrest + full_r0
        };
        let combined_inv = if lost_combined != 0 {
            F::from_u32(lost_combined).try_inverse().unwrap_or(F::ZERO)
        } else {
            F::ZERO
        };

        ShrnCols {
            x: core::array::from_fn(|i| F::from_u32(x[i])),
            shift: F::from_u32(shift as u32),
            enable: F::ONE,
            limb_shift_marker,
            bit_shift_marker,
            bit_mult: F::from_u32(1u32 << r),
            bit_shift_carry: core::array::from_fn(|i| F::from_u32(x[i] & ((1u32 << r) - 1))),
            y: core::array::from_fn(|i| F::from_u32(y[i])),
            guard: F::from_u32(guard),
            sticky: F::from_bool(sticky),
            pmsb: F::from_u32(pmsb),
            prest: F::from_u32(prest),
            hmsb: F::from_u32(hmsb),
            hrest: F::from_u32(hrest),
            lost_combined: F::from_u32(lost_combined),
            combined_inv,
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

    #[test]
    fn shrn_guard_sticky_match_host() {
        let mut rng = 0x1234_9876_CAFE_BABEu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..8000 {
            let x: [u32; N] = core::array::from_fn(|_| next() & 0xFF);
            let shift = (next() as usize) % (N * SHLIMB_BITS);
            let c = make_cols(x, shift);
            // honest witness must pass all constraints (incl. guard/sticky)
            let b = run(&c);
            assert!(b.ok(), "shift {} of {:?}: {:?}", shift, x, b.failures);
            // and the witness guard/sticky must match the host computation
            let (g, s) = run_shrn_round(&x, shift);
            assert_eq!(
                c.guard.as_canonical_u32(),
                g,
                "guard shift {} x={:?}",
                shift,
                x
            );
            assert_eq!(
                c.sticky.as_canonical_u32(),
                s as u32,
                "sticky shift {} x={:?}",
                shift,
                x
            );
        }
    }

    #[test]
    fn shrn_guard_sticky_known_values() {
        // shift 1: guard = bit 0, sticky = 0
        let c = make_cols([0b0000_0001, 0, 0, 0, 0, 0], 1);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (1, 0)
        );
        // shift 3: guard = bit 2 (=1), sticky = bits[0,1] (=1,0) -> 1
        let c = make_cols([0b0000_0101, 0, 0, 0, 0, 0], 3);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (1, 1)
        );
        // shift 8: guard = bit 7 of byte0 (=1), sticky = bits[0..7] (=0) -> 0
        let c = make_cols([0x80, 0, 0, 0, 0, 0], 8);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (1, 0)
        );
        // shift 9: guard = bit 0 of byte1 (=1), sticky = all of byte0 (=0x37) -> 1
        let c = make_cols([0x37, 0x01, 0, 0, 0, 0], 9);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (1, 1)
        );
        // shift 16: guard = bit 7 of byte1 (=1), sticky = bits below
        let c = make_cols([0xFF, 0x80, 0, 0, 0, 0], 16);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (1, 1)
        );
        // shift 0: no rounding
        let c = make_cols([0xFF, 0x80, 0x01, 0x00, 0xAB, 0xCD], 0);
        assert_eq!(
            (c.guard.as_canonical_u32(), c.sticky.as_canonical_u32()),
            (0, 0)
        );
    }

    #[test]
    fn shrn_corrupt_guard_rejected() {
        // Flip the guard bit to a wrong value -> must fail.
        let mut c = make_cols([0xFF, 0x80, 0x01, 0x00, 0xAB, 0xCD], 13);
        c.guard = F::from_u32(1 - c.guard.as_canonical_u32());
        let b = run(&c);
        assert!(!b.ok(), "corrupted guard must be rejected");
    }

    #[test]
    fn shrn_corrupt_sticky_rejected() {
        let mut c = make_cols([0xFF, 0x80, 0x01, 0x00, 0xAB, 0xCD], 13);
        c.sticky = F::from_u32(1 - c.sticky.as_canonical_u32());
        let b = run(&c);
        assert!(!b.ok(), "corrupted sticky must be rejected");
    }

    // === ShlnGadget tests ===

    fn make_shln_cols(x: [u32; N], shift: usize) -> ShlnCols<F, N> {
        let shift = shift.min(N * SHLIMB_BITS);
        let (y, q, r) = run_shln(&x, shift);
        let mut limb_shift_marker = [F::ZERO; N];
        let mut bit_shift_marker = [F::ZERO; SHLIMB_BITS];
        limb_shift_marker[q] = F::ONE;
        bit_shift_marker[r] = F::ONE;
        // bit_shift_carry[i] = x[i] >> (8 - r): the high r bits.
        let carry_val = |v: u32| if r == 0 { 0 } else { v >> (SHLIMB_BITS - r) };
        ShlnCols {
            x: core::array::from_fn(|i| F::from_u32(x[i])),
            shift: F::from_u32(shift as u32),
            enable: F::ONE,
            limb_shift_marker,
            bit_shift_marker,
            bit_mult: F::from_u32(1u32 << r),
            bit_shift_carry: core::array::from_fn(|i| F::from_u32(carry_val(x[i]))),
            y: core::array::from_fn(|i| F::from_u32(y[i])),
        }
    }

    fn run_shln_test(c: &ShlnCols<F, N>) -> NumericBuilder<F> {
        let mut b = NumericBuilder::new(vec![]);
        ShlnGadget::new(VariableRangeCheckerBus::new(0, 16)).eval_shln(&mut b, c);
        b
    }

    #[test]
    fn shln_honest_witness_passes_all_shifts() {
        let mut rng = 0xDEAD_BEEF_CAFE_1234u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u32
        };
        for _ in 0..4000 {
            let shift = (next() as usize) % (N * SHLIMB_BITS);
            // x must fit in (N*8 - shift) bits so nothing overflows the field.
            let bits = N * SHLIMB_BITS - shift;
            let mut val = 0u128;
            for b in 0..bits {
                if b < 64 && (next() & 1) == 1 {
                    val |= 1u128 << b;
                }
            }
            let x: [u32; N] = core::array::from_fn(|i| ((val >> (i * SHLIMB_BITS)) & 0xFF) as u32);
            let c = make_shln_cols(x, shift);
            let b = run_shln_test(&c);
            assert!(b.ok(), "shift {} of {:?}: {:?}", shift, x, b.failures);
            let (y_exp, _, _) = run_shln(&x, shift);
            for (i, &ye) in y_exp.iter().enumerate() {
                assert_eq!(c.y[i].as_canonical_u32(), ye, "byte {}", i);
            }
        }
    }

    #[test]
    fn shln_zero_and_pow2() {
        for &shift in &[0usize, 1, 7, 8, 9, 15, 16, 23, 31, 39, 47] {
            let b = run_shln_test(&make_shln_cols([0u32; N], shift));
            assert!(b.ok(), "zero shift {}: {:?}", shift, b.failures);
            // 1 << shift
            let mut x = [0u32; N];
            x[0] = 1;
            let b = run_shln_test(&make_shln_cols(x, shift));
            assert!(b.ok(), "pow2 shift {}: {:?}", shift, b.failures);
        }
    }

    #[test]
    fn shln_corrupt_output_rejected() {
        let mut c = make_shln_cols([0x01, 0x80, 0xAB, 0xCD, 0x00, 0x00], 13);
        c.y[2] = F::from_u32(c.y[2].as_canonical_u32() ^ 1);
        assert!(!run_shln_test(&c).ok(), "corrupted output must be rejected");
    }

    #[test]
    fn shln_overflow_rejected() {
        // x has a set high byte that would shift out the top (byte 5 + limb 1 >= N=6),
        // so the no-overflow constraint must reject even an "honest-looking" witness.
        let c = make_shln_cols([0, 0, 0, 0, 0, 1], 8);
        assert!(
            !run_shln_test(&c).ok(),
            "overflowing shift must be rejected"
        );
    }

    #[test]
    fn shln_disabled_does_not_constrain() {
        let mut c = make_shln_cols([1, 2, 3, 4, 5, 6], 5);
        c.enable = F::ZERO;
        c.y[0] = F::from_u32(222);
        assert!(run_shln_test(&c).ok(), "disabled gadget must not constrain");
    }
}
