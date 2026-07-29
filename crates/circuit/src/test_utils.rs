//! A minimal *numeric* constraint evaluator for testing `SubAir`s (and any
//! constraint logic expressed purely through `assert_*` / `when`) in
//! milliseconds, without spinning up the full VM STARK prover.
//!
//! `Var` and `Expr` are both the concrete field element, so every constraint
//! expression evaluates to a single value that we check is zero. Interaction
//! sends (`push_interaction`) are ignored here — they only affect LogUp balance,
//! which is verified separately by the real prover; this builder exercises the
//! polynomial (gate) constraints only.

use std::marker::PhantomData;

use openvm_stark_backend::{
    interaction::{BusIndex, Interaction, InteractionBuilder},
    p3_air::AirBuilder,
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

/// Records constraint failures encountered during evaluation.
pub struct NumericBuilder<F> {
    /// The single trace row under test (the SubAir's aux columns, with IO
    /// values spliced in by the caller as needed).
    pub local: Vec<F>,
    pub failures: Vec<String>,
    pub num_asserts: usize,
    _m: PhantomData<F>,
}

impl<F> NumericBuilder<F> {
    pub fn new(local: Vec<F>) -> Self {
        Self {
            local,
            failures: Vec::new(),
            num_asserts: 0,
            _m: PhantomData,
        }
    }

    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

// Implement AirBuilder with Var = Expr = F, so all constraint arithmetic reduces
// to concrete field arithmetic that we can evaluate immediately.
impl<F> AirBuilder for NumericBuilder<F>
where
    F: Field + PrimeCharacteristicRing + Copy + Default + 'static,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrix<F>;

    fn main(&self) -> Self::M {
        // Constraint logic under test receives its columns explicitly (via
        // `local_core` / the SubAir context), so `main` is never read.
        RowMajorMatrix::new(vec![], self.local.len().max(1))
    }

    fn is_first_row(&self) -> Self::Expr {
        F::ONE
    }
    fn is_last_row(&self) -> Self::Expr {
        F::ZERO
    }
    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        F::ONE
    }

    fn assert_zero<E: Into<Self::Expr>>(&mut self, x: E) {
        self.num_asserts += 1;
        let v = x.into();
        if !v.is_zero() {
            self.failures
                .push("assert_zero failed: non-zero value".to_string());
        }
    }
}

impl<F> InteractionBuilder for NumericBuilder<F>
where
    F: Field + PrimeCharacteristicRing + Copy + Default + 'static,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_index: BusIndex,
        _fields: impl IntoIterator<Item = E>,
        _count: impl Into<Self::Expr>,
        _count_weight: u32,
    ) {
        // Ignored: LogUp balance is verified by the real prover, not here.
    }
    fn num_interactions(&self) -> usize {
        0
    }
    fn all_interactions(&self) -> &[Interaction<Self::Expr>] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvm_circuit_primitives::SubAir;
    use openvm_stark_backend::p3_air::AirBuilder;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    type F = BabyBear;

    /// A trivial SubAir that constrains one column to be boolean and (when
    /// `enabled`) equal to `expected`. Used only to validate NumericBuilder.
    struct CheckBool;

    impl<AB: AirBuilder> SubAir<AB> for CheckBool {
        type AirContext<'a>
            = (AB::Expr, AB::Expr)
        where
            Self: 'a,
            AB: 'a;

        fn eval<'a>(&'a self, builder: &'a mut AB, (val, expected): Self::AirContext<'a>)
        where
            AB::Var: 'a,
            AB::Expr: 'a,
        {
            builder.assert_bool(val.clone());
            builder.assert_eq(val, expected);
        }
    }

    #[test]
    fn numeric_builder_accepts_honest_witness() {
        let mut b = NumericBuilder::new(vec![F::ONE]);
        CheckBool.eval(&mut b, (F::ONE, F::ONE));
        assert!(b.ok(), "honest witness should pass: {:?}", b.failures);
        assert!(b.num_asserts > 0);
    }

    #[test]
    fn numeric_builder_rejects_non_bool() {
        let mut b = NumericBuilder::new(vec![F::from_u32(2)]);
        CheckBool.eval(&mut b, (F::from_u32(2), F::from_u32(2)));
        assert!(!b.ok(), "non-boolean value must be rejected");
    }

    #[test]
    fn numeric_builder_rejects_mismatch() {
        let mut b = NumericBuilder::new(vec![F::ZERO]);
        CheckBool.eval(&mut b, (F::ZERO, F::ONE));
        assert!(!b.ok(), "mismatched value must be rejected");
    }

    #[test]
    fn numeric_builder_when_filter_disables_constraint() {
        // when(0) should disable the constraint even if it would otherwise fail.
        let mut b = NumericBuilder::new(vec![F::from_u32(5)]);
        b.when(F::ZERO).assert_eq(F::from_u32(5), F::from_u32(0));
        assert!(b.ok(), "when(0) must disable the constraint");
    }
}
