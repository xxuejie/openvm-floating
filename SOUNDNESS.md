# openvm-floating soundness work — handoff notes

This branch (`soundness`) is incrementally eliminating the soundness bugs in the
RISC-V F (single-precision float) extension circuits. This file captures state,
patterns, gotchas, and the remaining plan so the work can continue consistently.

## Goal

Every arithmetic chip must **cryptographically enforce** that its output equals
the true IEEE-754 result of the operation on its inputs. Originally most chips
only decomposed/classified inputs and asserted booleans — the *result bytes*
were unconstrained for finite inputs, so a malicious prover could output
anything. We are filling in those gaps, common cases first.

The honest executor/filler already computes correct results via native `f32`
ops; the task is to add AIR constraints that *verify* the witness is internally
consistent and matches the result, for **all** inputs (including rare ones like
subnormals — a malicious prover picks the unconstrained edge case).

## Current state (commit `c3d4051` + fadd/fsub finite-no-cancellation)

Per-chip soundness of the **result**:

| Chip        | Special (NaN/±inf/±0) | Finite result                     |
| ----------- | -------------------- | --------------------------------- |
| fsgnj       | n/a                  | ✅ fully sound (bit manipulation) |
| fclass      | n/a                  | ✅ fully sound                    |
| fcmp        | ✅                   | ✅ fully sound                    |
| fminmax     | ✅                   | ✅ fully sound                    |
| fxfer       | n/a                  | ✅ fully sound (copy)             |
| fmul        | ✅                   | normal ✅, overflow ✅; subnormal/underflow ❌ |
| fadd        | ✅                   | same-sign both-normal ✅ (normal+overflow+deep); cancellation / subnormal ❌ |
| fsub        | ✅                   | opposite-sign both-normal ✅ (magnitude add); cancellation / subnormal ❌ |
| fdiv        | ✅ (pre-existing)    | ❌ finite unconstrained           |
| fma         | ✅ (pre-existing)    | ❌ finite unconstrained           |
| fsqrt       | ✅ (pre-existing)    | ❌ finite unconstrained           |
| fcvt_f2i    | ✅ (pre-existing)    | ❌ in-range truncation unconstrained |
| fcvt_i2f    | ✅ (zero only)       | ❌ RNE rounding unconstrained     |

**fadd/fsub finite no-cancellation path** (new): for two normal operands whose
exponents differ by < 25 and whose effective operation is a magnitude addition
(fadd same-sign, fsub opposite-sign), the result is fully constrained — alignment
via `ShrnGadget`, byte carry-chain sum, 1-bit carry-out normalize, RNE round, and
output reconstruction (normal result, or `±inf` on overflow). When `ediff ≥ 25`
the smaller operand is < 0.5 ULP so the result is provably the larger operand
(for fadd, a copy; for fsub, the larger operand's magnitude with sign `a_sign`),
constrained via a sound `is_deep` selector (byte-witness `deep_delta`).
Cancellation (effective subtraction) and subnormal inputs remain unconstrained
(item 2). Both chips switch from `RangeTupleCheckerBus` to
`VariableRangeCheckerBus` (mirroring fdiv) and add `fuzz_prove_fadd` /
`fuzz_prove_fsub` STARK-prove tests.

Reusable infrastructure built and tested:
- `crates/circuit/src/test_utils.rs` — `NumericBuilder`: a numeric
  `InteractionBuilder` over concrete `BabyBear` values, for millisecond-scale
  positive/negative gate tests without the full STARK prover.
- `crates/circuit/src/fp_shift.rs` — `ShrnGadget` / `ShrnCols<N>`: variable
  logical right shift of a byte-decomposed value with guard/sticky rounding
  metadata (mirrors rv32im `SRL`).

## Build / test commands

```bash
# fast: lib + unit/numeric tests (debug)
cargo build -p openvm-rv32f-circuit
cargo test  -p openvm-rv32f-circuit <name-substring>

# prove tests (RELEASE — proving in debug is far too slow)
cargo test --release -p openvm-rv32f-example --features proving chip_proving_tests::test_prove_<name>

# gates before every commit
cargo fmt --all -- --check
cargo clippy -p openvm-rv32f-circuit --all-targets --tests -- -D warnings
```

Notes:
- There is no `fast` cargo profile in this workspace (it belongs to the openvm
  repo). Use default profile.
- The risc0 guest target is **not** installed, so the guest-program prove tests
  (`test_prove_fmul_only`, `dot3`, …) won't build. Use the `chip_proving_tests`
  (hand-crafted instructions) and the `#[cfg(test)] mod` numeric tests instead.
- `OPENVM_SKIP_DEBUG` is *not* set during prove tests — keep the debug prover on;
  it pinpoints which constraint fails with a clear message.

## The verified workflow (follow this for every change)

1. **Write the new constraint logic as a standalone helper method** on the core
   AIR, e.g. `pub fn eval_finite_output<AB: InteractionBuilder>(&self, b: &mut AB, c: &FmulCoreCols<AB::Var>)`.
   It must depend only on `AB: InteractionBuilder` (**not** the `I` adapter
   interface), so it is callable from `eval` *and* from a numeric test.
2. **Numeric-test it first** (`#[cfg(test)] mod` in the same file): build a
   `FmulCoreCols<BabyBear>` witness from known inputs, call the helper via
   `NumericBuilder`, assert `ok()`; then corrupt a column and assert `!ok()`.
   This is the fast loop and catches "too strong" (honest fails) and "too weak"
   (corrupt passes) instantly.
3. **Wire it into `eval`** (call the helper) and **fill the new columns in the
   filler** (`fill_*_witness`). Add `debug_assert!`s in the witness for the key
   relations so drift surfaces during the prove test.
4. **Prove-test** the honest path in release: add/extend a `chip_proving_tests`
   test. Optionally bite-check by temporarily corrupting the witness and
   confirming the proof fails (then revert).
5. **`cargo fmt --all` then `cargo clippy ... -D warnings`**, fix everything.
6. **Commit** one logical unit with a concise imperative message (see `git log`
   for style). Keep commits focused (clear boundaries).

Reference implementation: **`fmul`** (`crates/circuit/src/fmul/core.rs`). It has
special cases (`eval_special_cases`), finite normal output (`eval_finite_output`),
and overflow (`eval_finite_overflow`). Mirror its structure for other chips.

## Patterns and gotchas (hard-won)

- **`AB::Var` is `Copy`.** Never `.clone()` a `Var` field — clippy `clone_on_copy`
  fails the build under `-D warnings`. (`AB::Expr` is `Clone` but not `Copy`, so
  `.clone()` on expressions is fine.)
- **Max constraint degree is 4.** `air_test` aborts with
  `MaxConstraintDegreeExceeded { degree: 5, max_degree: 4 }`. Keep gates ≤ 4. The
  standard trick: store sub-expressions as **witness columns** (degree 1) and
  constrain them once at higher degree, then use the columns in downstream gates.
  (See fmul's classification columns vs. fdiv's `classify` closure.)
- **`F: Var` vs `Expr`:** `AB::F - AB::Var` is **not** allowed. Lift the constant
  with `AB::Expr::from(f)` first. `Var * AB::F`, `Var + Var`, `Expr + Var` are fine.
- **Padding rows are all-zero.** Any unconditional constraint must hold on a row
  of zeros. The fdiv-style classification (constrain `exp_is_zero`/`exp_is_max`/
  `mant_is_zero` columns `when(is_valid)`, then derive `nan/inf/zero` as
  *unconditional* products of those columns) vanishes on padding because all
  factor columns are 0 there. Use it; don't gate the derived flags on `is_valid`
  (degree would blow up).
- **`range_tuple_bus` is a lookup bus, not a plain range check.** Every
  `bus.send(vec![x, y]).eval(builder, enable)` in `eval` must be balanced by a
  `range_chip.add_count(&[x, y])` in the filler for the same key, or the LogUp
  check fails ("Bus N failed to balance the multiplicities"). The bus sizes are
  `[256, 2048]` (x∈[0,256), y∈[0,2048)). For a dynamic-width range check, use a
  **`VariableRangeCheckerBus`** instead (see `fp_shift.rs` — `send(v, max_bits)`
  takes a dynamic `max_bits`; `range_check(v, 8)` is for a static width).
- **Wiring a `VariableRangeCheckerBus` into a chip** that doesn't have one yet:
  add it to the core AIR struct + `new`, add a `SharedVariableRangeCheckerChip` to
  the filler + `new`, and pass `range_checker_bus` / `range_checker.clone()` in
  `crates/circuit/src/extension/mod.rs` (both `extend_circuit` and
  `extend_prover`). Mirror how `fdiv` is wired (it already has one).
- **`is_special` gating must be sound.** Several chips gate normalization on
  `not(is_special)` where `is_special` is *witness-provided*. After the finite
  path is constrained, redefine `is_special` to a **sound, input-derived**
  selector (e.g. fmul's `s_finite = 1 - s_nan - s_inf - s_zero`) so a prover can't
  fraudulently set `is_special=1` to skip normalization. (fmul already does this.)
- **Subtle FP semantics to get exactly right** (verify against the `_s` helper):
  - fmul `0 * inf` is invalid → canonical NaN.
  - **fadd** `inf + (-inf)` is NaN; same-sign `inf + inf` is inf.
  - **fsub** `inf - inf` is NaN **only for same-sign infs**; opposite-sign
    `inf - inf` yields ±inf (the old fsub witness had this wrong — fixed).
  - Result sign of `0 +/- 0`: fadd `= a_sign AND b_sign`; fsub `= a_sign AND (NOT b_sign)`.
  - inf sign for add/sub: the infinite operand's sign (pick a's if a is inf,
    else b's — well-defined because the both-inf NaN cases are excluded).
- **Overflow needs the rounding carry.** "result exp field == 255" alone is not
  enough — a prover could claim overflow for a normal product. Verify
  `exp_normal + c2 ≥ 255`, where `c2` (rounding carry) is recomputed via an
  all-ones cascade so it's available outside the normal-result path. See fmul's
  `eval_finite_overflow`.

## Remaining plan (common → rare)

Each is a multi-step build using the workflow above. The shift gadget handles
all variable shifts.

1. ~~**fadd / fsub finite, normal result, same-sign (no cancellation)**~~ — **DONE.**
   Both chips now fully constrain the magnitude-addition path (alignment via
   `ShrnGadget` → byte carry-chain sum → 1-bit carry-out normalize → RNE round →
   output recon / overflow), with `ediff ≥ 25` deep-clamp and STARK-prove tests.
2. **fadd / fsub finite, cancellation (effective subtraction)** — needs a variable
   *left*-shift normalization (count leading zeros). Build a left-shift gadget
   (analogue of `ShrnGadget`), or generalize. The existing finite-path columns
   (alignment, sum) are reused; only the normalize stage changes from a 1-bit
   right shift to a variable left shift. Subnormal inputs also fall here (result
   may be subnormal) and need the deep-clamp threshold re-derived for subtraction.
3. **fmul subnormal / underflow** — wire `ShrnGadget` into fmul; for
   `exp_normal ≤ 0`, shift `norm` right by `25 - exp_normal`, round, result is
   subnormal (exp field 0) or zero. Boundary: rounding up to smallest normal.
   Deep underflow (`shift ≥ 48`) → zero.
4. **fcvt_i2f / fcvt_f2i** — int↔float shift/round (fcvt_f2i is RTZ truncation,
   no rounding metadata needed; fcvt_i2f is RNE with leading-zero normalize).
5. **fma finite** — 48-bit product + 3-way alignment + single RNE rounding.
6. **fdiv / fsqrt finite** — long division / integer isqrt (hardest; rarest).

### Concrete next step

**fadd/fsub cancellation (variable left-shift normalize).** Build a left-shift
gadget `ShlnGadget` mirroring `ShrnGadget` (byte-granular + sub-byte one-hot,
range-checked low/high witnesses) but shifting left, producing the shift amount
= leading-zero count. Then gate the cancellation branch (`is_add=0` for fadd,
`is_add=1` for fsub, both-normal, `raw_top < 2^23`): normalize `raw_sum` left by
the CLZ, recompute guard/sticky from the bits shifted out, and reuse the existing
RNE-round + output-recon helpers. The same-sign no-cancellation columns already
added (large_sig/small_sig, align, raw_sum, rs_high7/rs_lowbit, etc.) are reused.

## Reference pointers in the upstream openvm dep

Useful patterns in the checked-out openvm git source
(`~/.cargo/git/checkouts/openvm-dac496c47acce1af/2c97976/`):
- `extensions/rv32im/circuit/src/shift/core.rs` — the SRL/SLL/SRA variable-shift
  implementation that `fp_shift.rs` is based on.
- `extensions/rv32im/circuit/src/divrem/` — long-division machinery (for fdiv).
- `crates/circuits/primitives/src/` — `is_less_than_array`, `var_range`,
  `range_tuple`, `sub_air`, etc.
