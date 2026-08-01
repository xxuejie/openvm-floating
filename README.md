# openvm-floating

An [OpenVM](https://github.com/openvm-org/openvm) extension adding the RISC-V F
(single-precision floating-point) extension — and, later, possibly the D
extension.

The 13 chips cover: `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, `fma`, `fsgnj`,
`fminmax`, `fcmp`, `fclass`, `fxfer` (`fmv.x.s`/`fmv.s.x`), `fcvt_f2i`
(`fcvt.w.s`/`fcvt.wu.s`), and `fcvt_i2f` (`fcvt.s.w`/`fcvt.s.wu`).

## Full soundness

Every arithmetic chip **cryptographically enforces** that its output equals the
true IEEE-754 result of the operation on its inputs. A malicious prover cannot
pick an unconstrained witness column or input class to forge a result.

### Architecture

- **Input-derived classification, not witness-provided.** Special-case
  selectors (`s_nan`, `s_inf`, `s_zero`, `s_special`, …) are redefined as
  *sound, input-derived* expressions (e.g. `s_finite = 1 - s_nan - s_inf -
  s_zero`), so a prover cannot fraudulently set a selector to skip a path.
- **Decompositions are pinned.** Every byte-decomposition half
  (`a_mant2`/`a_hi`/`b_mant2`/…, the `±128`/`±2^24` splits) is 7-bit
  range-checked on a `VariableRangeCheckerBus`, so `exp_lo`/`sign` flips cannot
  reclassify an operand (finite↔NaN/inf, sign reversal).
- **Reusable shift gadgets.** `ShrnGadget` / `ShlnGadget` (`fp_shift.rs`)
  implement variable logical right/left shifts with guard/sticky rounding
  metadata. They are sound **given their callers range-check the inputs**; the
  decomposition bugs historically lived in caller code, which is why the
  range checks are part of each chip.
- **Wide working fields for cancellation.** Heavy-cancellation paths (fadd/fsub,
  fma) use a 48-bit working significand so alignment-lost guard/sticky bits
  survive the left-shift normalization; 32-bit working fields are provably
  insufficient there.
- **Every valid row takes exactly one path.** Chips assert a partition over the
  output classes (`is_special` / `s_pzero` / `s_finite`, normal / overflow /
  subnormal / exact-zero), so there are no unconstrained rows.

### Verification harnesses

- **`NumericBuilder`** (`crates/circuit/src/test_utils.rs`) — a numeric
  `InteractionBuilder` over concrete `BabyBear` values for millisecond-scale
  gate tests (honest rows pass; corrupted columns fail) without the STARK
  prover.
- **Corruption sweeps** (`corruption_sweep` / `assert_columns_constrained` in
  `test_utils.rs`) — perturb every witness column of an honest row by +1 and
  re-eval; every accepted perturbation is triaged and whitelisted with a written
  justification. Live for the small chips (fcmp, fminmax, fclass, fsgnj, fxfer,
  fmem), plus focused result-byte + selector pinning sweeps for fadd/fsub/fmul.
- **STARK-level bite-checks** (`crates/example/src/lib.rs`, `bite_checks` module)
  — run a program through the full VM, verify the honest trace with
  `debug_proving_ctx` (constraints + LogUp bus balance), corrupt one witness
  cell, and assert the corruption is caught. Covers all 14 chips (a result-byte
  corruption plus opcode-routing / result-class selector bites).

### Cross-AS FMV and FLW/FSW

`fmv.x.s` / `fmv.s.x` are cross-address-space moves (FP register file in
AS 5 → integer registers in AS 1, and back) in F-extension mode. fxfer uses a
dedicated `Rv32FpMoveAdapter` whose `read_as`/`write_as` are witness columns
filled from the instruction's `d`/`e` fields and pinned by both the program bus
and the memory bus, so FMV is sound and complete in both Zfinx and F-extension
mode.

`flw` / `fsw` are implemented by the `fmem` chip with a dedicated
`Rv32FpMemAdapter`: it reads the base address register (AS 1), computes
`mem_ptr = base + sign_extend(imm)` (2×16-bit carry chain, 4-byte alignment
enforced), and moves a word between AS 2 at `mem_ptr` and the FP register file.
Both instructions work in Zfinx and F-extension modes, including negative
immediate offsets.

## Build / test commands

```bash
# fast: lib + unit/numeric tests (debug)
cargo build -p openvm-rv32f-circuit
cargo test  -p openvm-rv32f-circuit <name-substring>

# prove + bite-check tests (RELEASE — proving in debug is far too slow)
cargo test --release -p openvm-rv32f-example --features proving chip_proving_tests::test_prove_<name>
cargo test --release -p openvm-rv32f-example --features proving bite_

# gates before every commit
cargo fmt --all -- --check
cargo clippy -p openvm-rv32f-circuit --all-targets --tests -- -D warnings
```

Notes:
- There is no `fast` cargo profile in this workspace (it belongs to the openvm
  repo). Use default profile.
- The risc0 guest target is **not** installed, so the guest-program prove tests
  (`test_prove_fmul_only`, `dot3`, …) won't build. Use the
  `chip_proving_tests` (hand-crafted instructions) and the `#[cfg(test)] mod`
  numeric tests instead.
- Keep the debug prover on (`OPENVM_SKIP_DEBUG` unset): it pinpoints which
  constraint fails with a clear message.

## Soundness audit history

See `AUDIT.md` (parent directory) for the full per-chip audit, the fixed
findings with commit hashes, and the hardening plan.
