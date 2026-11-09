# Frozen configuration v3.1: raw and eligible populations

This is an engineering correction to v3, not a new model or a new blind test.
The full method and statistical specification remain in
`FROZEN_CONFIG_V3_2026-10-04.md`; that historical document and its manifest are
preserved unchanged. This addendum overrides only the population contract.

## Population contract

The locked SDF contains **500 raw records**. The pre-existing allowed-element
rule (`6,7,8,9,14,15,16,17,33,35,53,83`) admits **491 records**. Eight records
contain boron (`6jn6`, `1nyy`, `3ixg`, `5ee8`, `1p01`, `3sjt`, `3e9b`, `4hww`)
and one contains ruthenium (`3p3j`). These nine exclusions are determined solely
by reference element identity, not by predictions or method success.

The complete eligible name list and exclusion reasons are frozen in
`data/pdbbind_pose_locked500_population_v3_1.json`. All graph endpoints, paired
comparisons and native `(B,h,q)` endpoints use these same 491 names. Missing or
failed predictions among these names remain failures in the denominator.
Raw exporters may still read all 500 records to preserve original indexing.
Decoder preparation filters by the frozen eligible names before row encoding.

The raw SDF is unchanged, SHA256:
`fe032b5100e4b066faa4dbbd6a1b50ae6fbf7a8be73dbaa1476cc276abe160d2`.

## Unchanged choices

Checkpoints, model parameters, allowed elements, decoder hyperparameters,
equivalence definitions and statistical tests are unchanged. The final Holm
family still contains 78 tests; the main-only 42-test family is provisional.
Only the erroneous use of raw count 500 as eligible count is corrected.

## Interrupted-run provenance and resumption

The v3 run failed at the old comparator's unified scorer population assertion.
Its partial files and prior recovery provenance are retained. An explicit
state-migration receipt verifies that every previously frozen file outside the
enumerated population-repair source files is unchanged, including all model
weights and inputs. It preserves the old state and manifest hash before
updating the current state's hash to the new v3.1 manifest.

This run is **not certified as an untouched test**. Resumption is an engineering
continuation, not permission to tune against locked-set outcomes. No new model
training or locked-set scoring is performed while making this repair.

New byte manifest: `results/FROZEN_MANIFEST_V3_1_2026-10-05.json`.
After tests, manifest verification and explicit state migration, resume with
`powershell -ExecutionPolicy Bypass -File scripts/run_locked_eval.ps1 -Resume`.
