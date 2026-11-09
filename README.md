# Template-Free Heavy-Atom Bond Perception: source release candidate v2.1

Local candidate, not a public release or license grant. This archive pairs with
`BondNet_heavy_Zenodo_data_v2_1_candidate.zip`. The default manuscript method is
the single-encoder supplied-Q decoder; old two-stage code is a comparator and
scoring dependency. The missing-Q pilot and the independent ring-coherence study
are separate experiments. No manuscript conclusions have been changed.

## Validate and reproduce

Extract the source and data ZIPs into the same empty directory. Check both SHA-256
inventories. Install the primary environment from `environment.yml`; the separate
ring study recorded its own RDKit/SciPy versions. From the extraction root:

```text
python -B experiments/prob_mask_20261002/test_eval_v3.py
python -B experiments/unknown_charge_20261108/test_pilot.py
python -B bondnet2_paper/plot_current_figures.py
```

The plot uses aggregate inputs; it is not an inference rerun. The full research
guide is `bondnet2_paper/README.md`, including the missing-Q reproduction steps.

## Original frozen replay versus current diagnostics

The root holds the current diagnostic source; do not use its bytes as a substitute
for the original frozen source. `historical_v3_1/overlay/` contains the two
different original files, individually verified against the historical manifest.
`results/DIAGNOSTIC_MANIFEST_2026-11-02_v2.json` records the additive differences
and does not rewrite the original. `.gitattributes` disables newline conversion.

With authorized exact inputs restored under `INPUT_ROOT`, the following stages
all original files in a new **local-only** replay tree. The authorized source
directory is read-only to the helper. Neither command below runs inference:

```text
python -B scripts/release_replay.py stage-history --authorized-root INPUT_ROOT --inventory release_input_inventory.json --output NEW_REPLAY_TREE
python -B scripts/release_replay.py audit-saved --authorized-root INPUT_ROOT --inventory release_input_inventory.json --output NEW_RECEIPT.json
```

From the resulting replay tree, verify the original preflight with
`pwsh -NoProfile -File scripts/run_locked_eval.ps1 -Python PATH_TO_PYTHON -DryRun`.
The scoring-only path reconstructs native molecules from the saved tuples and
scores saved graph predictions on the same 491 eligible references. It does not
generate predictions, authenticate external software, or certify a new blind run.

## Separate ring-coherence study

The `ring_coherence/` subtree overlays with the same subtree in the data ZIP.
It preserves the original Windows evaluator and frozen receipts. The exact
85 Astex training ligand SDFs are available through the [PoseBusters data
release](https://zenodo.org/records/8278563), file `posebusters_paper_data.zip`,
MD5 `f004ac7c4e68317b5348497d2bb6bee6`. Extract that file under
`ring_coherence/data/processed/posebusters_paper_data/`, so that `astex_diverse_set`
is its child; verify the 85 ligand hashes in `release_input_inventory.json`.
Run Python from `ring_coherence/` for imports and the original relative paths.
The archived entry point is `python -m scripts.run_phase3_external_once --output-dir
NEW_OUTPUT`. A rerun after unblinding is a reproduction, not a new external test.

## Rights and access

This archive does not contain PDBBind structures, model weights, training caches,
the Knodle executable, or a grant to distribute them. See the matching data
inventory and author approval checklist before publishing. Never assume that the
older public DOI contains this candidate or that the two ZIPs alone reproduce all
results.

See `MISSING_Q_REPRODUCTION.md` for saved-result and inference reproduction.
