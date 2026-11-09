# Ligand State Recovery

Research code for *Template-Free Heavy-Atom Bond Perception on Real Ligand Poses*.
The main method uses heavy-atom elements and coordinates and a supplied total
charge. The archived two-stage pipeline is a comparator. The missing-charge
pilot and ring-coherence evaluation are separate studies.

The [v2.1 data archive](https://doi.org/10.5281/zenodo.24066310) contains CCD data
and result artifacts. It does not contain the exact PDBBind pose inputs, the
new completion-head weights, training caches, or the training-derived catalogue.
The public code and data alone do not rerun the locked pose result end to end.

For prediction on user-supplied heavy-atom coordinates, see the
[supplied-charge inference instructions](tools/release_inference/README.md).

## Validate and reproduce

Use a clean code checkout and extract the v2.1 data ZIP at its root without
overwriting tracked code. Verify the archived inputs and install the environment
from `environment.yml`. From the checkout root:

```text
python -B experiments/prob_mask_20261002/test_eval_v3.py
python -B experiments/unknown_charge_20261108/test_pilot.py
python -B bondnet2_paper/plot_current_figures.py
```

The figure script uses aggregate inputs, not new inference. See
`bondnet2_paper/README.md` and `MISSING_Q_REPRODUCTION.md` for the scope of
the saved-result checks and the exact inputs required for inference.

## Frozen model and historical replay

The frozen encoder comes from the companion BondNet study. Its inference-only
equivalent is `checkpoints/joint_heavy_only/seed42/best_e2e.pt` in
`checkpoints_v4_weights.zip` from the
[BondNet v4.1 archive](https://doi.org/10.5281/zenodo.23080342). Its model tensor
digest matches the original training checkpoint recorded for this study. The
serialized files are not interchangeable for an original-byte preflight.

For historical replay, `historical_v3_1/overlay/` retains the two original
source files that differ from the current diagnostics. Use
`scripts/release_replay.py stage-history` with the exact authorized inputs in
the data archive's `release_input_inventory.json`. This includes the original
training checkpoint bytes and the required historical preflight receipts.
The scoring-only `audit-saved` command checks saved outputs, not new predictions.
See the reproduction guide and the individual commands' `--help` for paths.

## Separate ring-coherence study

The `ring_coherence/` subtree uses its separate environment and evaluation
protocol. Obtain the 85 Astex ligand SDFs from the
[PoseBusters data release](https://zenodo.org/records/8278563),
`posebusters_paper_data.zip` (MD5 `f004ac7c4e68317b5348497d2bb6bee6`), and
verify their hashes against `release_input_inventory.json`. Extract the
`astex_diverse_set` directory under
`ring_coherence/data/processed/posebusters_paper_data/`. Run the archived entry
point from `ring_coherence/`:

```text
python -m scripts.run_phase3_external_once --output-dir NEW_OUTPUT
```

## License

Original source code and associated software documentation for which the
contributors hold rights are available under the [MIT License](LICENSE).
See [license scope and third-party notices](THIRD_PARTY_NOTICES.md). That license
does not apply to the external benchmark materials, model artifacts, manuscript
or third-party binaries.
