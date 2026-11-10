# Public-source replay of 491 deposited ligand poses

This is a reproducibility reconstruction of an already evaluated cohort, **not** a new independent holdout. It contains no PDBBind ligand files, raw training rows, or model checkpoints. The plan identifies public wwPDB coordinate files, ligand instances, CCD atom order, and source hashes. The source records were downloaded anonymously and checked against the plan. In comparison with the historical PDBBind-derived evaluation, 488/491 coordinate arrays are identical. Three differ by no more than 0.0005 Å per coordinate component; all 491 ordered bond, formal-charge and attached-H references agree. The supplied total charge `Q` comes from the aligned CCD component, not from measured bound-state protonation.

Coordinates and components are obtained from [RCSB PDB](https://www.rcsb.org/pages/policies) and the [wwPDB versioned archive](https://www.wwpdb.org/ftp/pdb-versioned-ftp-site). Use the identifiers and source checksums for provenance. No restricted PDBBind material is needed for this replay. The plan describes source-derived identifiers and hashes; the source data retain their original archive terms.

## Rebuild inputs and run the public model

From the repository root, in the supplied environment (RDKit 2025.09.6; install Biopython 1.87 for mmCIF parsing), run:

```text
python scripts/fetch_public_pose_replay_inputs.py --plan reproduction/public_pose491/replay_plan.json --source-root pose_cache --ccd-dir pose_cache/ccd
python scripts/replay_public_locked_poses.py --plan reproduction/public_pose491/replay_plan.json --source-root pose_cache --ccd-dir pose_cache/ccd --output-sdf pose_replay/public_pose491.sdf
python scripts/benchmark_public_pose_replay.py --sdf pose_replay/public_pose491.sdf --outdir pose_replay
```

The fetch step verifies 491 coordinate files and 488 CCD component files. It stops rather than silently substituting a revised file if upstream bytes change. The rebuilt SDF hash on the verified Windows/RDKit environment is `a992f740ad314c8fbcfa71dd8158c744df7a75a9702d19c5d95e413e607820c9` (SDF serialization may vary by platform). The third command writes `inputs_zrq.json` and independently runs the OpenBabel 3.1.0 geometry-only bond-perception comparator with the shared graph scorer. Labels are never model inputs.

Download the three heads and aggregate catalogue from [Zenodo v2.2](https://doi.org/10.5281/zenodo.24071196) and the encoder from the [BondNet v4.1 archive](https://doi.org/10.5281/zenodo.23080342) as described in the [inference guide](../../tools/release_inference/README.md). For each of `20261002`, `20261003`, `20261004` run the following command with `SEED` replaced in both paths:

```text
python tools/release_inference/infer_minimal.py --input pose_replay/inputs_zrq.json --encoder inference_assets/encoder.pt --head inference_assets/head_seedSEED.pt --catalogue inference_assets/catalogue_counts.json --output pose_replay/stage_c_SEED.json --device cuda
python scripts/benchmark_public_pose_replay.py --sdf pose_replay/public_pose491.sdf --outdir pose_replay --predictions pose_replay/stage_c_20261002.json pose_replay/stage_c_20261003.json pose_replay/stage_c_20261004.json
```

The second command is run once, after all three predictions exist. The verified supplied-charge native counts are 407, 414 and 400 of 491; the constructible counts are 484, 483 and 485. This route reproduces current-model Stage C, not training, the archived MILP checkpoint, or the separate Stage D graph rewrite. The raw OpenBabel graph counts are 33 strict, 220 unified resonance-aware and 225 charge-blind graph-SMILES.

## External and descriptive controls

```text
python scripts/evaluate_pose_external_states.py --method OpenBabel --sdf pose_replay/public_pose491.sdf --edges pose_replay/openbabel_edges.jsonl --catalogue inference_assets/catalogue_counts.json --output pose_replay/openbabel_native.json
python scripts/export_knodle_geometry_only.py --sdf pose_replay/public_pose491.sdf --output pose_replay/knodle_geometry_only
```

The OpenBabel native route adds hydrogens with no pH correction or supplied charge; the separate catalogue route takes the raw OpenBabel bonds, uses the same supplied `Q` as our model and adds no neural atom-state scores. The two routes give 223/491 native matches each, with 322 and 373 constructible outputs, respectively. They need not recover the same molecules. Both are evaluated over the same denominator, including output failures. The Knodle export is charge-column- and `CONECT`-free; it does not run Knodle or replace the historical Knodle results.

The separate [YuelBond](https://pmc.ncbi.nlm.nih.gov/articles/PMC13067372/) control uses the authors' source code and the `geom_3d.ckpt` checkpoint from [their archive](https://zenodo.org/records/15353365); neither is bundled here. Point `scripts/benchmark_yuelbond_public_poses.py` at a local author source checkout and checkpoint. The script uses `Z,R` only and the upstream no-connectivity 3D inference path. One unsupported element inventory counts as a failed prediction. Then run `scripts/evaluate_pose_external_states.py` with `--method YuelBond` and the resulting `yuelbond_edges.jsonl` to apply the shared catalogue and total-charge constraint. This is a composite system, not a native YuelBond charge/hydrogen predictor (155/491 native matches, 326/491 constructible outputs).
