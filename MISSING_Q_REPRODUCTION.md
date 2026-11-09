# Missing-net-charge pilot: reproduce the saved results

This is a retrospective development analysis of the current single-stage model.
It does not use the locked pose or held-back CCD cohorts. It does not change the
supplied-Q primary benchmark. The development CCD cohort contains 332 molecules;
the pose cohort contains 488. Each cohort has 3 head seeds and 6 arms.

## CCD: score from the public candidates, without checkpoints

Extract the v2.1 source and data archives into a new directory. The archives
share an identical copy of this guide. From the extraction root, with the
primary scoring environment (RDKit 2025.09.6, Python, PyTorch and NumPy):

```powershell
$study = 'experiments/unknown_charge_20261108'
$run = 'work/unknown_charge_20261108'
python -B "$study/test_pilot.py"
python -B "$study/audit_saved.py" --root $run --datasets ccd_dev --output ccd_saved_audit.json
python -B "$study/summarize.py" --root $run --datasets ccd_dev --output ccd_summary_recomputed.json
```

The 18 CCD cells contain selected atom states, per-molecule verdicts and the
full positive predicted/reference edge union for rescoring. The audit rebuilds
the molecules, checks graph and complete-state verdicts, and verifies subgroup
metrics. It does not run model inference. The `ccd_dev` section of the recomputed
summary is comparable directly to `work/unknown_charge_20261108/SUMMARY.json`.
`missing_q_ccd_cells.json` records per-cell paths and hashes. The first two
seeds' original frozen and adapted files are in separate run directories.

## Poses and model execution: authorized local overlay only

The public candidates do not include PDBBind-derived reference files or pose
predictions, model weights, the joint training cache, or the catalogue source.
They cannot reproduce pose scores or run inference on their own. Use the
authorized exact files identified by `release_input_inventory.json`; there is
no promise of unrestricted public access. The separate local overlay, if supplied
to you with permission, is **not for upload or onward distribution**. Extract it
over the candidates in a disposable local directory, not into the original repo.

With the local overlay, use the same audit and summary commands with
`--datasets pose ccd_dev` and new output filenames. To regenerate predictions
from the retained heads or repeat training, use new output paths (the commands
below illustrate one seed; repeat for 20261003 and 20261004):

```powershell
$seed = 20261002
python -B -u "$study/pilot.py" evaluate --seed $seed --retrained "$run/train_s$seed/best.pt" --out "replay_eval_s$seed"
python -B -u "$study/pilot.py" train --seed $seed --out "replay_train_s$seed"
```

Use the documented CUDA environment for model execution. The retraining command
starts with the original seed head and writes a new adapted checkpoint. It does
not overwrite any archived prediction or checkpoint. Hardware and library
differences can affect training; saved-result rescoring and checksum verification
do not imply a bit-identical training rerun.
