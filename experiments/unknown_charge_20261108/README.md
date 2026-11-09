# Current single-stage BondNet without supplied net charge

Development pilot. The locked PDBBind and held-back CCD sets are not opened by this driver. The archived hybrid/two-stage ILP is not used. This is not a robustness test for deliberately wrong Q.

## Matched conditions

All methods use the same frozen PaiNN encoder, heavy-atom coordinates, training-only atom-state catalogue, 2.5 Å candidate cutoff and local beam of at most 48 bond graphs. `known_q` is the original controlled comparison: head sees reference Q, and charge-sum decoding enforces it. The no-Q branches use geometry-only rows and a fixed zero *embedding input* (not supplied by a reference). Only `frozen_fixed` additionally constrains the output to net zero. `frozen_free` and `trained_free` propose the best independent local joint (q,h) state for each atom of each bond graph. `frozen_multi_q` and `trained_multi_q` additionally compute one maximum-score joint state for each proposed net Q in {-2,-1,0,1,2} per bond graph. All proposed states share the same atom logits and are ranked without reference Q, with constructibility first. Multi-Q thus has a larger state-search budget than free or fixed; frozen versus retrained is matched within either decoder. No outcome oracle is used.

Retraining warm-starts the published head, freezes the encoder and Q embedding, and uses the existing cached 13,500 training geometries and 1,650 validation geometries. The true per-atom targets remain available for training, but every forward pass has the same Q embedding input. Six epochs, AdamW, LR 1e-4, batch 32. Choose the checkpoint by all-MASK bond CE plus the mean of the neutral and charged molecule-averaged atom-state CEs on the existing training-validation split; no external development rows enter optimization or checkpoint selection.

## Evaluation

- PDBBind 488 already-used clean development poses. The true total charge is nonzero for 19 of these.
- The 332 already-used CCD ideal development structures, intentionally enriched in ions: 100 cations, 82 anions, 50 internally charge-separated neutrals, and 100 uncharged neutrals. Do not interpret its overall score as a population-weighted rate or as real-pose charged accuracy.
- Full-denominator native identity and buildability; indexed graph strict/resonance/graph-SMILES using the existing unified scorer; selected atom-state net charge and indexed q-vector exactness; atom-charge macro-F1 over each stratum's fixed true support; native conditional on the selected net charge being correct (descriptive). A correct charge assignment is not by itself a successful build. A correct native identity necessarily has the correct Q. Known-Q charge correctness is imposed, not a prediction statistic.
- The known-Q control reproduces the 488 saved development B selections, validity and native verdicts for all three seeds. `test_pilot.py` checks the multi-Q solver against exhaustive enumeration.

## Reproduce seed 20261002

From `E:\zxcproject\bondnet2.0` in PowerShell, use new output directories if these already exist:

```powershell
$py = 'D:\minconda\python.exe'
$s = 'experiments\unknown_charge_20261108\pilot.py'
& $py -B -u $s evaluate --out work\unknown_charge_20261108\frozen_s20261002
& $py -B -u $s train --out work\unknown_charge_20261108\train_s20261002
& $py -B -u $s evaluate --out work\unknown_charge_20261108\matched_s20261002 --retrained work\unknown_charge_20261108\train_s20261002\best.pt
& $py -B experiments\unknown_charge_20261108\test_pilot.py
```

Use `--seed 20261003` or `--seed 20261004` and corresponding output/checkpoint paths for additional heads. Existing model checkpoints, manuscript, and frozen evaluation artifacts are not modified.

```powershell
& $py -B experiments\unknown_charge_20261108\summarize.py --root work\unknown_charge_20261108 --output work\unknown_charge_20261108\SUMMARY.json
```

The summary reports the mean and sample SD of three head seeds on the same molecules, plus per-seed paired discordances. It does not treat repeated seed × molecule predictions as independent observations. For the first two runs, the frozen arms and adapted arms are stored in separate run folders; the aggregator checks that molecule order and reference charges match before pairing them.
