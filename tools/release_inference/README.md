# Supplied Charge Inference

This entry point predicts the Stage C bond and atom state from heavy-atom
elements, 3D coordinates, and a supplied total charge. It does not require
reference labels, per-molecule training data, or `sources.pt`. It does not
perform the separate Stage D graph rewrite.

1. Install the Python dependencies specified in the repository's
   `environment.yml`.
2. Download `ligand_state_recovery_inference_v1.0.0.zip` from
   [Zenodo v2.2](https://doi.org/10.5281/zenodo.24071196) (CC BY 4.0) and
   extract it into a directory named `inference_assets`. Verify its included
   `SHA256SUMS.txt`. The other ZIP in the record contains data and result
   artifacts; it is not needed for this command-line inference.
3. From the repository root, fetch the checked public encoder from the
   [BondNet v4.1 archive](https://doi.org/10.5281/zenodo.23080342):

   ```text
   python tools/release_inference/fetch_public_encoder.py --output inference_assets/encoder.pt
   ```

Write a JSON array to `input.json`. Atoms must be heavy atoms in the same order
in `Z` and `R`; coordinates are in angstroms, and `Q` is an integer:

```json
[{"id":"example","Z":[6,6],"R":[[0,0,0],[1.52,0,0]],"Q":0}]
```

Run the decoder (one line):

```text
python tools/release_inference/infer_minimal.py --input input.json --encoder inference_assets/encoder.pt --head inference_assets/head_seed20261002.pt --catalogue inference_assets/catalogue_counts.json --output prediction.json
```

The output contains candidate atom pairs (zero-based), bond labels (0 means
no bond), formal charges, hydrogen counts, the selected SMILES, and a
constructibility flag. Treat `good=false` as a failed construction; a
tentative assignment may still be returned. The supported inputs have at least
two heavy atoms and at least one candidate pair. The reference `Q` is not inferred.

The equivalent Stage C release was checked on 24 CCD and 24 PDBBind development
inputs at two noise levels across three heads. It matched the original
predictions and candidate scores on all 288 runs and 9,582 candidate assignments
in that check. This is a software parity test, not a new locked evaluation.
The inference-only encoder does not have the serialized byte hash of the
original training checkpoint and cannot replace it in historical preflight.
