#!/usr/bin/env python3
# P0(d) harness -- BondNet vs Knodle, unified scorer.
r"""
P0(d) -- Head-to-head harness: BondNet vs. Knodle on identical molecules.

Why this exists
---------------
Knodle (Kadukova & Grudinin, JCIM 2016, 56(8):1410-1419) reports 3.9% errors on
3,000 PDBBindCN ligands.  That number is **per molecule**, not per bond: Table 5
of the paper gives absolute counts 116/383/140/181 for Knodle/fconv/NAOMI/
I-interpret over 3,000 ligands, and 116/3000 = 3.87%.  So the headline reads as
96.1% molecule-level accuracy against our 0.566 strict graph exact.

Those two numbers are NOT the same measurement, and the difference is not in our
favour by default -- it has to be demonstrated.  Knodle's error criterion is far
more permissive than strict graph exact match:

  * "Differences in the perception of tautomeric forms were not considered as
    errors" -- explicitly guanidinium, amidinium, amino-substituted pyrimidines,
    pyrimidones and keto-enol groups.
  * Errors were adjudicated by manual visual inspection: Table 5 lists 544 raw
    inconsistencies which become 116 counted errors for Knodle.
  * The automatic comparison targets mol2 atom types and hybridisations, not an
    exact bond-order graph.
  * Structures containing metal/metalloid atoms were excluded.
  * Ground truth is the PDBBindCN mol2 files, not the CCD.
  * On Labute's 179-complex set the authors note 59 were in Knodle's training
    set; the 3,000-ligand set is held out but still same-distribution.

The only way to settle it is to run Knodle on OUR molecules and score it with
OUR scorer.  That collapses metric, distribution and resonance-convention
differences in one shot.

Input/output contract
---------------------
Per the paper, Knodle takes a PDB file of the ligand (or complex) and writes a
mol2 ligand file.  So it consumes heavy-atom (Z, R) only -- note it is NOT given
the total charge Q that BondNet receives, which is a genuine asymmetry to
disclose rather than paper over.

Subcommands
-----------
  export  : write one heavy-atom PDB per molecule from the shared SDF, plus a
            manifest binding each file to its reference index.  Atom order is
            preserved so predicted bond indices map straight back.
  adapt   : parse Knodle's mol2 outputs into the normalized, fingerprinted
            prediction JSON consumed by ``p0c_yuelbond_compare.py score``.

Workflow
--------
  python scripts/p0d_knodle_compare.py export \
      --sdf data/pdbbind_ccd_pose500.sdf --out_dir work/knodle_in

  # run Knodle over every PDB in work/knodle_in, writing <stem>.mol2 to work/knodle_out
  for f in work/knodle_in/*.pdb; do knodle "$f" "work/knodle_out/$(basename ${f%.pdb}).mol2"; done

  python scripts/p0d_knodle_compare.py adapt \
      --sdf data/pdbbind_ccd_pose500.sdf --mol2_dir work/knodle_out \
      --manifest work/knodle_in/manifest.json \
      --output results/p0c/preds_knodle_pdbbind488.json

  python scripts/p0c_yuelbond_compare.py score \
      --sdf data/pdbbind_ccd_pose500.sdf \
      --predictions results/p0c/preds_knodle_pdbbind488.json --label knodle
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.p0c_yuelbond_compare import (  # noqa: E402
    _heavy_count,
    _prediction_meta,
    _read_valid_molecules,
    _resolve,
)


def _heavy_only(mol):
    """Reference heavy-atom view; atom order defines the shared index space."""
    from rdkit import Chem

    return Chem.RemoveHs(mol, sanitize=True)


def cmd_export(a):
    from rdkit import Chem

    mols = _read_valid_molecules(a.sdf)
    out_dir = _resolve(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"sdf": os.path.basename(_resolve(a.sdf)),
                "meta": _prediction_meta(_resolve(a.sdf), mols),
                "records": []}
    written = 0
    for mol_idx, mol in enumerate(mols):
        heavy = _heavy_only(mol)
        if heavy.GetNumConformers() == 0:
            continue
        name = heavy.GetProp("_Name") if heavy.HasProp("_Name") else f"mol{mol_idx}"
        stem = f"{mol_idx:05d}_{name}"
        path = os.path.join(out_dir, stem + ".pdb")
        # PDB is Knodle's documented input format.  Crucially, do *not* emit
        # CONECT records: RDKit's default PDB writer serializes the reference
        # bond graph there, which would hand the answer to a bond-perception
        # baseline.  RDKit flavor 10 is the topology-free writer variant
        # (verified by the regression audit below): it suppresses every CONECT
        # record while preserving the atom-record order needed by the manifest.
        Chem.MolToPDBFile(heavy, path, flavor=10)
        with open(path, encoding="utf-8") as handle:
            if any(line.startswith("CONECT") for line in handle):
                raise RuntimeError(
                    f"topology leakage: RDKit wrote CONECT records to {path}; "
                    "do not run Knodle on this export"
                )
        manifest["records"].append({
            "stem": stem,
            "mol_index": mol_idx,
            "name": name,
            "n_heavy": heavy.GetNumAtoms(),
            "elements": [a_.GetAtomicNum() for a_ in heavy.GetAtoms()],
        })
        written += 1

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"[export] wrote {written} heavy-atom PDB files -> {out_dir}")
    print(f"[export] manifest -> {manifest_path}")
    print("[export] verified topology-free PDB input: ATOM/HETATM coordinates only; "
          "no CONECT records are written.")
    print("[export] Knodle receives (Z, R) only; it is NOT given the total charge Q "
          "that BondNet receives. Disclose this asymmetry when reporting.")


def _parse_mol2_bonds(path):
    """Return (n_atoms, elements, bonds) from a Tripos mol2 file.

    Knodle emits standard Tripos records.  Bond ids are 1-based; the type field
    uses ``ar`` for aromatic, ``am`` for amide (a single bond in our label
    space), and ``1``/``2``/``3`` otherwise.
    """
    section = None
    elements, bonds = [], []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                section = stripped[len("@<TRIPOS>"):].strip().upper()
                continue
            if not stripped or section is None:
                continue
            if section == "ATOM":
                parts = stripped.split()
                if len(parts) < 6:
                    continue
                # SYBYL type is element[.hybridisation]; take the element part.
                elements.append(parts[5].split(".")[0])
            elif section == "BOND":
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                i, j, kind = int(parts[1]) - 1, int(parts[2]) - 1, parts[3].lower()
                if kind == "ar":
                    order = "ar"
                elif kind in ("2", "3"):
                    order = int(kind)
                else:
                    # "1", "am", "du", "un" and anything unexpected map to single;
                    # amide is a single bond in the reference label space.
                    order = 1
                bonds.append([min(i, j), max(i, j), order])
    return len(elements), elements, bonds


_SYMBOL_TO_Z = {
    "C": 6, "N": 7, "O": 8, "F": 9, "SI": 14, "P": 15, "S": 16, "CL": 17,
    "AS": 33, "BR": 35, "I": 53, "BI": 83, "B": 5, "SE": 34, "FE": 26,
}


def cmd_adapt(a):
    mols = _read_valid_molecules(a.sdf)
    with open(_resolve(a.manifest)) as handle:
        manifest = json.load(handle)

    reference_meta = _prediction_meta(_resolve(a.sdf), mols)
    if manifest.get("meta", {}).get("sdf_sha256") != reference_meta["sdf_sha256"]:
        raise ValueError(
            "manifest was built for a different SDF than --sdf. Re-run `export` "
            "against the SDF you intend to score; silently re-binding would "
            "shift every atom index."
        )

    out = {"__meta__": reference_meta}
    mol2_dir = _resolve(a.mol2_dir)
    missing, mismatched, parsed = [], [], 0

    for record in manifest["records"]:
        path = os.path.join(mol2_dir, record["stem"] + ".mol2")
        if not os.path.exists(path):
            # A tool failure. Omitted predictions get zero credit in `score`,
            # which is the intended accounting -- do not silently substitute.
            missing.append(record["stem"])
            continue
        n_atoms, elements, bonds = _parse_mol2_bonds(path)
        if n_atoms != record["n_heavy"]:
            mismatched.append(f"{record['stem']}: mol2={n_atoms} ref={record['n_heavy']}")
            continue
        observed = [_SYMBOL_TO_Z.get(sym.upper()) for sym in elements]
        if observed != record["elements"]:
            mismatched.append(f"{record['stem']}: element sequence differs from reference order")
            continue
        out[str(record["mol_index"])] = sorted(bonds)
        parsed += 1

    output = _resolve(a.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(out, handle)

    print(f"[adapt] normalized {parsed}/{len(manifest['records'])} Knodle predictions -> {output}")
    if missing:
        print(f"[adapt] {len(missing)} molecules had no mol2 output (counted as failures): "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if mismatched:
        print(f"[adapt] {len(mismatched)} molecules rejected on atom-order/count checks:")
        for entry in mismatched[:8]:
            print(f"          {entry}")
        if len(mismatched) > 8:
            print(f"          ... and {len(mismatched) - 8} more")
        print("[adapt] Rejected molecules are omitted, not guessed. If this count is "
              "large, Knodle is reordering atoms and the adapter needs a name-based map.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write heavy-atom PDB inputs + manifest")
    export.add_argument("--sdf", required=True)
    export.add_argument("--out_dir", required=True)
    export.set_defaults(func=cmd_export)

    adapt = sub.add_parser("adapt", help="parse Knodle mol2 outputs into normalized JSON")
    adapt.add_argument("--sdf", required=True)
    adapt.add_argument("--mol2_dir", required=True)
    adapt.add_argument("--manifest", required=True)
    adapt.add_argument("--output", required=True)
    adapt.set_defaults(func=cmd_adapt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
