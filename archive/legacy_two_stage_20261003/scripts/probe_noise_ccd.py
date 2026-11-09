#!/usr/bin/env python3
r"""Paper 2, reviewer-requested robustness probe: Knodle vs BondNet off-distribution.

Motivation
----------
On the 488 clean PDBBind poses BondNet and the fixed Knodle binary are tied on the
primary resonance-aware endpoint (0.801 vs 0.797).  That tie is measured in Knodle's
home distribution: Knodle is an SVM trained on PDBBindCN ligand poses, so the clean
pose benchmark is the single condition most favourable to it.  This probe asks the
obvious follow-up question: does the tie survive when the input leaves that
distribution?

Two axes are swept, with both methods receiving BIT-IDENTICAL coordinates:

  * coordinate noise   -- Gaussian sigma in {0.00, 0.05, 0.10, 0.15} A on the 488 poses,
                          the same levels already reported for the nobond diagnosis
                          (Table 6 of the manuscript);
  * chemical domain    -- the CCD ideal heavy-only control (n=1000), at sigma 0.00 and
                          sigma 0.10.

The perturbed coordinates are written ONCE, to `reference_for_scoring.sdf` inside each
probe directory.  BondNet is then run on that file with `--noise 0.0`, and Knodle is run
on topology-free PDB exports of the same file.  Neither method re-samples noise, so the
comparison is exactly paired per molecule and per condition.

Phases (separate because they have different dependencies)
----------------------------------------------------------
  prepare : rdkit only      -- build every probe directory (noised SDF + CONECT-free PDB)
  bondnet : rdkit + torch   -- run BondNet on each probe's reference_for_scoring.sdf
  adapt   : rdkit only      -- normalize Knodle mol2 output into prediction JSON
  score   : rdkit only      -- score both methods with the same method-neutral functions

Knodle itself is run between `prepare` and `adapt` by scripts/run_knodle_noise_ccd.sh
(a static Linux binary; it needs no Python at all).

Usage (Windows, from the repository root)
-----------------------------------------
  D:\minconda\python.exe scripts\probe_noise_ccd.py prepare
  D:\minconda\python.exe scripts\probe_noise_ccd.py bondnet
  ... run scripts/run_knodle_noise_ccd.sh under Linux/WSL ...
  D:\minconda\python.exe scripts\probe_noise_ccd.py adapt
  D:\minconda\python.exe scripts\probe_noise_ccd.py score
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PROBE_ROOT = REPO / "work" / "paper2_noise_ccd_probe"
RESULT_DIR = REPO / "results" / "paper2_noise_ccd_probe"

STAGE1_CKPT = REPO / "checkpoints" / "diverse80k_heavy_stage1" / "best.pt"
REFINER_CKPT = REPO / "checkpoints" / "diverse80k_heavy_struct6_vector_hardneg_to50" / "best.pt"

ALLOWED = [6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83]

# (condition key, source sdf, sigma, max_mols)
CONDITIONS = [
    ("pdbbind_sigma0.00", "data/pdbbind_ccd_pose500.sdf", 0.00, None),
    ("pdbbind_sigma0.05", "data/pdbbind_ccd_pose500.sdf", 0.05, None),
    ("pdbbind_sigma0.10", "data/pdbbind_ccd_pose500.sdf", 0.10, None),
    ("pdbbind_sigma0.15", "data/pdbbind_ccd_pose500.sdf", 0.15, None),
    ("ccd_sigma0.00", "data/ccd_heavy_leakagefree.sdf", 0.00, 1000),
    ("ccd_sigma0.10", "data/ccd_heavy_leakagefree.sdf", 0.10, 1000),
]

SEED = 42


def cond_dir(key: str) -> Path:
    return PROBE_ROOT / key.replace(".", "p")


def selected(args):
    """Conditions to act on, optionally narrowed by --only (substring match)."""
    if not getattr(args, "only", None):
        return CONDITIONS
    picked = [c for c in CONDITIONS if any(token in c[0] for token in args.only)]
    if not picked:
        raise SystemExit(f"--only {args.only} matched no condition; "
                         f"known: {[c[0] for c in CONDITIONS]}")
    return picked


# --------------------------------------------------------------------------- prepare
def phase_prepare(_args):
    """Write the noised reference SDF and the CONECT-free PDB inputs for each condition.

    The noise draw is keyed on (seed, source molecule index, sigma), matching
    scripts/prepare_knodle_probe.py, so a condition is reproducible on its own and
    sigma levels are independent draws rather than a nested sequence.
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    from types import SimpleNamespace
    from scripts.p0d_knodle_compare import cmd_export

    RDLogger.DisableLog("rdApp.*")
    allowed = set(ALLOWED)
    cache: dict[str, list] = {}

    for key, rel_sdf, sigma, max_mols in selected(_args):
        directory = cond_dir(key)
        # A condition is complete only when the manifest exists; an interrupted
        # export leaves PDB files behind without one, and must be redone.
        if (directory / "pdb" / "manifest.json").exists() and not _args.force:
            print(f"[prepare] {key}: manifest present, skipping")
            continue
        cache_key = f"{rel_sdf}|{max_mols}"
        if cache_key not in cache:
            source = []
            supplier = Chem.SDMolSupplier(str(REPO / rel_sdf), removeHs=False)
            for index, mol in enumerate(supplier):
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                mol = Chem.RemoveHs(mol, sanitize=True)
                if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
                    continue
                source.append((index, mol))
                if max_mols is not None and len(source) >= max_mols:
                    break
            cache[cache_key] = source
        source = cache[cache_key]

        directory.mkdir(parents=True, exist_ok=True)
        sdf_path = directory / "reference_for_scoring.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for source_index, mol in source:
            copied = Chem.Mol(mol)
            if sigma:
                rng = np.random.default_rng(SEED + source_index * 1009 + round(sigma * 1_000_000))
                conformer = copied.GetConformer()
                xyz = np.asarray(conformer.GetPositions(), dtype=float)
                xyz = xyz + rng.normal(0.0, sigma, size=xyz.shape)
                for atom_index, point in enumerate(xyz):
                    conformer.SetAtomPosition(atom_index, [float(v) for v in point])
            writer.write(copied)
        writer.close()

        cmd_export(SimpleNamespace(sdf=str(sdf_path), out_dir=str(directory / "pdb")))
        print(f"[prepare] {key}: {len(source)} molecules -> {directory}")


# --------------------------------------------------------------------------- bondnet
def phase_bondnet(args):
    """Run BondNet on each probe's already-noised coordinates (so --noise is 0.0)."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for key, _rel_sdf, _sigma, _max_mols in selected(args):
        directory = cond_dir(key)
        sdf_path = directory / "reference_for_scoring.sdf"
        if not (directory / "pdb" / "manifest.json").exists():
            print(f"[bondnet] {key}: incomplete prepare, skipping")
            continue
        if not sdf_path.exists():
            raise SystemExit(f"missing {sdf_path}; run `prepare` first")
        per_edge = directory / "bondnet_per_edge.jsonl"
        if per_edge.exists() and not args.force:
            print(f"[bondnet] {key}: per-edge dump exists, skipping")
            continue
        cmd = [
            sys.executable, "-u", str(REPO / "scripts" / "benchmark_templatefree_motifs.py"),
            "--reference_sdf", str(sdf_path),
            "--output", str(directory / "bondnet_benchmark.json"),
            "--per_molecule_out", str(directory / "bondnet_per_molecule.json"),
            "--per_edge_out", str(per_edge),
            "--methods", "bondnet",
            "--input_mode", "heavy_only",
            "--noise", "0.0",
            "--stage1_ckpt", str(STAGE1_CKPT),
            "--refiner_ckpt", str(REFINER_CKPT),
            "--allowed_atomic_nums", *[str(z) for z in ALLOWED],
        ]
        # One retry: the inference process has been observed to die with a
        # native fail-fast (Windows 0xC0000409) that does not reproduce on a
        # fresh process, so a single condition failing must not abort the sweep.
        for attempt in (1, 2):
            print(f"[bondnet] {key}: attempt {attempt}")
            completed = subprocess.run(cmd, cwd=str(REPO))
            if completed.returncode == 0 and per_edge.exists():
                break
            per_edge.unlink(missing_ok=True)
            print(f"[bondnet] {key}: attempt {attempt} failed "
                  f"(exit {completed.returncode})")
        else:
            failures.append(key)
            continue

    if failures:
        raise SystemExit(
            "[bondnet] these conditions did not complete and must be re-run: "
            + ", ".join(failures))


# --------------------------------------------------------------------------- adapt
def phase_adapt(_args):
    from types import SimpleNamespace
    from scripts.p0d_knodle_compare import cmd_adapt

    for key, _rel_sdf, _sigma, _max_mols in selected(_args):
        directory = cond_dir(key)
        mol2_dir = directory / "mol2"
        if not mol2_dir.exists():
            print(f"[adapt] {key}: no mol2 directory, skipping")
            continue
        cmd_adapt(SimpleNamespace(
            sdf=str(directory / "reference_for_scoring.sdf"),
            mol2_dir=str(mol2_dir),
            manifest=str(directory / "pdb" / "manifest.json"),
            output=str(directory / "preds_knodle.json"),
        ))
        print(f"[adapt] {key}: -> {directory / 'preds_knodle.json'}")


# --------------------------------------------------------------------------- score
def phase_score(_args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "note": "Knodle vs BondNet under coordinate noise and on the CCD ideal control. "
                "Both methods receive the identical perturbed coordinates written once to "
                "reference_for_scoring.sdf; BondNet is run with --noise 0.0 on that file and "
                "Knodle on CONECT-free PDB exports of the same file. Scoring uses the same "
                "method-neutral functions as scripts/score_knodle_vs_bondnet.py.",
        "seed": SEED,
        "conditions": {},
    }
    for key, rel_sdf, sigma, max_mols in selected(_args):
        directory = cond_dir(key)
        out_json = RESULT_DIR / f"{key.replace('.', 'p')}.json"
        knodle_pred = directory / "preds_knodle.json"
        per_edge = directory / "bondnet_per_edge.jsonl"
        if not (knodle_pred.exists() and per_edge.exists()):
            print(f"[score] {key}: missing inputs, skipping "
                  f"(knodle={knodle_pred.exists()}, bondnet={per_edge.exists()})")
            continue
        cmd = [
            sys.executable, "-u", str(REPO / "scripts" / "score_knodle_vs_bondnet.py"),
            "--knodle_pred", str(knodle_pred),
            "--reference_sdf", str(directory / "reference_for_scoring.sdf"),
            "--bondnet_per_edge", str(per_edge),
            "--output", str(out_json),
            # A molecule Knodle fails to convert is scored as an incorrect
            # prediction, per the manuscript's declared baseline accounting,
            # rather than dropped from the comparison.
            "--count_missing_as_failure",
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO))
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        summary["conditions"][key] = {
            "source_sdf": rel_sdf,
            "sigma": sigma,
            "max_mols": max_mols,
            "n": payload["bondnet"]["n"],
            "bondnet": {k: payload["bondnet"][k] for k in ("strict", "resonance", "species")},
            "knodle": {k: payload["knodle"][k] for k in ("strict", "resonance", "species")},
            "wilson_ci95": {
                "bondnet": payload["bondnet"]["wilson_ci95"],
                "knodle": payload["knodle"]["wilson_ci95"],
            },
            "paired_tests": payload["paired_tests"],
        }
    path = RESULT_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[score] wrote {path}")

    header = f"{'condition':<20}{'n':>6}{'BN strict':>11}{'KN strict':>11}" \
             f"{'BN reson':>11}{'KN reson':>11}{'BN smiles':>11}{'KN smiles':>11}"
    print(header)
    for key, entry in summary["conditions"].items():
        b, k = entry["bondnet"], entry["knodle"]
        print(f"{key:<20}{entry['n']:>6}{b['strict']:>11.4f}{k['strict']:>11.4f}"
              f"{b['resonance']:>11.4f}{k['resonance']:>11.4f}"
              f"{b['species']:>11.4f}{k['species']:>11.4f}")


PHASES = {"prepare": phase_prepare, "bondnet": phase_bondnet,
          "adapt": phase_adapt, "score": phase_score}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--force", action="store_true",
                        help="recompute a phase even if its outputs already exist")
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict the phase to conditions whose key contains "
                             "any of these substrings, e.g. --only ccd sigma0.15")
    args = parser.parse_args()
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
