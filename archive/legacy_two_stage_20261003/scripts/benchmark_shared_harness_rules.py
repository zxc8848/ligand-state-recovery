"""Same-harness rule baselines for heavy-only graph recovery.

Motivation
----------
``benchmark_templatefree_motifs.py`` runs RDKit ``DetermineBonds``, which
performs its *own* connectivity determination and then assigns bond orders in
one shot.  On heavy-only input it fails almost entirely, so its score conflates
two distinct causes: (a) the xyz2mol-family algorithm being unsuited to missing
hydrogen, and (b) that particular API being called outside its documented
contract (a complete atom inventory).

This script separates the two.  Both rule arms receive the *identical*
externally computed adjacency matrix (``covalent_factor`` x sum of covalent
radii -- the same rule the vendored xyz2mol wrapper already used) and are asked
only to assign bond orders under the given total charge:

  * ``xyz2mol``           -- vendored Jensen reference implementation
  * ``rdkit_bondorders``  -- RDKit ``DetermineBondOrders`` on a pre-bonded mol

OpenBabel is deliberately NOT re-run here: its API accepts no external
adjacency (``ConnectTheDots`` always perceives its own), so it cannot join the
shared harness.  Its numbers are carried over unchanged from the existing
``pdbbind_ccd_pose500_heavy_bondnet.json`` run over the same molecules.

Two stages, so that an expensive prediction pass is never repeated:

  ``--stage predict``  per-molecule predicted bonds -> resumable JSONL
  ``--stage score``    JSONL -> strict / resonance-aware / aromatic-Kekule
                       metrics, using the scoring functions imported verbatim
                       from ``benchmark_templatefree_motifs``

xyz2mol bond-order enumeration is combinatorial and blows up on large ligands
(a 43-heavy-atom PDBBind ligand takes ~27 s), so every molecule runs in a child
process under a hard wall-clock budget.  Exceeding it is recorded as a declared
timeout failure and reported, never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------
# The reference benchmark imports torch at module scope for its BondNet arm and
# decorates ``bondnet_predict`` with ``@torch.no_grad()``.  This script never
# touches BondNet (its numbers are reused from the existing result JSON), so a
# minimal stub lets us import the *exact* scoring functions without pulling in
# the training stack.
def _install_torch_stub():
    if "torch" in sys.modules:
        return
    stub = types.ModuleType("torch")

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    stub.no_grad = lambda: _NoGrad()
    stub.long = "long"
    stub.float32 = "float32"
    stub.tensor = lambda *a, **k: None
    stub.Tensor = object
    stub.cuda = _Cuda()
    stub.device = lambda spec: spec
    sys.modules["torch"] = stub


_install_torch_stub()

from scripts.benchmark_templatefree_motifs import (  # noqa: E402
    aromatic_kekule_align_six,
    empty_cell,
    heavy_graph,
    load_vendored_xyz2mol,
    motif_bins,
    resonance_align_six,
    summarize,
    true_graph,
    update,
)

METHODS = ("xyz2mol", "rdkit_bondorders")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["predict", "score"], required=True)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--jsonl", required=True, help="Resumable per-molecule prediction record.")
    p.add_argument("--output", help="Metrics JSON (required for --stage score).")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+", default=None,
                   help="Supported-element filter; must match the BondNet run for a shared denominator.")
    p.add_argument("--covalent_factor", type=float, default=1.25,
                   help="Shared adjacency cutoff = factor * (r_cov_i + r_cov_j).")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="Declared per-molecule wall-clock budget in seconds.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--wall_budget", type=float, default=None,
                   help="Stop and checkpoint after this many seconds (for chunked execution).")
    p.add_argument("--baseline_json", default=None,
                   help="Existing result JSON supplying carried-over BondNet/OpenBabel arms.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# shared harness
# ---------------------------------------------------------------------------

def build_adjacency(atoms, xyz, factor):
    """Shared connectivity for every rule arm: purely element + geometry."""
    from rdkit import Chem

    periodic = Chem.GetPeriodicTable()
    n = len(atoms)
    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = factor * (periodic.GetRcovalent(atoms[i]) + periodic.GetRcovalent(atoms[j]))
            if float(np.linalg.norm(xyz[i] - xyz[j])) <= cutoff:
                adjacency[i, j] = adjacency[j, i] = 1
    return adjacency


def xyz2mol_shared(atoms, xyz, adjacency, total_charge, factor):
    from rdkit import Chem

    xyz2mol = load_vendored_xyz2mol()
    mols = xyz2mol(atoms, xyz, adjacency, factor, charge=int(total_charge),
                   allow_charged_fragments=True, use_graph=True,
                   use_huckel=False, embed_chiral=False)
    mol = mols[0] if isinstance(mols, (tuple, list)) else mols
    Chem.SanitizeMol(mol)
    return true_graph(mol)


def rdkit_bondorders_shared(atoms, xyz, adjacency, total_charge, factor):
    """RDKit's own xyz2mol port, given the SAME adjacency as the reference impl.

    ``DetermineBondOrders`` skips connectivity perception and solves only the
    bond-order / formal-charge assignment, making it directly comparable to the
    vendored ``xyz2mol`` call above.

    Two settings matter for a matched harness:

    * ``allowChargedFragments=True`` (the default) is the counterpart of
      xyz2mol's ``allow_charged_fragments=True``: missing hydrogens may be
      absorbed as formal charges rather than radicals.
    * ``maxIterations=0`` removes RDKit's internal iteration cap, so both arms
      are bounded by the *same* external wall-clock budget rather than one
      being cut off by an implementation-specific counter.

    Note: the RDKit docstring for this function says it disregards pre-existing
    bonds; that text is inherited from ``DetermineBonds``.  Verified empirically
    that removing an edge from the supplied adjacency changes the solve (it
    isolates the atom and raises), so the shared adjacency is genuinely used.
    """
    from rdkit import Chem
    from rdkit.Chem.rdDetermineBonds import DetermineBondOrders

    rw = Chem.RWMol()
    for z in atoms:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(atoms))
    for index, position in enumerate(np.asarray(xyz).tolist()):
        conf.SetAtomPosition(index, position)
    rw.AddConformer(conf, assignId=True)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if adjacency[i, j]:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    DetermineBondOrders(mol, charge=int(total_charge), allowChargedFragments=True,
                        maxIterations=0)
    Chem.SanitizeMol(mol)
    return true_graph(mol)


PREDICTORS = {
    "xyz2mol": xyz2mol_shared,
    "rdkit_bondorders": rdkit_bondorders_shared,
}


def _worker(connection, method, atoms, xyz, adjacency, total_charge, factor):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        predicted = PREDICTORS[method](atoms, np.asarray(xyz), np.asarray(adjacency),
                                       total_charge, factor)
        payload = {"ok": True,
                   "bonds": [[int(i), int(j), int(label)]
                             for (i, j), label in predicted.items()]}
    except Exception as exc:  # noqa: BLE001 - rule baselines fail in many ways
        payload = {"ok": False,
                   "error": f"{type(exc).__name__}: {str(exc)[:160]}".replace("\n", " ")}
    connection.send(payload)
    connection.close()


def _spawn(method, atoms, xyz, adjacency, total_charge, factor):
    """Hard per-molecule budget requires a killable child process.

    A signal-based alarm is not sufficient: xyz2mol spends most of its time
    inside RDKit/NetworkX C extensions where Python signal handlers cannot run.
    A separate process that can be SIGKILLed is the only reliable bound.
    """
    parent, child = mp.Pipe(duplex=False)
    process = mp.Process(target=_worker,
                         args=(child, method, atoms, xyz, adjacency, total_charge, factor))
    process.start()
    child.close()
    return process, parent, time.perf_counter()


def _collect(process, parent, started, timeout):
    """Non-blocking check; returns a payload once finished or over budget."""
    elapsed = time.perf_counter() - started
    if parent.poll(0):
        try:
            payload = parent.recv()
        except EOFError:
            payload = {"ok": False, "error": "EOFError: worker died without a result"}
        process.join(timeout=5)
        parent.close()
        payload["seconds"] = elapsed
        return payload
    if not process.is_alive():
        process.join(timeout=5)
        parent.close()
        return {"ok": False, "error": "ProcessDied: worker exited without a result",
                "seconds": elapsed}
    if elapsed > timeout:
        process.kill()
        process.join(timeout=5)
        parent.close()
        return {"ok": False, "error": "Timeout: declared per-molecule budget exceeded",
                "seconds": elapsed}
    return None


# ---------------------------------------------------------------------------
# molecule iteration
# ---------------------------------------------------------------------------

def iter_accepted(reference_sdf, allowed_atomic_nums):
    """Yield (index, mol) for molecules inside the shared denominator.

    Mirrors ``benchmark_templatefree_motifs`` exactly: hydrogens stripped, then
    an optional supported-element filter.  ``index`` counts accepted molecules
    only, so it is a stable key across runs.
    """
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(reference_sdf, removeHs=False)
    accepted = 0
    skipped = 0
    for mol in supplier:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            mol = Chem.RemoveHs(mol, sanitize=True)
        if allowed_atomic_nums is not None:
            observed = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
            if not observed.issubset(set(allowed_atomic_nums)):
                skipped += 1
                continue
        yield accepted, mol, skipped
        accepted += 1


def stage_predict(args):
    path = Path(args.jsonl)
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                done.add((record["index"], record["method"]))

    started = time.perf_counter()
    pending = []
    for index, mol, _ in iter_accepted(args.reference_sdf, args.allowed_atomic_nums):
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(index)
        for method in args.methods:
            if (index, method) not in done:
                pending.append((index, name, method, atoms, xyz, total_charge))

    if not pending:
        print(f"predict: nothing to do; {len(done)} records already in {path}")
        return

    completed = 0
    queue = list(pending)
    active = []
    with path.open("a", encoding="utf-8") as handle:
        while queue or active:
            out_of_budget = (args.wall_budget is not None
                             and time.perf_counter() - started > args.wall_budget)
            while queue and len(active) < args.workers and not out_of_budget:
                index, name, method, atoms, xyz, total_charge = queue.pop(0)
                adjacency = build_adjacency(atoms, xyz, args.covalent_factor)
                process, parent, launched = _spawn(method, atoms, xyz, adjacency,
                                                   total_charge, args.covalent_factor)
                active.append({"process": process, "parent": parent, "started": launched,
                               "index": index, "name": name, "method": method,
                               "n_heavy": len(atoms), "total_charge": int(total_charge)})
            if not active:
                break
            still_active = []
            for job in active:
                payload = _collect(job["process"], job["parent"], job["started"], args.timeout)
                if payload is None:
                    still_active.append(job)
                    continue
                payload.update({key: job[key] for key in
                                ("index", "name", "method", "n_heavy", "total_charge")})
                handle.write(json.dumps(payload) + "\n")
                handle.flush()
                completed += 1
            active = still_active
            if active:
                time.sleep(0.02)
    print(f"predict: +{completed} records ({len(done) + completed} total), "
          f"{len(pending) - completed} remaining, {time.perf_counter() - started:.1f}s")


# ---------------------------------------------------------------------------
# scoring and statistics
# ---------------------------------------------------------------------------

def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    phat = successes / n
    denominator = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar(a_outcomes, b_outcomes):
    """Exact paired test on per-molecule binary outcomes (a = reference arm)."""
    from math import comb

    b = sum(1 for x, y in zip(a_outcomes, b_outcomes) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(a_outcomes, b_outcomes) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {"b_only": b, "c_only": c, "p_value": 1.0}
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b_only": b, "c_only": c, "p_value": min(1.0, 2 * tail)}


def stage_score(args):
    records = defaultdict(dict)
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[record["index"]][record["method"]] = record

    result = {m: defaultdict(empty_cell) for m in args.methods}
    resonance_result = {m: defaultdict(empty_cell) for m in args.methods}
    aromatic_result = {m: defaultdict(empty_cell) for m in args.methods}
    failures = {m: Counter() for m in args.methods}
    seconds = {m: [] for m in args.methods}
    per_molecule = {m: [] for m in args.methods}
    names = []

    accepted = 0
    skipped_unsupported_elements = 0
    missing = 0
    for index, mol, skipped in iter_accepted(args.reference_sdf, args.allowed_atomic_nums):
        skipped_unsupported_elements = skipped
        if any(method not in records.get(index, {}) for method in args.methods):
            missing += 1
            continue
        truth = heavy_graph(mol)
        bins = motif_bins(mol)
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        names.append(records[index][args.methods[0]]["name"])
        for method in args.methods:
            record = records[index][method]
            if record["ok"]:
                predicted = {(int(i), int(j)): int(label) for i, j, label in record["bonds"]}
            else:
                predicted = {}
                failures[method][record["error"].split(":")[0]] += 1
            seconds[method].append(record.get("seconds", 0.0))
            predicted = {pair: label for pair, label in predicted.items()
                         if atoms[pair[0]] != 1 and atoms[pair[1]] != 1}
            resonance_predicted = resonance_align_six(predicted, truth, mol)
            aromatic_predicted = aromatic_kekule_align_six(resonance_predicted, truth, mol)

            strict_ok = all(predicted.get(pair, 0) == truth.get(pair, 0)
                            for pair in set(truth) | set(predicted))
            resonance_ok = all(resonance_predicted.get(pair, 0) == truth.get(pair, 0)
                               for pair in set(truth) | set(resonance_predicted))
            per_molecule[method].append({"name": record["name"], "n_heavy": record["n_heavy"],
                                         "strict": int(strict_ok),
                                         "resonance": int(resonance_ok),
                                         "ok": bool(record["ok"])})
            for motif in bins:
                update(result[method][motif], truth, predicted)
                update(resonance_result[method][motif], truth, resonance_predicted)
                update(aromatic_result[method][motif], truth, aromatic_predicted)
        accepted += 1

    def with_ci(summary_map, outcomes_key, method):
        out = {}
        for motif, cell in summary_map[method].items():
            entry = summarize(cell)
            n = entry["n"]
            successes = round(entry["strict_graph_exact_rate"] * n) if n else 0
            low, high = wilson_interval(successes, n)
            entry["strict_graph_exact_ci95"] = [low, high]
            out[motif] = entry
        return out

    import rdkit

    payload = {
        "task": "same-harness rule baselines for heavy-only graph recovery from (Z, R, Q)",
        # DetermineBonds/DetermineBondOrders behaviour is version sensitive: the
        # same heavy-only diagnostic scored 0.0020 strict under the environment
        # that produced pdbbind_ccd_pose500_heavy_bondnet.json and 0.0000 under
        # RDKit 2026.03.4.  Both are effectively zero, but the version must be
        # reported alongside any RDKit rule-baseline number.
        "rdkit_version": rdkit.__version__,
        "reference_sdf": str(args.reference_sdf),
        "n_molecules": accepted,
        "n_missing_predictions": missing,
        "skipped_unsupported_elements": skipped_unsupported_elements,
        "allowed_atomic_nums": args.allowed_atomic_nums,
        "covalent_factor": args.covalent_factor,
        "timeout_seconds": args.timeout,
        "methods": args.methods,
        "harness_note": {
            "xyz2mol": "external shared adjacency; assigns bond orders only",
            "rdkit_bondorders": "external shared adjacency; assigns bond orders only",
            "openbabel": "NOT re-run: API accepts no external adjacency; carried over",
        },
        "results": {m: with_ci(result, "strict", m) for m in args.methods},
        "resonance_aware_results": {m: with_ci(resonance_result, "resonance", m)
                                    for m in args.methods},
        "aromatic_kekule_equiv_results": {m: {k: summarize(v) for k, v in aromatic_result[m].items()}
                                          for m in args.methods},
        "failures": {m: dict(failures[m]) for m in args.methods},
        "mean_seconds_per_molecule": {m: (float(np.mean(seconds[m])) if seconds[m] else None)
                                      for m in args.methods},
        "max_seconds_per_molecule": {m: (float(np.max(seconds[m])) if seconds[m] else None)
                                     for m in args.methods},
        "per_molecule": per_molecule,
    }

    # Paired comparison against the carried-over BondNet arm, matched by name.
    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        payload["carried_over_from"] = str(args.baseline_json)
        payload["carried_over_arms"] = {
            arm: {"strict": baseline["results"]["0.0"][arm]["all"],
                  "resonance": baseline["resonance_aware_results"]["0.0"][arm]["all"]}
            for arm in baseline["results"]["0.0"]
        }
        for arm, cell in payload["carried_over_arms"].items():
            n = cell["strict"]["n"]
            for key in ("strict", "resonance"):
                rate = cell[key]["strict_graph_exact_rate"]
                cell[key]["strict_graph_exact_ci95"] = list(wilson_interval(round(rate * n), n))

    Path(args.output).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.output}  n={accepted} (missing predictions: {missing})")
    for method in args.methods:
        cell = payload["results"][method].get("all")
        if cell:
            low, high = cell["strict_graph_exact_ci95"]
            print(f"  {method:18s} strict={cell['strict_graph_exact_rate']:.4f} "
                  f"[{low:.3f},{high:.3f}] "
                  f"resonance={payload['resonance_aware_results'][method]['all']['strict_graph_exact_rate']:.4f} "
                  f"typed_recall={cell['true_edge_typed_recall']:.4f} "
                  f"fails={sum(failures[method].values())} "
                  f"mean_s={payload['mean_seconds_per_molecule'][method]:.2f}")


def main():
    args = parse_args()
    if args.stage == "predict":
        stage_predict(args)
    else:
        if not args.output:
            raise ValueError("--output is required for --stage score")
        stage_score(args)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
