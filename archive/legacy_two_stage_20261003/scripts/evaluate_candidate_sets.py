#!/usr/bin/env python3
r"""Coverage of ranked candidate graphs, at the labeling and species level.

Why
---
Single-answer decoding throws away recoverable information.  On the 488-molecule
PDBBind pose set the correct graph is within a two-edge flip of the argmax for
87% of molecules, while the argmax itself is exact for 58%.  Most of that gap is
not model error: 52% of residual edge errors are on symmetric terminal-oxy
groups whose assignment is decided by a hydrogen the input does not contain.

This script reports two curves, and the difference between them is the point:

  coverage@N (labeling) : the reference bond-order graph appears verbatim among
                          the top N decoded labelings
  coverage@N (species)  : some candidate in the top N is the reference species,
                          allowing the representation equivalences that geometry
                          provably cannot resolve (terminal-oxy permutation,
                          class-5 for a {1,2} pair, ring Kekule alternation)

A candidate list should be presented at the species level.  Ranking notational
variants against each other would inflate the apparent uncertainty -- 21.7% of
molecules already differ from the reference by notation alone.

Also reports the *distinct species count* per molecule, which is the honest
measure of how ambiguous a recovery actually is.  A molecule with eight
candidate labelings that collapse to one species is not ambiguous.

Usage
-----
  python scripts/evaluate_candidate_sets.py \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --stage1_ckpt checkpoints/diverse80k_heavy_stage1/best.pt \
      --refiner_ckpt checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt \
      --n_solutions 8 --output results/candidate_sets_to50.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--n_solutions", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian coordinate noise (A). Coverage@N is expected to separate "
                        "single- vs two-stage as noise makes connectivity ambiguous.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single_stage", action="store_true",
                   help="Evaluate a single-stage checkpoint (forces single-stage inference; "
                        "otherwise the checkpoint's own training config decides).")
    p.add_argument("--allow_non_kekulizable_candidates", dest="require_kekulizable_candidates",
                   action="store_false",
                   help="Diagnostic only: retain invalid aromatic labelings instead of re-decoding.")
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--candidate_radius", type=float, default=2.5)
    p.add_argument("--max_neighbors", type=int, default=15)
    p.add_argument("--candidate_knn", type=int, default=0)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--dump_candidates", type=int, default=0,
                   help="Print the decoded candidate SMILES for the first N molecules. "
                        "Use this to see what the alternatives actually are instead of "
                        "inferring it from aggregate counts.")
    p.add_argument("--device", default="auto")
    p.set_defaults(require_kekulizable_candidates=True)
    return p.parse_args()


def main():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = parse_args()

    from scripts.benchmark_templatefree_motifs import heavy_graph, load_bondnet
    from scripts.classify_candidate_ambiguity import classify

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device)
    # bondnet_predict's argument object doubles as the decoder config.
    args.input_mode = "heavy_only"
    args.bondnet_decode_mode = "full_graph"
    args.heavy_latent_charge_h = True
    args.aromatic_constraints = False
    args.aromatic_bridge_repair = False
    args.restrict_delocalized = False
    args.h_candidate_cutoff = 1.4
    args.methods = ["bondnet"]
    stage1, (refiner, train_args) = load_bondnet(args, device)
    # Mirror the benchmark: single_stage is driven by the checkpoint's own config,
    # and can be forced on explicitly. Keeps eval consistent with how the
    # checkpoint was trained.
    ckpt_single = bool(getattr(train_args, "single_stage", False))
    train_args.single_stage = ckpt_single or bool(getattr(args, "single_stage", False))
    if train_args.single_stage:
        print("using single-stage inference (full candidate message graph)")

    allowed = set(args.allowed_atomic_nums)
    max_n = args.n_solutions
    hit_label = [0] * (max_n + 1)
    hit_species = [0] * (max_n + 1)
    species_counts = Counter()
    skeleton_counts = Counter()
    hit_skeleton = [0] * (max_n + 1)
    records = []
    fallback_molecules = 0
    fallback_candidates = 0
    fallback_reasons = Counter()
    total = 0

    for source_index, mol in enumerate(Chem.SDMolSupplier(args.reference_sdf, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        if args.max_mols is not None and total >= args.max_mols:
            break
        total += 1

        truth = heavy_graph(mol)
        # When RDKit can sanitize a decoded candidate, canonical isomeric
        # SMILES is a stronger equivalence oracle than local ring heuristics.
        # In particular it handles fused aromatic systems consistently.  The
        # graph-rule classifier remains the explicit fallback for candidates
        # that cannot be built due to latent heavy-only atom states.
        truth_species_key = "smiles::" + Chem.MolToSmiles(mol)
        adjacency = defaultdict(set)
        for i, j in truth:
            adjacency[i].add(j)
            adjacency[j].add(i)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(source_index)

        args._mol_index = source_index
        candidates = enumerate_candidates(mol, args, stage1, refiner, train_args, device,
                                          max_n)

        # Rank-resolved coverage. Species-level collapses notational variants,
        # so its rank index counts distinct species, not distinct labelings.
        label_rank = None
        species_rank = None
        seen_species = []
        seen_skeleton = []
        skeleton_rank = None
        n_fallback = 0
        for rank, candidate in enumerate(candidates):
            predicted = candidate["bonds"]
            signature, skeleton, build_status = species_signature(
                mol, candidate, adjacency, return_status=True)
            if signature.startswith("fallback::"):
                n_fallback += 1
                fallback_candidates += 1
                fallback_reasons[build_status] += 1
            if signature == truth_species_key:
                verdict = "identical" if predicted == truth else "representation_equivalent"
            else:
                verdict, _ = classify(mol, truth, predicted, adjacency)
            if label_rank is None and verdict == "identical":
                label_rank = rank
            if signature not in seen_species:
                seen_species.append(signature)
            if skeleton not in seen_skeleton:
                seen_skeleton.append(skeleton)
            if verdict in ("identical", "representation_equivalent"):
                if species_rank is None:
                    species_rank = len(seen_species) - 1
                if skeleton_rank is None:
                    skeleton_rank = len(seen_skeleton) - 1
        if n_fallback:
            fallback_molecules += 1
        species_counts[len(seen_species)] += 1
        skeleton_counts[len(seen_skeleton)] += 1
        if total <= args.dump_candidates:
            print(f"\n--- {name}  ({len(candidates)} labelings, "
                  f"{len(seen_species)} species, {len(seen_skeleton)} skeletons)")
            for rank, candidate in enumerate(candidates):
                sig, skel = species_signature(mol, candidate, adjacency)
                verdict, _ = classify(mol, truth, candidate["bonds"], adjacency)
                print(f"   #{rank} obj={candidate['objective']:9.3f}  {verdict:26s} "
                      f"{sig[:80]}")
            n_failed = sum(1 for c in candidates
                           if species_signature(mol, c, adjacency)[1].startswith("skeleton_failed"))
            if n_failed:
                print(f"   ({n_failed}/{len(candidates)} neutralisation failures)")

        for n in range(1, max_n + 1):
            if label_rank is not None and label_rank < n:
                hit_label[n] += 1
            if species_rank is not None and species_rank < n:
                hit_species[n] += 1
            if skeleton_rank is not None and skeleton_rank < n:
                hit_skeleton[n] += 1
        records.append({"name": name, "n_labelings": len(candidates),
                        "n_species": len(seen_species),
                        "n_skeletons": len(seen_skeleton),
                        "n_fallback_candidates": n_fallback,
                        "label_rank": label_rank, "species_rank": species_rank,
                        "skeleton_rank": skeleton_rank})
        if total % 50 == 0:
            print(f"  {total} molecules", flush=True)

    print(f"\nmolecules: {total}\n")
    print(f"{'N':>3}  {'labeling':>10}  {'equiv.':>10}  {'skeleton':>10}")
    for n in range(1, max_n + 1):
        print(f"{n:>3}  {hit_label[n] / total:>10.4f}  {hit_species[n] / total:>10.4f}"
              f"  {hit_skeleton[n] / total:>10.4f}")
    print("  labeling = exact reference bond-order graph")
    print("  equiv.   = reference-equivalent molecule (canonical SMILES when possible;")
    print("             terminal-oxo/aromatic rules only when RDKit cannot kekulize)")
    print("  skeleton = correct heavy-atom bond graph, protonation neutralised")
    print("             (charge/H are declared latent in the heavy-only setting)")
    print(f"  canonicalisation fallback: {fallback_candidates}/{total * max_n} candidates "
          f"across {fallback_molecules}/{total} molecules")
    if fallback_reasons:
        print("  fallback reasons: " + ", ".join(
            f"{reason}={count}" for reason, count in fallback_reasons.most_common()))

    print("\ndistinct candidates per molecule:")
    print(f"   {'count':>5}  {'species':>8}  {'skeleton':>9}")
    for count in sorted(set(species_counts) | set(skeleton_counts)):
        print(f"   {count:>5}  {species_counts.get(count, 0):>8}  {skeleton_counts.get(count, 0):>9}")

    payload = {
        "noise": args.noise,
        "single_stage": bool(train_args.single_stage),
        "n_molecules": total,
        "n_solutions": max_n,
        "coverage_labeling": {str(n): hit_label[n] / total for n in range(1, max_n + 1)},
        # ``coverage_species`` is retained for old plotting notebooks.  The
        # explicit name is the one that should be used in new reports: without
        # hydrogens, some valid aromatic candidates cannot be uniquely
        # kekulized by RDKit and are scored by the declared equivalence rules.
        "coverage_equivalence": {str(n): hit_species[n] / total for n in range(1, max_n + 1)},
        "coverage_species": {str(n): hit_species[n] / total for n in range(1, max_n + 1)},
        "coverage_skeleton": {str(n): hit_skeleton[n] / total for n in range(1, max_n + 1)},
        "distinct_skeleton_histogram": {str(k): v for k, v in sorted(skeleton_counts.items())},
        "distinct_species_histogram": {str(k): v for k, v in sorted(species_counts.items())},
        "fallback_candidates": fallback_candidates,
        "fallback_molecules": fallback_molecules,
        "fallback_reasons": dict(fallback_reasons),
        "per_molecule": records,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


_BOND_TYPE = None


def _candidate_mol(candidate, return_error=False):
    """Build an RDKit molecule from one decoded candidate.

    Deliberately does NOT use the decoder's ``implicit_h``. In the heavy-only
    setting charge and H are latent -- they carry no term in the MILP objective,
    so the solver leaves implicit_h at zero for every atom. Building with those
    values produced radicals: toluene came out as ``[C]c1[c][c][c][c][c]1``
    instead of ``Cc1ccccc1``, every candidate sanitised to the same garbage or
    failed outright, and the species/skeleton curves became identical for a
    reason that had nothing to do with chemistry.

    Letting RDKit infer hydrogens from valence is what makes two notations of
    one species collapse: with inference on, both writings of neutral acetic
    acid canonicalise to ``CC(=O)O`` and both writings of acetate to
    ``CC(=O)[O-]``.

    Formal charges are still taken from the decoder, since those it does solve
    for as feasibility variables.
    """
    global _BOND_TYPE
    from rdkit import Chem

    def finish(mol, error=None):
        return (mol, error) if return_error else mol

    if _BOND_TYPE is None:
        _BOND_TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                      3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
                      5: Chem.BondType.SINGLE}
    elems = candidate["elems"].tolist()
    charges = candidate["formal_charge"].tolist()
    edges = [tuple(int(x) for x in e) for e in candidate["edges"]]
    labels = [int(v) for v in candidate["edge_labels"]]
    # Kekulize delocalized (class-5) terminal-oxy groups to {single, double}
    # before building. Mapping class 5 -> SINGLE unconditionally (as the
    # production builder does) turns a *neutral* carboxyl COOH into a geminal
    # diol C(OH)(OH): a different molecule. That was the entire 83-molecule gap
    # between the SMILES oracle and classify(). A delocalized carboxyl IS a
    # carboxyl, whose canonical form is {1,2}, so resolving it here makes the
    # {5,5} prediction and the {1,2} reference collapse to one species.
    labels = _kekulize_delocalized_oxy(elems, edges, labels)
    rw = Chem.RWMol()
    for z, q in zip(elems, charges):
        atom = Chem.Atom(int(z))
        atom.SetFormalCharge(int(q))
        rw.AddAtom(atom)
    aromatic_atoms = set()
    for (i, j), label in zip(edges, labels):
        if label > 0:
            rw.AddBond(int(i), int(j), _BOND_TYPE[int(label)])
            if int(label) == 4:
                aromatic_atoms.update((int(i), int(j)))
    mol = rw.GetMol()
    # Match the production sanitization path exactly.  Marking only bonds as
    # aromatic is insufficient for several N/S-containing aromatic systems;
    # RDKit must also know which atoms participate before Kekulization.
    for idx in aromatic_atoms:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    for (i, j), label in zip(candidate["edges"], candidate["edge_labels"]):
        if int(label) == 4:
            mol.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=False)
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return finish(None, f"sanitize:{type(exc).__name__}")
    return finish(mol)


def _kekulize_delocalized_oxy(elems, edges, labels):
    """Rewrite delocalized terminal-oxy groups to one double + rest single.

    Operates only on a centre (C/N/P/S) whose incident *delocalized* edges all
    go to a terminal oxygen (degree 1). Delocalized here means class 5
    (BondNet's label) OR class 4 aromatic on a terminal oxygen (Knodle marks
    carboxylate/sulfonate/phosphate oxygens ``ar`` with O.co2). Both are the
    same chemistry -- a resonance oxy group -- and an aromatic bond to a
    degree-1 oxygen can never be a real ring bond, so treating it here is safe
    and does not touch genuine aromatic rings. The Kekule form {double, single,
    ...} canonicalises identically to the conventional reference, which is what
    lets Knodle's aromatic encoding collapse to the same species as CCD's {1,2}.
    """
    incident = defaultdict(list)
    degree = defaultdict(int)
    for edge_id, (i, j) in enumerate(edges):
        if labels[edge_id] > 0:
            degree[i] += 1
            degree[j] += 1
    # A terminal oxygen carrying an aromatic (class-4) bond is a delocalized oxy
    # marker, not a ring atom; fold it in with class 5.
    for edge_id, (i, j) in enumerate(edges):
        if labels[edge_id] == 5:
            incident[i].append(edge_id)
            incident[j].append(edge_id)
        elif labels[edge_id] == 4:
            for a, b in ((i, j), (j, i)):
                if elems[b] == 8 and degree[b] == 1 and elems[a] in (6, 7, 15, 16):
                    incident[a].append(edge_id)

    out = list(labels)
    for centre, edge_ids in incident.items():
        if elems[centre] not in (6, 7, 15, 16):
            continue
        oxy_edges = []
        for edge_id in edge_ids:
            i, j = edges[edge_id]
            other = j if i == centre else i
            if elems[other] == 8 and degree[other] == 1 and out[edge_id] in (4, 5):
                oxy_edges.append(edge_id)
        if len(oxy_edges) < 2:
            continue
        # One double, the remainder single -- the standard Kekule carboxyl/
        # phosphate/sulfonate form.
        out[oxy_edges[0]] = 2
        for edge_id in oxy_edges[1:]:
            out[edge_id] = 1
    return out


def _looks_like_radical_artifact(mol):
    """True if the molecule carries the zero-hydrogen signature of the old bug.

    A neutral carbon with fewer than four bond orders and no hydrogens is not a
    normal drug-like atom; it is what appears when hydrogens are forced to zero.
    Guarding on it means that failure mode cannot come back silently.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() == 0 and atom.GetTotalValence() < 4:
            return True
    return False


def species_signature(mol, candidate, adjacency, return_status=False):
    """(species key, protonation-neutralised skeleton key) for one candidate."""
    from rdkit import Chem

    def finish(species, skeleton, status):
        return (species, skeleton, status) if return_status else (species, skeleton)

    built, build_error = _candidate_mol(candidate, return_error=True)
    if built is not None and not _looks_like_radical_artifact(built):
        full = "smiles::" + Chem.MolToSmiles(built)
        skeleton_mol = Chem.Mol(built)
        for atom in skeleton_mol.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(skeleton_mol)
            return finish(full, "skeleton::" + Chem.MolToSmiles(skeleton_mol), "canonical")
        except Exception:
            return finish(full, "skeleton_failed::" + full, "skeleton_sanitize_failed")

    from scripts.classify_candidate_ambiguity import terminal_oxy_groups

    predicted = candidate["bonds"]
    groups = terminal_oxy_groups(mol, adjacency)
    grouped_edges = {e for edges in groups.values() for e in edges}
    parts = [(edge, predicted.get(edge, 0))
             for edge in sorted(set(predicted) | grouped_edges)
             if edge not in grouped_edges]
    for centre, edges in sorted(groups.items()):
        parts.append((("centre", centre), tuple(sorted(predicted.get(e, 0) for e in edges))))
    fallback = "fallback::" + repr(parts)
    status = build_error or "radical_signature"
    return finish(fallback, fallback, status)


def enumerate_candidates(mol, args, stage1, refiner, train_args, device, max_n):
    """Top-N decoded graphs as {(i, j): label} dicts, best first.

    Delegates to bondnet_predict's n_solutions mode so the Stage-1 bridge,
    candidate-edge construction and heavy/all-atom handling are shared with
    ordinary evaluation. Rebuilding that path here would be a second place for
    edge indices to drift out of sync with the scorer.
    """
    from scripts.benchmark_templatefree_motifs import bondnet_predict, coordinate_view

    # Fixed per-molecule seed so the injected noise is reproducible and identical
    # across the single-stage and two-stage arms being compared.
    seed = getattr(args, "seed", 42) + getattr(args, "_mol_index", 0) * 1009
    elems, coord = coordinate_view(mol, getattr(args, "noise", 0.0), seed)
    total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return bondnet_predict(elems, coord, total_charge, stage1, refiner, train_args,
                           args, device, n_solutions=max_n)


if __name__ == "__main__":
    main()
