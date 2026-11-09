"""Template-free heavy-atom graph recovery benchmark with chemistry motifs.

The input SDF carries reference bonds solely for scoring.  Every method sees a
new bondless molecule consisting of heavy-atom element types, coordinates, and
the reference net formal charge; no SMILES/CCD/PDB template is exposed.

Motif bins overlap by design: ``aromatic``, ``conjugated_non_aromatic``, and
``hypervalent_sulfur`` are truth-defined strata in addition to ``all``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VENDORED_MOLCRAFT_SRC = ROOT / "external" / "MolCraftDiffusion" / "src"
_XYZ2MOL = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input_mode", choices=["heavy_only", "all_atom"], default="heavy_only",
                   help="heavy_only is an H-removed transfer stress test; all_atom preserves observed H coordinates.")
    p.add_argument("--diagnostic_allow_rdkit_heavy", action="store_true",
                   help="Run RDKit DetermineBonds on an incomplete heavy-only inventory as a diagnostic. "
                        "This is not a chemically fair supported RDKit protocol; exceptions count as failures.")
    p.add_argument("--bondnet_decode_mode", choices=["full_graph", "heavy_projection"],
                   default="full_graph",
                   help="In all_atom mode, full_graph also decodes H-X bonds; heavy_projection uses H coordinates "
                        "only as encoder context and decodes/scorers the heavy graph. The latter is diagnostic only.")
    p.add_argument("--heavy_latent_charge_h", action="store_true",
                   help="Do not use atom charge/implicit-H logits in MILP scoring.  Charge/H remain only "
                        "feasibility variables; automatically enabled for heavy_only input.")
    p.add_argument("--use_heavy_atom_state_logits", action="store_true",
                   help="Use a heavy-only checkpoint's supervised formal-charge and implicit-H logits "
                        "in MILP scoring. Intended for joint-state checkpoints; default behaviour is "
                        "unchanged and treats these states as latent.")
    p.add_argument("--methods", nargs="+", default=["bondnet", "rdkit", "openbabel", "xyz2mol"],
                   choices=["bondnet", "rdkit", "openbabel", "xyz2mol"])
    p.add_argument("--stage1_ckpt")
    p.add_argument("--refiner_ckpt")
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+", default=None,
                   help="Optional supported-element filter; skip molecules containing any other element.")
    p.add_argument("--candidate_radius", type=float, default=2.5)
    p.add_argument("--candidate_knn", type=int, default=0,
                   help="Optional heavy-atom kNN augmentation; 0 keeps the radius-only protocol.")
    p.add_argument("--max_neighbors", type=int, default=15)
    p.add_argument("--h_candidate_cutoff", type=float, default=1.4,
                   help="Maximum H--heavy distance for explicit-H candidate edges in all_atom mode.")
    p.add_argument("--single_stage", action="store_true",
                   help="Evaluate a single-stage checkpoint: full candidate message graph, "
                        "Stage-1 conn feature zeroed. Must match how the checkpoint was trained.")
    p.add_argument("--restrict_delocalized", action="store_true",
                   help="Forbid class 5 on edges that are not (C|N)--O, matching the necessary "
                        "element condition of the training label rule. Inference-only; no "
                        "retraining and no change to the scoring metric.")
    p.add_argument("--aromatic_constraints", action="store_true")
    p.add_argument("--aromatic_bridge_repair", action="store_true",
                   help="Demote bridge edges of the selected aromatic subgraph after MILP decoding.")
    p.add_argument("--state_catalogue_json", default=None,
                   help="Optional training-derived decoder state catalogue JSON. "
                        "When omitted, use the historical built-in catalogue.")
    p.add_argument("--report_confusion", action="store_true",
                   help="Record true->predicted labels on reference heavy-heavy edges for diagnostics.")
    p.add_argument("--per_edge_out", default=None,
                   help="JSONL of one row per candidate/true edge with class probabilities and "
                        "reference context. Use this to localise error modes instead of "
                        "inferring them from aggregate confusion counts.")
    p.add_argument("--per_molecule_out", default=None,
                   help="Optional JSON of per-molecule strict/resonance outcomes. Required for "
                        "paired significance tests (McNemar) between two runs, which aggregate "
                        "rates alone cannot support.")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def bond_label(bond):
    from rdkit import Chem
    if bond.GetIsAromatic():
        return 4
    return {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)


def true_graph(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): bond_label(b) for b in mol.GetBonds()}


def motif_bins(mol):
    aromatic = any(b.GetIsAromatic() for b in mol.GetBonds())
    conjugated = any(b.GetIsConjugated() and not b.GetIsAromatic() for b in mol.GetBonds())
    hyper_s = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16:
            continue
        order_sum = sum(1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        if order_sum >= 5.0:
            hyper_s = True
            break
    labels = ["all"]
    if aromatic:
        labels.append("aromatic")
    if conjugated:
        labels.append("conjugated_non_aromatic")
    if hyper_s:
        labels.append("hypervalent_sulfur")
    return labels


def coordinate_view(mol, noise, seed):
    import numpy as np

    elems = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    if noise:
        generator = torch.Generator().manual_seed(seed)
        coord = coord + torch.randn(coord.shape, generator=generator) * float(noise)
    return elems, coord


def raw_rdkit_mol(elems, coord):
    from rdkit import Chem
    rw = Chem.RWMol()
    for z in elems.tolist():
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(elems))
    for index, xyz in enumerate(coord.tolist()):
        conf.SetAtomPosition(index, xyz)
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol()


def rdkit_predict(elems, coord, total_charge):
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    mol = raw_rdkit_mol(elems, coord)
    DetermineBonds(mol, charge=int(total_charge), maxIterations=1000)
    return true_graph(mol)


def openbabel_predict(elems, coord, total_charge):
    from openbabel import openbabel as ob
    obmol = ob.OBMol()
    for z, xyz in zip(elems.tolist(), coord.tolist()):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(int(z))
        atom.SetVector(*map(float, xyz))
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    out = {}
    for bond in ob.OBMolBondIter(obmol):
        pair = tuple(sorted((int(bond.GetBeginAtomIdx()) - 1, int(bond.GetEndAtomIdx()) - 1)))
        out[pair] = 4 if bond.IsAromatic() else {1: 1, 2: 2, 3: 3}.get(int(bond.GetBondOrder()), 1)
    return out


def heavy_graph(mol):
    """Reference heavy-heavy graph, regardless of whether input includes H."""
    return {pair: label for pair, label in true_graph(mol).items()
            if mol.GetAtomWithIdx(pair[0]).GetAtomicNum() != 1
            and mol.GetAtomWithIdx(pair[1]).GetAtomicNum() != 1}


def resonance_align_six(predicted, truth, mol):
    """Make strict scoring fair to BondNet's explicit class-5 representation.

    The six-class model may assign both terminal C/N--O bonds in a neutral
    carboxyl/nitro group as ``delocalized`` (class 5), whereas CCD must store
    one conventional single and one double bond.  This scoring-only transform
    accepts the two representations as equivalent.  It never repairs missing
    edges or non-resonance errors, and strict metrics remain reported beside it.
    """
    aligned = dict(predicted)
    adjacency = defaultdict(set)
    for i, j in truth:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for center, neighbours in adjacency.items():
        center_z = mol.GetAtomWithIdx(center).GetAtomicNum()
        if center_z not in {6, 7}:
            continue
        terminal_oxygen = [n for n in neighbours
                            if mol.GetAtomWithIdx(n).GetAtomicNum() == 8
                            and len(adjacency[n]) == 1]
        if len(terminal_oxygen) != 2:
            continue
        edges = [tuple(sorted((center, oxygen))) for oxygen in terminal_oxygen]
        true_labels = [truth.get(edge, 0) for edge in edges]
        pred_labels = [aligned.get(edge, 0) for edge in edges]
        # A valid conventional or delocalized representation has bond-order
        # multiset {single, double}; class 5 denotes the symmetric alternative.
        conventional_ok = sorted(pred_labels) == [1, 2]
        delocalized_ok = pred_labels == [5, 5]
        if sorted(true_labels) == [1, 2] and (conventional_ok or delocalized_ok):
            for edge in edges:
                aligned[edge] = truth[edge]
    return aligned


def resonance_align_terminal_oxy(predicted, truth, mol, elements=(6, 7, 15, 16)):
    """Permutation equivalence over chemically indistinguishable terminal oxygens.

    Generalises ``resonance_align_six`` from C/N to any centre in ``elements``.
    The justification is measured, not conventional.  Terminal X--O bond lengths
    for double vs. single, expressed as Cohen's d:

        group                  CCD ideal    PDBBind pose
        aromatic ring C:C         0.46          0.12
        carboxyl  C--O            9.14          0.01
        phosphate P--O           13.12          0.08

    In real poses the two populations sit on top of each other, so *which*
    terminal oxygen carries the double bond is not recoverable from (Z, R, Q).
    CCD's ideal coordinates do impose a localised geometry on carboxyl and
    phosphate (d = 9.14 / 13.12) -- note they do *not* do this for aromatic
    rings, which are already delocalised at d = 0.46.  That inconsistency, not
    the model, is what the strict metric is measuring on these groups.

    Conservative by construction:

    * Only the *assignment* of orders to equivalent oxygens is relaxed.  The
      bond-order multiset must match the reference exactly, so predicting
      {1,1,1} where the reference is {1,1,2} stays an error -- that removes
      bond order rather than permuting it.
    * The all-delocalised representation (every edge class 5) is accepted, as
      that is the label the training rule assigns to the same motif.
    * Nothing else is rewritten; no missing edge is ever added.
    * A centre needs >= 2 terminal oxygens; with fewer there is no ambiguity.

    With ``elements=(6, 7)`` this reproduces ``resonance_align_six`` on every
    motif present in the PDBBind set, so the C/N and P/S contributions can be
    reported separately.
    """
    aligned = dict(predicted)
    adjacency = defaultdict(set)
    for i, j in truth:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for center, neighbours in adjacency.items():
        if mol.GetAtomWithIdx(center).GetAtomicNum() not in set(elements):
            continue
        terminal_oxygen = [n for n in neighbours
                           if mol.GetAtomWithIdx(n).GetAtomicNum() == 8
                           and len(adjacency[n]) == 1]
        if len(terminal_oxygen) < 2:
            continue
        edges = [tuple(sorted((center, oxygen))) for oxygen in terminal_oxygen]
        true_labels = sorted(truth.get(edge, 0) for edge in edges)
        pred_labels = [aligned.get(edge, 0) for edge in edges]
        # A permutation preserves the total bond order by construction.
        permutation_ok = sorted(pred_labels) == true_labels
        # The all-delocalized form is only accepted for the motif the training
        # rule actually defines: exactly two terminal oxygens carrying {1,2}.
        # Accepting it more broadly would equate representations with different
        # total bond order -- e.g. predicting {5,5} against a reference {1,1}
        # is not a permutation, it invents half a bond order per edge.
        delocalized_ok = (len(edges) == 2 and true_labels == [1, 2]
                          and all(label == 5 for label in pred_labels))
        # Method-neutral fairness: different tools encode the SAME delocalized
        # terminal-oxy system differently -- BondNet uses class 5, Knodle marks
        # both bonds aromatic (mol2 ``ar`` / O.co2), CCD kekulizes to {1,2}.
        # Accepting an all-aromatic encoding on a >=2 terminal-oxy oxyacid centre
        # is the exact analogue of accepting all-class-5; refusing it would relax
        # the metric for one method's convention and not another's. Gated to a
        # genuinely delocalizable reference: the multiset must carry a double
        # bond, so all-aromatic on a geminal diol {1,1} stays an error.
        aromatic_deloc_ok = (all(label == 4 for label in pred_labels)
                             and 2 in true_labels)
        if permutation_ok or delocalized_ok or aromatic_deloc_ok:
            for edge in edges:
                aligned[edge] = truth[edge]
    return aligned


def aromatic_kekule_align_six(predicted, truth, mol):
    """Ring-level equivalence between a complete aromatic and Kekule ring.

    A reference ring encoded entirely as single/double bonds is accepted as
    equivalent only when the prediction labels *every edge of that ring*
    aromatic.  This intentionally rejects partial aromatic fragments and
    non-ring aromatic predictions, so it is a conservative representation
    normalization rather than a generic label rewrite.
    """
    aligned = dict(predicted)
    ring_info = mol.GetRingInfo()
    eligible_atoms = {6, 7, 8, 15, 16}
    for bond_indices in ring_info.BondRings():
        if len(bond_indices) < 5:
            continue
        bonds = [mol.GetBondWithIdx(index) for index in bond_indices]
        edges = [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in bonds]
        true_labels = [truth.get(edge, 0) for edge in edges]
        ring_atoms = {atom for edge in edges for atom in edge}
        if (not set(true_labels).issubset({1, 2}) or true_labels.count(2) < 2
                or not all(mol.GetAtomWithIdx(atom).GetAtomicNum() in eligible_atoms for atom in ring_atoms)):
            continue
        if all(aligned.get(edge, 0) == 4 for edge in edges):
            for edge in edges:
                aligned[edge] = truth[edge]
    return aligned


def load_vendored_xyz2mol():
    """Load the self-contained xyz2mol files without importing MolCraft.

    ``MolecularDiffusion.utils.__init__`` imports the full training stack,
    including optional PyG extensions unavailable in this Windows environment.
    xyz2mol itself needs only ``elements.py`` and ``converter.py``.  Creating
    their package namespaces here keeps this benchmark faithful to the vendored
    baseline without introducing unrelated optional dependencies.
    """
    global _XYZ2MOL
    if _XYZ2MOL is not None:
        return _XYZ2MOL
    xyz_dir = VENDORED_MOLCRAFT_SRC / "MolecularDiffusion" / "utils" / "xyz2mol"
    if not xyz_dir.exists():
        raise RuntimeError("vendored xyz2mol source is unavailable")

    package_paths = {
        "MolecularDiffusion": VENDORED_MOLCRAFT_SRC / "MolecularDiffusion",
        "MolecularDiffusion.utils": VENDORED_MOLCRAFT_SRC / "MolecularDiffusion" / "utils",
        "MolecularDiffusion.utils.xyz2mol": xyz_dir,
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    for stem in ("elements", "converter"):
        name = f"MolecularDiffusion.utils.xyz2mol.{stem}"
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, xyz_dir / f"{stem}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load vendored xyz2mol {stem}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    _XYZ2MOL = sys.modules["MolecularDiffusion.utils.xyz2mol.converter"].xyz2mol
    return _XYZ2MOL


def xyz2mol_predict(elems, coord, total_charge):
    """Vendored xyz2mol baseline with radius-derived adjacency.

    The original xyz2mol package is not installed in the workspace, but the
    MolCraftDiffusion dependency vendors its published conversion core.  This
    wrapper deliberately constructs its adjacency only from elements/geometry,
    then lets xyz2mol assign bond orders under the given total charge.
    """
    import numpy as np
    from rdkit import Chem

    xyz2mol = load_vendored_xyz2mol()

    atoms = [int(z) for z in elems.tolist()]
    xyz = np.asarray(coord.detach().cpu(), dtype=float)
    periodic = Chem.GetPeriodicTable()
    adjacency = np.zeros((len(atoms), len(atoms)), dtype=int)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            cutoff = 1.25 * (periodic.GetRcovalent(atoms[i]) + periodic.GetRcovalent(atoms[j]))
            if float(np.linalg.norm(xyz[i] - xyz[j])) <= cutoff:
                adjacency[i, j] = adjacency[j, i] = 1
    mols = xyz2mol(atoms, xyz, adjacency, 1.25, charge=int(total_charge),
                   allow_charged_fragments=True, use_graph=True,
                   use_huckel=False, embed_chiral=False)
    mol = mols[0] if isinstance(mols, (tuple, list)) else mols
    Chem.SanitizeMol(mol)
    return true_graph(mol)


def load_bondnet(args, device):
    from scripts.evaluate_structured_refiner import load_refiner
    from scripts.train_bond_refiner import load_stage1
    if not args.stage1_ckpt or not args.refiner_ckpt:
        raise ValueError("bondnet requires --stage1_ckpt and --refiner_ckpt")
    return load_stage1(args.stage1_ckpt, device), load_refiner(args.refiner_ckpt, device)


def load_state_catalogue(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("catalogue", payload)
    catalogue = {}
    for atomic_number, states in raw.items():
        catalogue[int(atomic_number)] = tuple(
            (int(state["charge"]), float(state["max_valence"])) for state in states
        )
    return catalogue


@torch.no_grad()
def bondnet_predict(elems, coord, total_charge, stage1, refiner, train_args, args, device,
                    return_edge_detail=False, n_solutions=None):
    from bondnet.inference.chemical_repair import joint_decode
    from bondnet.integration.molcraft import (
        all_atom_candidate_edges, build_refiner_inference_batch,
        forward_coordinate_refiner, make_coordinate_batch, stage1_hydrogen_logits,
    )
    from scripts.run_molcraft_bondnet import undirected_labels

    candidates = all_atom_candidate_edges(elems, coord, args.candidate_radius,
                                           args.candidate_knn, args.max_neighbors,
                                           args.h_candidate_cutoff)
    batch = make_coordinate_batch(elems.to(device), coord.to(device), total_charge,
                                  args.candidate_radius, args.candidate_knn, args.max_neighbors,
                                  candidate_edge_index=candidates.to(device))
    explicit_h = bool(torch.any(elems == 1))
    # The refiner was supervised on heavy-heavy labels.  In all-atom mode use
    # its heavy readout and splice in the trained Stage-1 H-X connectivity head.
    built = build_refiner_inference_batch(stage1, batch, train_args,
                                           refiner_heavy_only=explicit_h)
    logits, aux = forward_coordinate_refiner(refiner, batch, built, return_aux=True)
    if explicit_h and args.bondnet_decode_mode == "full_graph":
        all_logits = stage1_hydrogen_logits(built["all_conn_logits"], batch["edge_index"])
        all_logits[built["readout_mask"]] = logits
        logits = all_logits
        readout_ei = batch["edge_index"]
        decoder_elems = batch["elems"]
        decoder_charge_logits = aux.get("charge_logits")
        decoder_implicit_h_logits = aux.get("implicit_h_logits")
        decoder_explicit_h = True
    elif explicit_h:
        # Diagnostic: preserve observed H coordinates in PaiNN's message
        # passing, but recover only the heavy graph.  This separates a bad
        # H-X edge decoder from the heavy-bond classifier itself.
        heavy = torch.where(batch["elems"] != 1)[0]
        inverse = torch.full((batch["elems"].numel(),), -1, dtype=torch.long, device=device)
        inverse[heavy] = torch.arange(heavy.numel(), device=device)
        readout_ei = built["readout_ei"]
        decoder_elems = batch["elems"][heavy]
        decoder_charge_logits = aux.get("charge_logits")
        decoder_charge_logits = (decoder_charge_logits[heavy]
                                 if decoder_charge_logits is not None else None)
        decoder_implicit_h_logits = aux.get("implicit_h_logits")
        decoder_implicit_h_logits = (decoder_implicit_h_logits[heavy]
                                      if decoder_implicit_h_logits is not None else None)
        decode_edge_index = inverse[readout_ei]
        decoder_explicit_h = False
    else:
        readout_ei = built["readout_ei"]
        decode_edge_index = readout_ei
        decoder_elems = batch["elems"]
        decoder_charge_logits = aux.get("charge_logits")
        decoder_implicit_h_logits = aux.get("implicit_h_logits")
        decoder_explicit_h = False
    if explicit_h and args.bondnet_decode_mode == "full_graph":
        decode_edge_index = readout_ei
    # Without observed H, atom-wise charge and implicit-H states are generally
    # non-identifiable from (Z, R, Q).  In particular, the heavy-only refiner
    # intentionally has no direct charge/H supervision.  Do not let arbitrary
    # or untrained auxiliary logits choose one resonance/protonation state.
    latent_charge_h = bool(args.heavy_latent_charge_h
                           or (not explicit_h and not args.use_heavy_atom_state_logits))
    if latent_charge_h:
        decoder_charge_logits = None
        decoder_implicit_h_logits = None
    decoded = joint_decode(logits, decode_edge_index, decoder_elems, total_charge,
                           charge_logits=decoder_charge_logits,
                           implicit_h_logits=decoder_implicit_h_logits,
                           explicit_h=decoder_explicit_h, aromatic_constraints=args.aromatic_constraints,
                           aromatic_bridge_repair=args.aromatic_bridge_repair,
                           restrict_delocalized=args.restrict_delocalized,
                           state_catalogue=getattr(args, "state_catalogue", None))
    if n_solutions is not None:
        # Ranked candidate enumeration reuses this whole setup path rather than
        # rebuilding it; duplicating the Stage-1 bridge is how index mismatches
        # get introduced.
        from bondnet.inference.chemical_repair import decode_top_n

        solutions = decode_top_n(
            logits, decode_edge_index, decoder_elems, total_charge,
            n_solutions=n_solutions,
            charge_logits=decoder_charge_logits,
            implicit_h_logits=decoder_implicit_h_logits,
            explicit_h=decoder_explicit_h,
            aromatic_constraints=args.aromatic_constraints,
            aromatic_bridge_repair=args.aromatic_bridge_repair,
            restrict_delocalized=args.restrict_delocalized,
            state_catalogue=getattr(args, "state_catalogue", None),
            require_kekulizable=getattr(args, "require_kekulizable_candidates", False),
        )
        # Return the decoder's own charge / implicit-H assignment alongside the
        # bonds. Without it a candidate cannot be canonicalised into a species,
        # and the caller is left comparing notations instead of molecules.
        from bondnet.inference.chemical_repair import _undirected_scores

        _, _, unique, _ = _undirected_scores(logits, decode_edge_index)
        out = []
        for result, per_edge in solutions:
            pairs, labels = undirected_labels(readout_ei, result.labels)
            out.append({
                "bonds": {pair: label for pair, label in zip(pairs, labels) if label > 0},
                "edges": list(unique),
                "edge_labels": per_edge,
                "formal_charge": result.formal_charge.detach().cpu().numpy(),
                "implicit_h": result.implicit_h_count.detach().cpu().numpy(),
                "elems": decoder_elems.detach().cpu(),
                "objective": float(result.objective),
            })
        return out
    if not decoded.solved:
        return ({}, {}) if return_edge_detail else {}
    pairs, labels = undirected_labels(readout_ei, decoded.labels)
    predicted = {pair: label for pair, label in zip(pairs, labels) if label > 0}
    if not return_edge_detail:
        return predicted
    # Mean class probability per undirected candidate pair, matching how the
    # decoder scores them. Kept separate from `predicted` so no existing caller
    # changes behaviour.
    probs = torch.softmax(logits.detach().float(), dim=-1).cpu()
    grouped = {}
    for row, (i, j) in enumerate(readout_ei.detach().cpu().tolist()):
        grouped.setdefault(tuple(sorted((int(i), int(j)))), []).append(row)
    detail = {pair: probs[rows].mean(dim=0).tolist() for pair, rows in grouped.items()}
    return predicted, detail


def edge_records(mol, truth, predicted, edge_detail, sigma, source_index):
    """One row per edge that is either a true bond or a prediction.

    Written so error modes can be localised directly instead of inferred from
    aggregate confusion counts. Two earlier attempts to guess where the class-5
    errors lived from molecule-level statistics were wrong, which is what this
    dump exists to prevent.

    Context fields describe the *reference* environment (they are scoring-side
    metadata, never fed to the model): the C/N centre's terminal-oxygen count
    and bond-order sum are exactly the conditions that define the class-5
    training label, so rows can be split by whether the trained pattern holds.
    """
    from rdkit import Chem

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(source_index)
    degree = {atom.GetIdx(): atom.GetDegree() for atom in mol.GetAtoms()}
    ring_info = mol.GetRingInfo()
    order_of = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}

    def centre_context(i, j):
        zi = mol.GetAtomWithIdx(i).GetAtomicNum()
        zj = mol.GetAtomWithIdx(j).GetAtomicNum()
        if zi in (6, 7) and zj == 8:
            centre = i
        elif zj in (6, 7) and zi == 8:
            centre = j
        else:
            return None, None, None
        terminal = [nb for nb in mol.GetAtomWithIdx(centre).GetNeighbors()
                    if nb.GetAtomicNum() == 8 and degree[nb.GetIdx()] == 1]
        orders = []
        for nb in terminal:
            bond = mol.GetBondBetweenAtoms(centre, nb.GetIdx())
            orders.append(order_of.get(bond.GetBondType(), 0))
        return centre, len(terminal), sum(orders)

    rows = []
    for pair in set(truth) | set(predicted):
        i, j = pair
        true_label = truth.get(pair, 0)
        pred_label = predicted.get(pair, 0)
        centre, n_terminal_o, terminal_order_sum = centre_context(i, j)
        bond = mol.GetBondBetweenAtoms(i, j)
        rows.append({
            "name": name,
            "sigma": float(sigma),
            "i": int(i), "j": int(j),
            "z_i": int(mol.GetAtomWithIdx(i).GetAtomicNum()),
            "z_j": int(mol.GetAtomWithIdx(j).GetAtomicNum()),
            "true": int(true_label),
            "pred": int(pred_label),
            "correct": int(true_label == pred_label),
            "in_ring": bool(bond.IsInRing()) if bond is not None else False,
            "ring_size_min": (min((size for size in range(3, 9)
                                   if ring_info.IsBondInRingOfSize(bond.GetIdx(), size)),
                                  default=0) if bond is not None else 0),
            "oxy_centre": (int(centre) if centre is not None else None),
            "n_terminal_o": n_terminal_o,
            "terminal_order_sum": terminal_order_sum,
            "matches_class5_training_pattern": bool(
                n_terminal_o == 2 and terminal_order_sum == 3),
            "probs": [round(p, 5) for p in edge_detail.get(pair, [])],
        })
    return rows


def empty_cell():
    return defaultdict(float)


def update(cell, truth, predicted):
    cell["n"] += 1
    true_ok = all(predicted.get(pair, 0) == label for pair, label in truth.items())
    strict_ok = all(predicted.get(pair, 0) == truth.get(pair, 0) for pair in set(truth) | set(predicted))
    cell["true_bond_exact"] += int(true_ok)
    cell["strict_graph_exact"] += int(strict_ok)
    cell["true_edges"] += len(truth)
    cell["typed_true_edges"] += sum(predicted.get(pair, 0) == label for pair, label in truth.items())
    cell["false_positive_edges"] += sum(pair not in truth for pair in predicted)


def summarize(cell):
    n = int(cell["n"])
    return {
        "n": n,
        "true_bond_exact_rate": cell["true_bond_exact"] / n if n else None,
        "strict_graph_exact_rate": cell["strict_graph_exact"] / n if n else None,
        "true_edge_typed_recall": cell["typed_true_edges"] / cell["true_edges"] if cell["true_edges"] else None,
        "mean_false_positive_edges": cell["false_positive_edges"] / n if n else None,
    }


def main():
    from rdkit import Chem

    args = parse_args()
    args.state_catalogue = load_state_catalogue(args.state_catalogue_json)
    if (args.input_mode == "heavy_only" and "rdkit" in args.methods
            and not args.diagnostic_allow_rdkit_heavy):
        raise ValueError(
            "RDKit DetermineBonds requires a complete atom inventory and is not a valid "
            "heavy-only baseline. Use xyz2mol (with a declared time budget) or run RDKit "
            "only under --input_mode all_atom."
        )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    stage1 = refiner = train_args = None
    if "bondnet" in args.methods:
        stage1, loaded = load_bondnet(args, device)
        refiner, train_args = loaded
        # The inference bridge reads single_stage off train_args (the checkpoint's
        # saved training config), so a single-stage checkpoint is handled
        # automatically. The explicit --single_stage flag can also force it on,
        # and we warn if the two disagree (usually a wrong --refiner_ckpt).
        ckpt_single = bool(getattr(train_args, "single_stage", False))
        if args.single_stage and not ckpt_single:
            print("warning: --single_stage set but the checkpoint was trained two-stage; "
                  "forcing single-stage inference (train/test mismatch, results not comparable)")
        elif ckpt_single and not args.single_stage:
            print("note: checkpoint was trained single-stage; using single-stage inference")
        train_args.single_stage = ckpt_single or args.single_stage
    result = {str(sigma): {method: defaultdict(empty_cell) for method in args.methods} for sigma in args.noise}
    resonance_result = {str(sigma): {method: defaultdict(empty_cell) for method in args.methods} for sigma in args.noise}
    aromatic_kekule_result = {str(sigma): {method: defaultdict(empty_cell) for method in args.methods} for sigma in args.noise}
    per_molecule = {str(sigma): {method: [] for method in args.methods} for sigma in args.noise}
    per_edge = []
    # Additive tiers: existing outputs are untouched, the extension is reported
    # beside them so the C/N and P/S contributions stay separable.
    resonance_cn_result = {str(sigma): {method: defaultdict(empty_cell) for method in args.methods} for sigma in args.noise}
    resonance_ps_result = {str(sigma): {method: defaultdict(empty_cell) for method in args.methods} for sigma in args.noise}
    failures = {str(sigma): {method: Counter() for method in args.methods} for sigma in args.noise}
    confusion = {str(sigma): {method: Counter() for method in args.methods} for sigma in args.noise}
    supplier = Chem.SDMolSupplier(args.reference_sdf, removeHs=False)
    accepted = 0
    skipped_unsupported_elements = 0
    for source_index, mol in enumerate(supplier):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        # ``heavy_only`` deliberately tests covariate shift.  ``all_atom`` is
        # the fair primary protocol for the explicit-H trained checkpoint.
        if args.input_mode == "heavy_only" and any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            mol = Chem.RemoveHs(mol, sanitize=True)
        if args.allowed_atomic_nums is not None:
            observed = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
            if not observed.issubset(set(args.allowed_atomic_nums)):
                skipped_unsupported_elements += 1
                continue
        truth = heavy_graph(mol)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        bins = motif_bins(mol)
        for sigma in args.noise:
            elems, coord = coordinate_view(mol, sigma, args.seed + source_index * 1009 + round(sigma * 1_000_000))
            for method in args.methods:
                try:
                    if method == "rdkit":
                        predicted = rdkit_predict(elems, coord, total_charge)
                    elif method == "openbabel":
                        predicted = openbabel_predict(elems, coord, total_charge)
                    elif method == "xyz2mol":
                        predicted = xyz2mol_predict(elems, coord, total_charge)
                    else:
                        want_detail = bool(args.per_edge_out) and method == "bondnet"
                        # NB: do not name this `result` -- that is the metrics
                        # accumulator defined in main() and shadowing it breaks
                        # `result[str(sigma)][method][motif]` further down.
                        bondnet_out = bondnet_predict(elems, coord, total_charge, stage1, refiner,
                                                      train_args, args, device,
                                                      return_edge_detail=want_detail)
                        predicted, edge_detail = bondnet_out if want_detail else (bondnet_out, {})
                    predicted = {pair: label for pair, label in predicted.items()
                                 if int(elems[pair[0]]) != 1 and int(elems[pair[1]]) != 1}
                except Exception as exc:
                    message = str(exc).replace("\n", " ")[:160]
                    failures[str(sigma)][method][f"{type(exc).__name__}: {message}"] += 1
                    predicted = {}
                    edge_detail = {}
                if args.per_edge_out and method == "bondnet":
                    per_edge.extend(edge_records(mol, truth, predicted, edge_detail, sigma,
                                                 source_index))
                resonance_predicted = resonance_align_six(predicted, truth, mol)
                resonance_cn = resonance_align_terminal_oxy(predicted, truth, mol, elements=(6, 7))
                resonance_ps = resonance_align_terminal_oxy(predicted, truth, mol,
                                                            elements=(6, 7, 15, 16))
                aromatic_kekule_predicted = aromatic_kekule_align_six(resonance_predicted, truth, mol)
                if args.report_confusion:
                    for pair, label in truth.items():
                        pred_label = predicted.get(pair, 0)
                        confusion[str(sigma)][method][f"{label}->{pred_label}"] += 1
                        bond = mol.GetBondBetweenAtoms(*pair)
                        if label in {1, 2} and pred_label == 4:
                            key = "kekule_to_aromatic_in_ring" if bond is not None and bond.IsInRing() else "kekule_to_aromatic_not_in_ring"
                            confusion[str(sigma)][method][key] += 1
                if args.per_molecule_out:
                    strict_ok = all(predicted.get(pair, 0) == truth.get(pair, 0)
                                    for pair in set(truth) | set(predicted))
                    resonance_ok = all(resonance_predicted.get(pair, 0) == truth.get(pair, 0)
                                       for pair in set(truth) | set(resonance_predicted))
                    per_molecule[str(sigma)][method].append({
                        "name": mol.GetProp("_Name") if mol.HasProp("_Name") else str(source_index),
                        "n_heavy": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1),
                        "strict": int(strict_ok),
                        "resonance": int(resonance_ok),
                    })
                for motif in bins:
                    update(resonance_cn_result[str(sigma)][method][motif], truth, resonance_cn)
                    update(resonance_ps_result[str(sigma)][method][motif], truth, resonance_ps)
                    update(result[str(sigma)][method][motif], truth, predicted)
                    update(resonance_result[str(sigma)][method][motif], truth, resonance_predicted)
                    update(aromatic_kekule_result[str(sigma)][method][motif], truth, aromatic_kekule_predicted)
        accepted += 1
        if args.max_mols is not None and accepted >= args.max_mols:
            break
        if accepted % 100 == 0:
            print(f"evaluated {accepted}", flush=True)
    report = {
        "task": "template-free heavy-atom graph recovery from (Z, R, Q); reference bonds are scoring-only",
        "input_mode": args.input_mode,
        "diagnostic_allow_rdkit_heavy": args.diagnostic_allow_rdkit_heavy,
        "bondnet_decode_mode": args.bondnet_decode_mode,
        "heavy_latent_charge_h": bool(
            args.heavy_latent_charge_h
            or (args.input_mode == "heavy_only" and not args.use_heavy_atom_state_logits)
        ),
        "use_heavy_atom_state_logits": bool(args.use_heavy_atom_state_logits),
        "state_catalogue_json": (str(Path(args.state_catalogue_json).resolve())
                                 if args.state_catalogue_json else None),
        "scoring": "heavy-heavy bonds only; all_atom input may use observed H coordinates but never reference bonds",
        "reference_sdf": str(Path(args.reference_sdf).resolve()), "n_molecules": accepted,
        "allowed_atomic_nums": args.allowed_atomic_nums,
        "skipped_unsupported_elements": skipped_unsupported_elements,
        "methods": args.methods, "noise": args.noise,
        "motif_definition": {
            "aromatic": "at least one aromatic reference bond",
            "conjugated_non_aromatic": "at least one non-aromatic RDKit-conjugated reference bond",
            "hypervalent_sulfur": "a sulfur atom with reference bond-order sum >= 5",
        },
        "results": {sigma: {method: {motif: summarize(cell) for motif, cell in motifs.items()}
                              for method, motifs in methods.items()}
                    for sigma, methods in result.items()},
        "resonance_aware_results": {sigma: {method: {motif: summarize(cell) for motif, cell in motifs.items()}
                                              for method, motifs in methods.items()}
                                    for sigma, methods in resonance_result.items()},
        "aromatic_kekule_equiv_results": {sigma: {method: {motif: summarize(cell) for motif, cell in motifs.items()}
                                                    for method, motifs in methods.items()}
                                          for sigma, methods in aromatic_kekule_result.items()},
        "resonance_terminal_oxy_CN_results": {sigma: {method: {motif: summarize(cell) for motif, cell in motifs.items()}
                                                        for method, motifs in methods.items()}
                                              for sigma, methods in resonance_cn_result.items()},
        "resonance_terminal_oxy_CNPS_results": {sigma: {method: {motif: summarize(cell) for motif, cell in motifs.items()}
                                                          for method, motifs in methods.items()}
                                                for sigma, methods in resonance_ps_result.items()},
        "resonance_tier_note": (
            "CN reproduces resonance_align_six; CNPS extends the same permutation "
            "equivalence to phosphorus and sulfur centres. Justified by measured "
            "indistinguishability of terminal X-O bond lengths in real poses "
            "(Cohen d 0.01 carboxyl, 0.08 phosphate) versus CCD ideal coordinates "
            "(9.14 and 13.12). Report beside strict, never instead of it."),
        "method_exceptions": {sigma: {method: dict(counts) for method, counts in methods.items()}
                              for sigma, methods in failures.items()},
        "true_edge_label_confusion": ({sigma: {method: dict(counts) for method, counts in methods.items()}
                                        for sigma, methods in confusion.items()}
                                      if args.report_confusion else None),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.per_edge_out:
        per_edge_path = Path(args.per_edge_out)
        per_edge_path.parent.mkdir(parents=True, exist_ok=True)
        with per_edge_path.open("w", encoding="utf-8") as handle:
            for row in per_edge:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {len(per_edge)} per-edge rows to {per_edge_path}")
    if args.per_molecule_out:
        per_molecule_path = Path(args.per_molecule_out)
        per_molecule_path.parent.mkdir(parents=True, exist_ok=True)
        per_molecule_path.write_text(json.dumps(
            {"reference_sdf": str(Path(args.reference_sdf).resolve()),
             "n_molecules": accepted,
             "per_molecule": per_molecule}, indent=1), encoding="utf-8")
        print(f"wrote per-molecule outcomes to {per_molecule_path}")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
