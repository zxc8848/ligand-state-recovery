from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


@dataclass(frozen=True)
class GraphMetrics:
    l1_connectivity: bool
    l2_strict: bool
    l2_aromatic: bool
    l2_isomorphic: bool
    l3_full: bool
    l3_isomorphic: bool


def heavy_atom_copy(reference: Chem.Mol) -> Chem.Mol:
    """Return an atom-only molecule preserving heavy-atom order and coordinates."""
    if reference.GetNumConformers() != 1:
        raise ValueError("expected exactly one conformer")

    heavy_indices = [a.GetIdx() for a in reference.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy_indices:
        raise ValueError("molecule has no heavy atoms")

    rw = Chem.RWMol()
    for idx in heavy_indices:
        atom = reference.GetAtomWithIdx(idx)
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))

    source_conf = reference.GetConformer()
    conf = Chem.Conformer(len(heavy_indices))
    conf.Set3D(True)
    for new_idx, old_idx in enumerate(heavy_indices):
        conf.SetAtomPosition(new_idx, source_conf.GetAtomPosition(old_idx))
    rw.AddConformer(conf)
    return rw.GetMol()


def heavy_reference(reference: Chem.Mol) -> Chem.Mol:
    """Remove explicit hydrogens while retaining per-atom total-H information."""
    return Chem.RemoveHs(reference, sanitize=True)


def determine_rdkit(atom_cloud: Chem.Mol, charge: int) -> Chem.Mol:
    prediction = Chem.RWMol(atom_cloud)
    rdDetermineBonds.DetermineBonds(prediction, charge=int(charge), embedChiral=False)
    Chem.SanitizeMol(prediction)
    return prediction.GetMol()


def determine_rdkit_connectivity(atom_cloud: Chem.Mol, use_vdw: bool) -> Chem.Mol:
    prediction = Chem.RWMol(atom_cloud)
    rdDetermineBonds.DetermineConnectivity(prediction, useVdw=use_vdw, covFactor=1.3)
    return prediction.GetMol()


def _adjacency_matrix(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    matrix = np.zeros((n, n), dtype=np.bool_)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        matrix[i, j] = matrix[j, i] = True
    return matrix


def compare_connectivity(reference: Chem.Mol, prediction: Chem.Mol) -> bool:
    if reference.GetNumAtoms() != prediction.GetNumAtoms():
        return False
    if [a.GetAtomicNum() for a in reference.GetAtoms()] != [
        a.GetAtomicNum() for a in prediction.GetAtoms()
    ]:
        return False
    return bool(np.array_equal(_adjacency_matrix(reference), _adjacency_matrix(prediction)))


def _bond_matrix(mol: Chem.Mol, mode: str) -> np.ndarray:
    work = Chem.Mol(mol)
    if mode == "strict":
        Chem.Kekulize(work, clearAromaticFlags=True)
    elif mode == "aromatic":
        Chem.SanitizeMol(work)
        Chem.SetAromaticity(work)
    else:
        raise ValueError(f"unknown bond mode: {mode}")

    n = work.GetNumAtoms()
    matrix = np.zeros((n, n), dtype=np.int8)
    for bond in work.GetBonds():
        if mode == "aromatic" and bond.GetIsAromatic():
            value = 4
        else:
            value = int(round(bond.GetBondTypeAsDouble()))
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        matrix[i, j] = matrix[j, i] = value
    return matrix


def atom_states(mol: Chem.Mol) -> tuple[tuple[int, int], ...]:
    return tuple(
        (atom.GetFormalCharge(), int(atom.GetTotalNumHs(includeNeighbors=True)))
        for atom in mol.GetAtoms()
    )


def _isomorphic(
    reference: Chem.Mol, prediction: Chem.Mol, *, include_atom_state: bool
) -> bool:
    def build(mol: Chem.Mol) -> nx.Graph:
        matrix = _bond_matrix(mol, "aromatic")
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            state = (
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetTotalNumHs(includeNeighbors=True)),
            )
            graph.add_node(
                atom.GetIdx(),
                label=state if include_atom_state else atom.GetAtomicNum(),
            )
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                if matrix[i, j]:
                    graph.add_edge(i, j, order=int(matrix[i, j]))
        return graph

    return bool(
        nx.is_isomorphic(
            build(reference),
            build(prediction),
            node_match=lambda left, right: left["label"] == right["label"],
            edge_match=lambda left, right: left["order"] == right["order"],
        )
    )


def compare_graphs(reference: Chem.Mol, prediction: Chem.Mol) -> GraphMetrics:
    if reference.GetNumAtoms() != prediction.GetNumAtoms():
        return GraphMetrics(False, False, False, False, False, False)
    ref_z = [a.GetAtomicNum() for a in reference.GetAtoms()]
    pred_z = [a.GetAtomicNum() for a in prediction.GetAtoms()]
    if ref_z != pred_z:
        return GraphMetrics(False, False, False, False, False, False)

    l1 = compare_connectivity(reference, prediction)
    try:
        ref_ar = _bond_matrix(reference, "aromatic")
        pred_ar = _bond_matrix(prediction, "aromatic")
        l2_ar = np.array_equal(ref_ar, pred_ar)
    except (Chem.KekulizeException, Chem.AtomValenceException, ValueError):
        l2_ar = False
    try:
        l2_iso = _isomorphic(reference, prediction, include_atom_state=False)
    except (Chem.KekulizeException, Chem.AtomValenceException, ValueError):
        l2_iso = False
    try:
        l2_strict = np.array_equal(
            _bond_matrix(reference, "strict"), _bond_matrix(prediction, "strict")
        )
    except Chem.KekulizeException:
        l2_strict = False
    l3 = l2_ar and atom_states(reference) == atom_states(prediction)
    try:
        l3_iso = _isomorphic(reference, prediction, include_atom_state=True)
    except (Chem.KekulizeException, Chem.AtomValenceException, ValueError):
        l3_iso = False
    return GraphMetrics(l1, l2_strict, l2_ar, l2_iso, l3, l3_iso)


def bond_order_agreement_counts(
    reference: Chem.Mol, prediction: Chem.Mol
) -> tuple[int, int]:
    """Correct and attempted labels over the union of aromatic-normalized edges."""
    fallback_total = max(reference.GetNumBonds(), prediction.GetNumBonds())
    if reference.GetNumAtoms() != prediction.GetNumAtoms():
        return 0, fallback_total
    if [a.GetAtomicNum() for a in reference.GetAtoms()] != [
        a.GetAtomicNum() for a in prediction.GetAtoms()
    ]:
        return 0, fallback_total
    try:
        ref = _bond_matrix(reference, "aromatic")
        pred = _bond_matrix(prediction, "aromatic")
    except (Chem.KekulizeException, Chem.AtomValenceException, ValueError):
        return 0, fallback_total
    upper = np.triu(np.logical_or(ref != 0, pred != 0), k=1)
    total = int(upper.sum())
    correct = int(np.logical_and(upper, ref == pred).sum())
    return correct, total


def successful_charge_candidates(
    atom_cloud: Chem.Mol, charges: Iterable[int]
) -> dict[int, Chem.Mol]:
    candidates: dict[int, Chem.Mol] = {}
    for charge in charges:
        try:
            candidates[int(charge)] = determine_rdkit(atom_cloud, int(charge))
        except Exception:
            continue
    return candidates
