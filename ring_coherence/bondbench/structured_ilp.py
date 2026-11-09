"""Phase-3 structured extension of the frozen hybridization-first ILP.

This module leaves :mod:`bondbench.hybrid_ilp` unchanged.  It adds two kinds of
structure-level constraints that cannot be represented by independent edge costs:

* coherent Kekule patterns for planar five- and six-membered rings;
* conservative topology/geometry rules for unambiguous functional groups.

The model must be selected on development data only.  The already unblinded 2024
RCSB external set is not read by this module or its development evaluator.
"""

from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from bondbench.graphs import determine_rdkit_connectivity
from bondbench.hybrid_ilp import (
    HybridizationILPModel,
    _neighbours,
    atom_geometry_feature,
    bond_torsion_feature,
    ring_category,
)
from bondbench.ilp import AtomState, ILPResult


def _ring_edges(ring: tuple[int, ...]) -> list[tuple[int, int]]:
    return [tuple(sorted((ring[index], ring[(index + 1) % len(ring)]))) for index in range(len(ring))]


def _ring_geometry(pos: np.ndarray, ring: tuple[int, ...]) -> tuple[float, float, float]:
    points = pos[np.asarray(ring, dtype=int)]
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    distances = np.abs(centered @ vh[-1])
    lengths = [float(np.linalg.norm(pos[i] - pos[j])) for i, j in _ring_edges(ring)]
    return float(np.sqrt(np.mean(distances * distances))), float(np.mean(lengths)), float(np.std(lengths))


class StructuredHybridILPModel(HybridizationILPModel):
    """Hybrid ILP with ring-coherence variables and functional-group constraints."""

    def __init__(
        self,
        *args,
        ring_reward: float = 1.0,
        ring_planarity_rms: float = 0.15,
        ring_length_std: float = 0.10,
        functional_group_constraints: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ring_reward = float(ring_reward)
        self.ring_planarity_rms = float(ring_planarity_rms)
        self.ring_length_std = float(ring_length_std)
        self.functional_group_constraints = bool(functional_group_constraints)

    def _candidate_rings(self, connectivity: Chem.Mol, pos: np.ndarray) -> list[tuple[int, ...]]:
        allowed = {6, 7, 8, 15, 16}
        rings: list[tuple[int, ...]] = []
        for raw in Chem.GetSymmSSSR(connectivity):
            ring = tuple(int(value) for value in raw)
            if len(ring) not in (5, 6):
                continue
            if any(connectivity.GetAtomWithIdx(index).GetAtomicNum() not in allowed for index in ring):
                continue
            rms, mean_length, length_std = _ring_geometry(pos, ring)
            if rms <= self.ring_planarity_rms and 1.25 <= mean_length <= 1.50 and length_std <= self.ring_length_std:
                rings.append(ring)
        return rings

    def predict(self, atom_cloud: Chem.Mol, total_charge: int | None) -> ILPResult:
        connectivity = determine_rdkit_connectivity(atom_cloud, use_vdw=False)
        edges = sorted(
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            for b in connectivity.GetBonds()
        )
        if not edges:
            raise ValueError("connectivity inference returned no bonds")
        edge_set = set(edges)
        pos = atom_cloud.GetConformer().GetPositions()
        nb = _neighbours(atom_cloud.GetNumAtoms(), edges)
        Chem.FastFindRings(connectivity)
        ring_info = connectivity.GetRingInfo()
        atom_ring_categories = ring_category(connectivity)

        costs: list[float] = []
        edge_vars: dict[tuple[int, int, int], int] = {}
        for i, j in edges:
            zi = atom_cloud.GetAtomWithIdx(i).GetAtomicNum()
            zj = atom_cloud.GetAtomWithIdx(j).GetAtomicNum()
            length = float(np.linalg.norm(pos[i] - pos[j]))
            torsion = bond_torsion_feature(pos, nb, i, j)
            in_ring = ring_info.NumBondRings(connectivity.GetBondBetweenAtoms(i, j).GetIdx()) > 0
            for order in (1, 2, 3):
                edge_vars[(i, j, order)] = len(costs)
                costs.append(self.edge_cost(zi, zj, order, length, torsion, in_ring))

        candidate_rings = [ring for ring in self._candidate_rings(connectivity, pos) if set(_ring_edges(ring)) <= edge_set]
        ring_vars: dict[int, int] = {}
        for ring_index, _ in enumerate(candidate_rings):
            ring_vars[ring_index] = len(costs)
            costs.append(-self.ring_reward)

        atom_vars: dict[tuple[int, AtomState], int] = {}
        cands: dict[int, list[AtomState]] = {}
        for atom in connectivity.GetAtoms():
            idx = atom.GetIdx()
            feature = atom_geometry_feature(pos, nb, idx)
            states_and_costs = self._states_and_costs(
                atom.GetAtomicNum(), len(nb[idx]), feature, total_charge is None, atom_ring_categories[idx]
            )
            cands[idx] = [state for state, _ in states_and_costs]
            for state, cost in states_and_costs:
                atom_vars[(idx, state)] = len(costs)
                costs.append(cost)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        def constraint(coefficients: dict[int, float], lo: float, hi: float | None = None) -> None:
            row = len(lower)
            for column, value in coefficients.items():
                rows.append(row)
                cols.append(column)
                vals.append(value)
            lower.append(float(lo))
            upper.append(float(lo if hi is None else hi))

        for i, j in edges:
            constraint({edge_vars[(i, j, order)]: 1.0 for order in (1, 2, 3)}, 1.0)
        for idx, states in cands.items():
            constraint({atom_vars[(idx, state)]: 1.0 for state in states}, 1.0)
            valence: dict[int, float] = {}
            for neighbour in nb[idx]:
                i, j = sorted((idx, neighbour))
                for order in (1, 2, 3):
                    valence[edge_vars[(i, j, order)]] = float(order)
            for state in states:
                valence[atom_vars[(idx, state)]] = -float(state.capacity)
            constraint(valence, 0.0)
        if total_charge is not None:
            constraint(
                {atom_vars[(idx, state)]: float(state.formal_charge) for idx, states in cands.items() for state in states},
                float(total_charge),
            )

        # A selected six-ring has an alternating perfect matching of three double
        # bonds.  A selected five-ring has two non-adjacent doubles and one donor
        # atom.  When the ring variable is off, its bond orders remain unrestricted.
        for ring_index, ring in enumerate(candidate_rings):
            selected = ring_vars[ring_index]
            ring_edges = _ring_edges(ring)
            target = 3 if len(ring) == 6 else 2
            doubles = {edge_vars[(*edge, 2)]: 1.0 for edge in ring_edges}
            constraint({**doubles, selected: -float(target)}, 0.0, math.inf)
            constraint({**doubles, selected: float(len(ring) - target)}, -math.inf, float(len(ring)))
            for edge in ring_edges:
                constraint({edge_vars[(*edge, 3)]: 1.0, selected: 1.0}, -math.inf, 1.0)
            for atom in ring:
                incident = [edge for edge in ring_edges if atom in edge]
                coefficients = {edge_vars[(*edge, 2)]: 1.0 for edge in incident}
                if len(ring) == 6:
                    constraint({**coefficients, selected: -1.0}, 0.0, math.inf)
                constraint({**coefficients, selected: 1.0}, -math.inf, 2.0)

        if self.functional_group_constraints:
            self._add_functional_group_constraints(connectivity, pos, nb, edge_vars, constraint)

        matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), len(costs))).tocsr()
        result = milp(
            c=np.asarray(costs),
            integrality=np.ones(len(costs), dtype=np.int8),
            bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 15.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            raise RuntimeError(f"MILP failed ({result.status}): {result.message}")

        rw = Chem.RWMol()
        chosen: dict[int, AtomState] = {}
        for atom in atom_cloud.GetAtoms():
            idx = atom.GetIdx()
            state = max(cands[idx], key=lambda candidate: result.x[atom_vars[(idx, candidate)]])
            chosen[idx] = state
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(state.formal_charge)
            new_atom.SetNoImplicit(True)
            new_atom.SetNumExplicitHs(state.hydrogens)
            rw.AddAtom(new_atom)
        for i, j in edges:
            order = max((1, 2, 3), key=lambda value: result.x[edge_vars[(i, j, value)]])
            rw.AddBond(i, j, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order])
        rw.AddConformer(Chem.Conformer(atom_cloud.GetConformer()))
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
        return ILPResult(
            molecule=molecule,
            objective=float(result.fun),
            solver_status=int(result.status),
            solver_message=str(result.message),
            selected_charge=sum(state.formal_charge for state in chosen.values()),
        )

    def _add_functional_group_constraints(self, connectivity, pos, neighbours, edge_vars, add) -> None:
        def var(i: int, j: int, order: int) -> int:
            a, b = sorted((i, j))
            return edge_vars[(a, b, order)]

        for atom in connectivity.GetAtoms():
            center = atom.GetIdx()
            z = atom.GetAtomicNum()
            adjacent = neighbours[center]
            terminal_oxygen = [
                index for index in adjacent
                if connectivity.GetAtomWithIdx(index).GetAtomicNum() == 8 and len(neighbours[index]) == 1
            ]
            nitrogen = [index for index in adjacent if connectivity.GetAtomWithIdx(index).GetAtomicNum() == 7]

            if z == 6 and len(adjacent) == 2:
                terminal_nitrogen = [index for index in nitrogen if len(neighbours[index]) == 1]
                if terminal_nitrogen:
                    add({var(center, terminal_nitrogen[0], 3): 1.0}, 1.0)
            if z == 6 and len(adjacent) == 3:
                if len(terminal_oxygen) >= 2:
                    add({var(center, index, 2): 1.0 for index in terminal_oxygen}, 1.0)
                elif len(terminal_oxygen) == 1:
                    feature = atom_geometry_feature(pos, neighbours, center)
                    if feature is not None and feature >= 345.0:
                        add({var(center, terminal_oxygen[0], 2): 1.0}, 1.0)
                        # Once this centre is carbonyl-like, an attached amide
                        # nitrogen and all of its substituent bonds remain single.
                        for nitrogen_index in nitrogen:
                            add({var(center, nitrogen_index, 1): 1.0}, 1.0)
                            for substituent in neighbours[nitrogen_index]:
                                if substituent != center:
                                    add({var(nitrogen_index, substituent, 1): 1.0}, 1.0)
                elif len(nitrogen) >= 2:
                    feature = atom_geometry_feature(pos, neighbours, center)
                    if feature is not None and feature >= 345.0:
                        add({var(center, index, 2): 1.0 for index in nitrogen}, 1.0)
            if z == 7 and len(adjacent) == 3 and len(terminal_oxygen) == 2:
                add({var(center, index, 2): 1.0 for index in terminal_oxygen}, 1.0)
            if z == 16 and len(terminal_oxygen) >= 2:
                add({var(center, index, 2): 1.0 for index in terminal_oxygen}, float(min(2, len(terminal_oxygen))))
