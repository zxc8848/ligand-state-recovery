from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from bondbench.graphs import determine_rdkit_connectivity, heavy_atom_copy, heavy_reference


@dataclass(frozen=True, order=True)
class AtomState:
    capacity: int
    formal_charge: int
    hydrogens: int


@dataclass(frozen=True)
class Gaussian:
    mean: float
    sigma: float
    count: int


@dataclass(frozen=True)
class ILPResult:
    molecule: Chem.Mol
    objective: float
    solver_status: int
    solver_message: str
    selected_charge: int


def _kekule_heavy(reference: Chem.Mol) -> Chem.Mol:
    work = Chem.Mol(heavy_reference(reference))
    Chem.Kekulize(work, clearAromaticFlags=True)
    return work


def _distance(mol: Chem.Mol, i: int, j: int) -> float:
    conf = mol.GetConformer()
    left = np.asarray(conf.GetAtomPosition(i), dtype=float)
    right = np.asarray(conf.GetAtomPosition(j), dtype=float)
    return float(np.linalg.norm(left - right))


def _bond_order(bond: Chem.Bond) -> int:
    order = int(round(bond.GetBondTypeAsDouble()))
    if order not in (1, 2, 3):
        raise ValueError(f"unsupported Kekule bond order {order}")
    return order


def reference_atom_state(mol: Chem.Mol, atom_idx: int) -> AtomState:
    atom = mol.GetAtomWithIdx(atom_idx)
    capacity = sum(_bond_order(bond) for bond in atom.GetBonds())
    return AtomState(
        capacity=capacity,
        formal_charge=int(atom.GetFormalCharge()),
        hydrogens=int(atom.GetTotalNumHs(includeNeighbors=True)),
    )


def _fallback_states(atomic_num: int, degree: int) -> set[AtomState]:
    # These are broad feasibility states, not asserted labels. Learned states receive
    # a much lower objective cost. They prevent an unseen fold-level state from making
    # an otherwise sensible test molecule infeasible.
    targets: dict[int, tuple[tuple[int, int], ...]] = {
        6: ((4, 0),),
        7: ((3, 0), (4, 1), (2, -1)),
        8: ((2, 0), (1, -1), (3, 1)),
        9: ((1, 0),),
        15: ((3, 0), (5, 0)),
        16: ((2, 0), (4, 0), (6, 0)),
        17: ((1, 0),),
        35: ((1, 0),),
        53: ((1, 0),),
    }
    states: set[AtomState] = set()
    for target, charge in targets.get(atomic_num, ()):
        for capacity in range(degree, min(3 * degree, target) + 1):
            hydrogens = target - capacity
            if 0 <= hydrogens <= 4:
                states.add(AtomState(capacity, charge, hydrogens))
    return states


class HeavyAtomILPModel:
    """Cross-validation-safe bond-length and atom-state model for heavy atom clouds."""

    def __init__(self, sigma_floor: float = 0.025, atom_prior_weight: float = 2.5):
        self.sigma_floor = float(sigma_floor)
        self.atom_prior_weight = float(atom_prior_weight)
        self.pair_gaussians: dict[tuple[int, int, int], Gaussian] = {}
        self.order_gaussians: dict[int, Gaussian] = {}
        self.pair_order_counts: Counter[tuple[int, int, int]] = Counter()
        self.atom_state_counts: Counter[tuple[int, int, AtomState]] = Counter()
        self.atom_key_totals: Counter[tuple[int, int]] = Counter()
        self.atom_states_by_key: dict[tuple[int, int], set[AtomState]] = defaultdict(set)
        self.training_molecules = 0

    @staticmethod
    def _fit_gaussian(values: list[float], sigma_floor: float) -> Gaussian:
        array = np.asarray(values, dtype=float)
        sigma = float(array.std(ddof=1)) if len(array) > 1 else sigma_floor
        return Gaussian(float(array.mean()), max(sigma, sigma_floor), len(values))

    def fit(self, references: Iterable[Chem.Mol]) -> "HeavyAtomILPModel":
        pair_lengths: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        order_lengths: dict[int, list[float]] = defaultdict(list)
        self.training_molecules = 0

        for reference in references:
            mol = _kekule_heavy(reference)
            self.training_molecules += 1
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                zi = mol.GetAtomWithIdx(i).GetAtomicNum()
                zj = mol.GetAtomWithIdx(j).GetAtomicNum()
                order = _bond_order(bond)
                key = (*sorted((zi, zj)), order)
                length = _distance(mol, i, j)
                pair_lengths[key].append(length)
                order_lengths[order].append(length)
                self.pair_order_counts[key] += 1

            for atom in mol.GetAtoms():
                key = (atom.GetAtomicNum(), atom.GetDegree())
                state = reference_atom_state(mol, atom.GetIdx())
                self.atom_state_counts[(key[0], key[1], state)] += 1
                self.atom_key_totals[key] += 1
                self.atom_states_by_key[key].add(state)

        if not self.training_molecules:
            raise ValueError("cannot fit ILP model without reference molecules")
        self.pair_gaussians = {
            key: self._fit_gaussian(values, self.sigma_floor)
            for key, values in pair_lengths.items()
        }
        self.order_gaussians = {
            order: self._fit_gaussian(values, self.sigma_floor)
            for order, values in order_lengths.items()
        }
        return self

    def _bond_cost(self, zi: int, zj: int, order: int, length: float) -> float:
        key = (*sorted((zi, zj)), order)
        local = self.pair_gaussians.get(key)
        global_fit = self.order_gaussians.get(order)
        if local is not None and global_fit is not None:
            # Shrink sparse element-pair estimates towards the order-wide estimate.
            weight = local.count / (local.count + 5.0)
            mean = weight * local.mean + (1.0 - weight) * global_fit.mean
            sigma = max(
                self.sigma_floor,
                weight * local.sigma + (1.0 - weight) * global_fit.sigma,
            )
        elif global_fit is not None:
            mean, sigma = global_fit.mean, max(global_fit.sigma, 0.08)
        else:
            radii = Chem.GetPeriodicTable()
            base = radii.GetRcovalent(zi) + radii.GetRcovalent(zj)
            mean = base * {1: 1.00, 2: 0.90, 3: 0.84}[order]
            sigma = 0.12

        count = self.pair_order_counts.get(key, 0)
        pair_total = sum(
            self.pair_order_counts.get((*sorted((zi, zj)), candidate), 0)
            for candidate in (1, 2, 3)
        )
        order_prior = (count + 0.5) / (pair_total + 1.5)
        return 0.5 * ((length - mean) / sigma) ** 2 + math.log(sigma) - math.log(order_prior)

    def _states_and_costs(
        self, atomic_num: int, degree: int, unknown_charge: bool
    ) -> list[tuple[AtomState, float]]:
        key = (atomic_num, degree)
        states = set(self.atom_states_by_key.get(key, set()))
        states.update(_fallback_states(atomic_num, degree))
        if not states:
            # Last-resort neutral states for elements absent from training.
            states = {AtomState(capacity, 0, 0) for capacity in range(degree, 3 * degree + 1)}

        total = self.atom_key_totals.get(key, 0)
        denom = total + 0.5 * len(states)
        result: list[tuple[AtomState, float]] = []
        for state in sorted(states):
            count = self.atom_state_counts.get((atomic_num, degree, state), 0)
            prior = (count + 0.5) / denom if denom else 1.0 / len(states)
            cost = self.atom_prior_weight * -math.log(prior)
            if count == 0:
                cost += 1.5
            if unknown_charge:
                cost += 1.25 * abs(state.formal_charge)
            result.append((state, cost))
        return result

    def predict(self, atom_cloud: Chem.Mol, total_charge: int | None) -> ILPResult:
        connectivity = determine_rdkit_connectivity(atom_cloud, use_vdw=False)
        edges = sorted(
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            for b in connectivity.GetBonds()
        )
        if not edges:
            raise ValueError("connectivity inference returned no bonds")

        variable_costs: list[float] = []
        edge_variables: dict[tuple[int, int, int], int] = {}
        for i, j in edges:
            zi = atom_cloud.GetAtomWithIdx(i).GetAtomicNum()
            zj = atom_cloud.GetAtomWithIdx(j).GetAtomicNum()
            length = _distance(atom_cloud, i, j)
            for order in (1, 2, 3):
                edge_variables[(i, j, order)] = len(variable_costs)
                variable_costs.append(self._bond_cost(zi, zj, order, length))

        atom_variables: dict[tuple[int, AtomState], int] = {}
        candidates_by_atom: dict[int, list[AtomState]] = {}
        for atom in connectivity.GetAtoms():
            candidates = self._states_and_costs(
                atom.GetAtomicNum(), atom.GetDegree(), total_charge is None
            )
            candidates_by_atom[atom.GetIdx()] = [state for state, _ in candidates]
            for state, cost in candidates:
                atom_variables[(atom.GetIdx(), state)] = len(variable_costs)
                variable_costs.append(cost)

        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        def add_constraint(coefficients: dict[int, float], bound: float) -> None:
            row = len(lower)
            for column, value in coefficients.items():
                row_indices.append(row)
                col_indices.append(column)
                values.append(value)
            lower.append(bound)
            upper.append(bound)

        for i, j in edges:
            add_constraint({edge_variables[(i, j, order)]: 1.0 for order in (1, 2, 3)}, 1.0)

        for atom_idx, states in candidates_by_atom.items():
            add_constraint({atom_variables[(atom_idx, state)]: 1.0 for state in states}, 1.0)

            valence: dict[int, float] = {}
            for i, j in edges:
                if atom_idx in (i, j):
                    for order in (1, 2, 3):
                        valence[edge_variables[(i, j, order)]] = float(order)
            for state in states:
                valence[atom_variables[(atom_idx, state)]] = -float(state.capacity)
            add_constraint(valence, 0.0)

        if total_charge is not None:
            charge_coefficients = {
                atom_variables[(atom_idx, state)]: float(state.formal_charge)
                for atom_idx, states in candidates_by_atom.items()
                for state in states
            }
            add_constraint(charge_coefficients, float(total_charge))

        matrix = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(lower), len(variable_costs)),
        ).tocsr()
        result = milp(
            c=np.asarray(variable_costs, dtype=float),
            integrality=np.ones(len(variable_costs), dtype=np.int8),
            bounds=Bounds(np.zeros(len(variable_costs)), np.ones(len(variable_costs))),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 15.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            raise RuntimeError(f"MILP failed ({result.status}): {result.message}")

        rw = Chem.RWMol()
        chosen_states: dict[int, AtomState] = {}
        for atom in atom_cloud.GetAtoms():
            atom_idx = atom.GetIdx()
            state = max(
                candidates_by_atom[atom_idx],
                key=lambda candidate: result.x[atom_variables[(atom_idx, candidate)]],
            )
            chosen_states[atom_idx] = state
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(state.formal_charge)
            new_atom.SetNoImplicit(True)
            new_atom.SetNumExplicitHs(state.hydrogens)
            rw.AddAtom(new_atom)
        for i, j in edges:
            order = max((1, 2, 3), key=lambda value: result.x[edge_variables[(i, j, value)]])
            rw.AddBond(i, j, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order])

        conf = Chem.Conformer(atom_cloud.GetConformer())
        rw.AddConformer(conf)
        prediction = rw.GetMol()
        Chem.SanitizeMol(prediction)
        selected_charge = sum(state.formal_charge for state in chosen_states.values())
        return ILPResult(
            molecule=prediction,
            objective=float(result.fun),
            solver_status=int(result.status),
            solver_message=str(result.message),
            selected_charge=selected_charge,
        )

    def to_dict(self) -> dict:
        return {
            "training_molecules": self.training_molecules,
            "sigma_floor": self.sigma_floor,
            "atom_prior_weight": self.atom_prior_weight,
            "pair_gaussians": {
                f"{z1}-{z2}-{order}": vars(fit)
                for (z1, z2, order), fit in sorted(self.pair_gaussians.items())
            },
            "order_gaussians": {
                str(order): vars(fit) for order, fit in sorted(self.order_gaussians.items())
            },
            "atom_state_counts": [
                {
                    "atomic_num": z,
                    "degree": degree,
                    "capacity": state.capacity,
                    "formal_charge": state.formal_charge,
                    "hydrogens": state.hydrogens,
                    "count": count,
                }
                for (z, degree, state), count in sorted(self.atom_state_counts.items())
            ],
        }
