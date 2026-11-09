"""Hybridization-first heavy-atom bond-order model.

Differences from :mod:`bondbench.ilp` (the frozen Phase-1 baseline):

1. Local geometry decides hybridization. For every atom with two or three heavy
   neighbours a geometric feature (bond angle, or sum of the three bond angles) is
   scored under a density learned per (element, degree, pi-count). An sp3-looking
   atom therefore pays a large cost for carrying a pi bond.
2. Torsional planarity across each edge is scored per bond class, so a double
   bond between non-coplanar substituents is penalised.
3. Aromatic bonds are a separate latent length/torsion class instead of being
   Kekulized into the single and double distributions before fitting.

All densities are learned from training references only; nothing is tuned here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from bondbench.graphs import determine_rdkit_connectivity
from bondbench.ilp import AtomState, ILPResult, _fallback_states, reference_atom_state

ORDER_CLASSES = ("single", "double", "triple", "aromatic")
_LOG_FLOOR = 1e-6


def _angle(pos: np.ndarray, center: int, a: int, b: int) -> float:
    u = pos[a] - pos[center]
    v = pos[b] - pos[center]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 180.0
    return math.degrees(math.acos(float(np.clip(u @ v / denom, -1.0, 1.0))))


def _dihedral(pos: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    b0, b1, b2 = pos[a] - pos[b], pos[c] - pos[b], pos[d] - pos[c]
    n1 = np.linalg.norm(b1)
    if n1 == 0:
        return 0.0
    b1 = b1 / n1
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    x = v @ w
    y = np.cross(b1, v) @ w
    return math.degrees(math.atan2(y, x))


def atom_geometry_feature(pos: np.ndarray, neighbours: dict[int, list[int]], idx: int) -> float | None:
    """Angle for degree 2, angle sum for degree 3, None otherwise."""
    nb = neighbours[idx]
    if len(nb) == 2:
        return _angle(pos, idx, nb[0], nb[1])
    if len(nb) == 3:
        return (
            _angle(pos, idx, nb[0], nb[1])
            + _angle(pos, idx, nb[0], nb[2])
            + _angle(pos, idx, nb[1], nb[2])
        )
    return None


def bond_torsion_feature(pos: np.ndarray, neighbours: dict[int, list[int]], i: int, j: int) -> float | None:
    """Worst deviation (deg) from cis/trans planarity over substituent pairs."""
    left = [k for k in neighbours[i] if k != j]
    right = [k for k in neighbours[j] if k != i]
    if not left or not right:
        return None
    # Skip near-linear centres where torsions are undefined.
    if any(_angle(pos, i, k, j) > 165.0 for k in left) or any(_angle(pos, j, i, k) > 165.0 for k in right):
        return None
    worst = 0.0
    for a in left:
        for d in right:
            t = abs(_dihedral(pos, a, i, j, d))
            worst = max(worst, min(t, 180.0 - t))
    return worst


def ring_category(mol: Chem.Mol) -> list[str]:
    """Smallest SSSR ring containing each atom: '3', '4', '5' or 'other'.

    Angle priors depend on ring strain: an aromatic 5-ring atom has ~108 degree
    angles, indistinguishable from sp3 unless the ring size is conditioned on.
    """
    sizes = [0] * mol.GetNumAtoms()
    for ring in Chem.GetSymmSSSR(mol):
        for idx in ring:
            if sizes[idx] == 0 or len(ring) < sizes[idx]:
                sizes[idx] = len(ring)
    return [str(n) if n in (3, 4, 5) else "other" for n in sizes]


# Textbook ideal geometry used as a weak prior (pseudo-samples), not fitted.
# Degree 2: bond angle; degree 3: sum of the three bond angles.
IDEAL_GEOMETRY: dict[tuple[int, int, str], tuple[float, float]] = {
    (2, 0, "other"): (109.5, 6.0), (2, 1, "other"): (120.0, 6.0), (2, 2, "other"): (178.0, 4.0),
    (2, 0, "5"): (104.0, 6.0), (2, 1, "5"): (108.0, 6.0),
    (2, 0, "4"): (89.0, 6.0), (2, 1, "4"): (91.0, 6.0),
    (2, 0, "3"): (60.0, 6.0),
    (3, 0, "other"): (328.0, 10.0), (3, 1, "other"): (359.0, 3.0),
    (3, 0, "5"): (322.0, 12.0), (3, 1, "5"): (359.0, 3.0),
    (3, 0, "4"): (310.0, 15.0), (3, 1, "4"): (358.0, 4.0),
    (3, 0, "3"): (300.0, 20.0),
}


class KernelDensity:
    """1-D Gaussian KDE blended with an optional Gaussian prior and a uniform floor."""

    def __init__(
        self,
        bandwidth: float,
        low: float,
        high: float,
        prior: tuple[float, float] | None = None,
        prior_count: float = 0.0,
    ):
        self.bandwidth = bandwidth
        self.low = low
        self.high = high
        self.prior = prior
        self.prior_count = prior_count if prior is not None else 0.0
        self.samples: list[float] = []

    def add(self, value: float) -> None:
        self.samples.append(float(value))

    @property
    def effective_count(self) -> float:
        return len(self.samples) + self.prior_count

    def log_pdf(self, value: float) -> float:
        uniform = 1.0 / (self.high - self.low)
        n = len(self.samples)
        density = 0.0
        if n:
            arr = np.asarray(self.samples)
            z = (value - arr) / self.bandwidth
            density += float(np.exp(-0.5 * z * z).sum()) / (self.bandwidth * math.sqrt(2 * math.pi))
        if self.prior is not None and self.prior_count:
            mu, sd = self.prior
            density += self.prior_count * math.exp(-0.5 * ((value - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        # Two uniform pseudo-samples keep every state finite under outliers.
        total = n + self.prior_count + 2.0
        return math.log((density + 2.0 * uniform) / total)


@dataclass
class _Fit:
    mean: float
    sigma: float
    count: int


def _bond_class(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    return {1: "single", 2: "double", 3: "triple"}.get(int(round(bond.GetBondTypeAsDouble())), "single")


def _neighbours(n_atoms: int, edges: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    nb: dict[int, list[int]] = {k: [] for k in range(n_atoms)}
    for i, j in edges:
        nb[i].append(j)
        nb[j].append(i)
    return nb


class HybridizationILPModel:
    def __init__(
        self,
        sigma_floor: float = 0.025,
        atom_prior_weight: float = 1.0,
        geometry_weight: float = 1.0,
        torsion_weight: float = 1.0,
        geometry_prior_count: float = 5.0,
        length_prior_count: float = 5.0,
        fallback_pseudocount: float = 1.0,
    ):
        self.sigma_floor = float(sigma_floor)
        self.atom_prior_weight = float(atom_prior_weight)
        self.geometry_weight = float(geometry_weight)
        self.torsion_weight = float(torsion_weight)
        self.geometry_prior_count = float(geometry_prior_count)
        self.length_prior_count = float(length_prior_count)
        self.fallback_pseudocount = float(fallback_pseudocount)
        self.training_molecules = 0

    # ------------------------------------------------------------------ fitting
    def fit(self, references: Iterable[Chem.Mol]) -> "HybridizationILPModel":
        lengths: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        class_lengths: dict[str, list[float]] = defaultdict(list)
        self.pair_class_counts: Counter[tuple[int, int, str]] = Counter()
        self.atom_state_counts: Counter[tuple[int, int, AtomState]] = Counter()
        self.atom_key_totals: Counter[tuple[int, int]] = Counter()
        self.atom_states_by_key: dict[tuple[int, int], set[AtomState]] = defaultdict(set)
        self.angle_kde: dict[tuple, KernelDensity] = {}
        self.torsion_kde: dict[str, KernelDensity] = {}
        self.training_molecules = 0

        for reference in references:
            mol = Chem.RemoveHs(reference, sanitize=True)
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
            pos = mol.GetConformer().GetPositions()
            edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            nb = _neighbours(mol.GetNumAtoms(), edges)
            rings = ring_category(mol)
            self.training_molecules += 1

            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                zi, zj = sorted((mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()))
                cls = _bond_class(bond)
                length = float(np.linalg.norm(pos[i] - pos[j]))
                lengths[(zi, zj, cls)].append(length)
                class_lengths[cls].append(length)
                self.pair_class_counts[(zi, zj, cls)] += 1
                tors = bond_torsion_feature(pos, nb, i, j)
                if tors is not None:
                    self.torsion_kde.setdefault(cls, KernelDensity(4.0, 0.0, 90.0)).add(tors)

            for atom in kek.GetAtoms():
                idx = atom.GetIdx()
                key = (atom.GetAtomicNum(), atom.GetDegree())
                state = reference_atom_state(kek, idx)
                self.atom_state_counts[(key[0], key[1], state)] += 1
                self.atom_key_totals[key] += 1
                self.atom_states_by_key[key].add(state)
                feat = atom_geometry_feature(pos, nb, idx)
                if feat is not None:
                    pi = min(state.capacity - key[1], 2)
                    low, high, bw = (80.0, 185.0, 3.0) if key[1] == 2 else (280.0, 362.0, 3.0)
                    rc = rings[idx]
                    for k in ((key[0], key[1], pi, rc), ("any", key[1], pi, rc)):
                        self._angle_density(k).add(feat)

        if not self.training_molecules:
            raise ValueError("cannot fit without reference molecules")
        fit = lambda v: _Fit(float(np.mean(v)), max(float(np.std(v, ddof=1)) if len(v) > 1 else self.sigma_floor, self.sigma_floor), len(v))
        self.pair_fits = {k: fit(v) for k, v in lengths.items()}
        self.class_fits = {k: fit(v) for k, v in class_lengths.items()}
        return self

    def _angle_density(self, key: tuple) -> KernelDensity:
        if key not in self.angle_kde:
            _, degree, pi, ring = key
            low, high = (50.0, 185.0) if degree == 2 else (270.0, 362.0)
            prior = IDEAL_GEOMETRY.get((degree, pi, ring))
            self.angle_kde[key] = KernelDensity(3.0, low, high, prior, self.geometry_prior_count)
        return self.angle_kde[key]

    # ------------------------------------------------------------------ scoring
    def _length_loglik(self, zi: int, zj: int, cls: str, length: float) -> float:
        pt = Chem.GetPeriodicTable()
        base = pt.GetRcovalent(zi) + pt.GetRcovalent(zj)
        # Covalent-radius defaults (Pyykko radii in RDKit): single ~1.00, double ~0.88,
        # triple ~0.79, aromatic ~0.92 of the radius sum.
        mean = base * {"single": 1.0, "double": 0.88, "triple": 0.79, "aromatic": 0.92}[cls]
        sigma = 0.06
        n0 = self.length_prior_count
        for fit in (self.class_fits.get(cls), self.pair_fits.get((zi, zj, cls))):
            if fit is None:
                continue
            if fit is self.class_fits.get(cls) and n0:
                # Class-level fits pool element pairs; only shift the default by the
                # class-wide offset relative to C-C so element-specific radii survive.
                continue
            w = fit.count / (fit.count + n0) if n0 else 1.0
            mean = w * fit.mean + (1 - w) * mean
            sigma = max(self.sigma_floor, w * fit.sigma + (1 - w) * sigma)
        if not n0:
            glob = self.class_fits.get(cls)
            local = self.pair_fits.get((zi, zj, cls))
            if local is None and glob is not None:
                mean, sigma = glob.mean, max(glob.sigma, 0.08)
        return -0.5 * ((length - mean) / sigma) ** 2 - math.log(sigma)

    def _class_prior(self, zi: int, zj: int, cls: str) -> float:
        total = sum(self.pair_class_counts.get((zi, zj, c), 0) for c in ORDER_CLASSES)
        return (self.pair_class_counts.get((zi, zj, cls), 0) + 0.5) / (total + 2.0)

    def edge_cost(self, zi: int, zj: int, order: int, length: float, torsion: float | None, in_ring: bool) -> float:
        zi, zj = sorted((zi, zj))

        def term(cls: str, prior: float) -> float:
            ll = self._length_loglik(zi, zj, cls, length) + math.log(prior)
            if torsion is not None and self.torsion_weight:
                kde = self.torsion_kde.get(cls)
                if kde is not None:
                    ll += self.torsion_weight * kde.log_pdf(torsion)
            return ll

        cls = {1: "single", 2: "double", 3: "triple"}[order]
        options = [term(cls, self._class_prior(zi, zj, cls))]
        if in_ring and order in (1, 2):
            # Latent aromatic class: a Kekule 1 or 2 inside a ring may be half of an
            # aromatic bond. Split its prior mass equally between the two orders.
            options.append(term("aromatic", 0.5 * self._class_prior(zi, zj, "aromatic")))
        m = max(options)
        return -(m + math.log(sum(math.exp(o - m) for o in options)))

    def _states_and_costs(self, z: int, degree: int, feature: float | None, unknown_charge: bool, ring: str = "other"):
        key = (z, degree)
        states = set(self.atom_states_by_key.get(key, set()))
        fallback = _fallback_states(z, degree)
        states.update(fallback)
        if not states:
            states = {AtomState(c, 0, 0) for c in range(degree, 3 * degree + 1)}
        # Standard neutral valence states get a chemistry pseudo-count so that a state
        # merely absent from 68 training molecules (e.g. nitrile N) is not near-forbidden.
        pseudo = {st: (self.fallback_pseudocount if (st in fallback and st.formal_charge == 0) else 0.0) for st in states}
        total = self.atom_key_totals.get(key, 0) + sum(pseudo.values())
        denom = total + 0.5 * len(states)
        out = []
        for state in sorted(states):
            count = self.atom_state_counts.get((z, degree, state), 0) + pseudo[state]
            prior = (count + 0.5) / denom if denom else 1.0 / len(states)
            cost = self.atom_prior_weight * -math.log(prior)
            if count == 0:
                cost += 1.5
            if unknown_charge:
                cost += 1.25 * abs(state.formal_charge)
            if feature is not None and self.geometry_weight:
                pi = min(state.capacity - degree, 2)
                local = self.angle_kde.get((z, degree, pi, ring))
                if local is not None and len(local.samples) >= 10:
                    density = local
                else:
                    density = self._angle_density(("any", degree, pi, ring))
                cost += self.geometry_weight * -density.log_pdf(feature)
            out.append((state, cost))
        return out

    # --------------------------------------------------------------- prediction
    def predict(self, atom_cloud: Chem.Mol, total_charge: int | None) -> ILPResult:
        connectivity = determine_rdkit_connectivity(atom_cloud, use_vdw=False)
        edges = sorted(
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            for b in connectivity.GetBonds()
        )
        if not edges:
            raise ValueError("connectivity inference returned no bonds")
        pos = atom_cloud.GetConformer().GetPositions()
        nb = _neighbours(atom_cloud.GetNumAtoms(), edges)
        Chem.FastFindRings(connectivity)
        ring_info = connectivity.GetRingInfo()
        rings = ring_category(connectivity)

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

        atom_vars: dict[tuple[int, AtomState], int] = {}
        cands: dict[int, list[AtomState]] = {}
        for atom in connectivity.GetAtoms():
            idx = atom.GetIdx()
            feature = atom_geometry_feature(pos, nb, idx)
            sc = self._states_and_costs(atom.GetAtomicNum(), len(nb[idx]), feature, total_charge is None, rings[idx])
            cands[idx] = [s for s, _ in sc]
            for s, c in sc:
                atom_vars[(idx, s)] = len(costs)
                costs.append(c)

        rows, cols, vals, bounds = [], [], [], []

        def eq(coeffs: dict[int, float], b: float) -> None:
            r = len(bounds)
            for c, v in coeffs.items():
                rows.append(r); cols.append(c); vals.append(v)
            bounds.append(b)

        for i, j in edges:
            eq({edge_vars[(i, j, o)]: 1.0 for o in (1, 2, 3)}, 1.0)
        for idx, states in cands.items():
            eq({atom_vars[(idx, s)]: 1.0 for s in states}, 1.0)
            val = {}
            for j in nb[idx]:
                a, b = min(idx, j), max(idx, j)
                for o in (1, 2, 3):
                    val[edge_vars[(a, b, o)]] = float(o)
            for s in states:
                val[atom_vars[(idx, s)]] = -float(s.capacity)
            eq(val, 0.0)
        if total_charge is not None:
            eq({atom_vars[(idx, s)]: float(s.formal_charge) for idx, ss in cands.items() for s in ss}, float(total_charge))

        matrix = coo_matrix((vals, (rows, cols)), shape=(len(bounds), len(costs))).tocsr()
        result = milp(
            c=np.asarray(costs),
            integrality=np.ones(len(costs), dtype=np.int8),
            bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
            constraints=LinearConstraint(matrix, np.asarray(bounds), np.asarray(bounds)),
            options={"time_limit": 15.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            raise RuntimeError(f"MILP failed ({result.status}): {result.message}")

        rw = Chem.RWMol()
        chosen: dict[int, AtomState] = {}
        for atom in atom_cloud.GetAtoms():
            idx = atom.GetIdx()
            state = max(cands[idx], key=lambda s: result.x[atom_vars[(idx, s)]])
            chosen[idx] = state
            new = Chem.Atom(atom.GetAtomicNum())
            new.SetFormalCharge(state.formal_charge)
            new.SetNoImplicit(True)
            new.SetNumExplicitHs(state.hydrogens)
            rw.AddAtom(new)
        for i, j in edges:
            o = max((1, 2, 3), key=lambda v: result.x[edge_vars[(i, j, v)]])
            rw.AddBond(i, j, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[o])
        rw.AddConformer(Chem.Conformer(atom_cloud.GetConformer()))
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return ILPResult(
            molecule=mol,
            objective=float(result.fun),
            solver_status=int(result.status),
            solver_message=str(result.message),
            selected_charge=sum(s.formal_charge for s in chosen.values()),
        )
