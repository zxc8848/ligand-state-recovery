"""Constraint-based chemical repair for coordinate-first molecule generators.

The module deliberately separates three operations:

* ``joint_decode`` selects all bond orders and formal-charge states together;
* ``project_coordinates`` makes an optional *minimal* local geometry projection,
  without treating one conformer as the unique coordinate label;
* ``RepairabilityCalibrator`` can learn when the repair should be rejected.

An accepted joint decode is guaranteed to obey the selected valence/charge
constraints and is explicitly checked for aromatic Kekulizability.  This is a
graph-constraint guarantee, not a claim that every accepted molecule is
energetically stable or synthetically plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .structured_decoder import BOND_ORDER, MAX_VALENCE

# Valence *capacity* accounting must not use the fractional aromatic order 1.5.
# With 1.5 a pyrrole-type N-H (2 aromatic + 1 H = 4.0) exceeds neutral N's cap of 3
# and is forced into N+, which then breaks global charge conservation; a fused
# aromatic junction carbon (3 x 1.5 = 4.5) exceeds carbon's cap outright. Both make
# the *correct* graph infeasible, so the MILP is forced to return a wrong one.
# Counting an aromatic/delocalized bond as 1.0 for capacity keeps every true
# heteroaromatic and fused motif feasible and neutral; the exact chemistry is still
# arbitrated by the final RDKit sanitization, which is what these bounds defer to.
VALENCE_BOND_ORDER = np.asarray((0.0, 1.0, 2.0, 3.0, 1.0, 1.0), dtype=np.float64)


@dataclass
class JointDecodeResult:
    labels: torch.Tensor
    formal_charge: torch.Tensor
    implicit_h_count: torch.Tensor
    solved: bool
    kekulizable: bool
    objective: float
    local_overvalent_atoms: int


def _undirected_scores(logits: torch.Tensor, edge_index: torch.Tensor):
    keys = [tuple(sorted((int(i), int(j)))) for i, j in edge_index.detach().cpu().tolist()]
    rows = {}
    for row, key in enumerate(keys):
        rows.setdefault(key, []).append(row)
    unique = list(rows)
    logp = torch.log_softmax(logits.detach(), dim=-1).cpu()
    scores = np.stack([logp[rows[key]].mean(dim=0).numpy() for key in unique])
    return keys, rows, unique, scores


def _candidate_cycle_edges(edges: list[tuple[int, int]], n_atoms: int) -> set[tuple[int, int]]:
    """Return candidate edges that belong to at least one undirected cycle.

    The decoder has only a local candidate graph, not a reference ring graph.
    An edge is nevertheless cycle-capable exactly when its endpoints remain
    connected after the edge is removed.  Molecules here are small, so the
    deliberately simple per-edge DFS is clearer and more robust than a second
    graph dependency.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for edge_id, (i, j) in enumerate(edges):
        adjacency[i].append((j, edge_id))
        adjacency[j].append((i, edge_id))
    in_cycle: set[tuple[int, int]] = set()
    for blocked, (start, target) in enumerate(edges):
        stack = [start]
        seen = {start}
        while stack:
            node = stack.pop()
            if node == target:
                in_cycle.add((start, target))
                break
            for neighbour, edge_id in adjacency[node]:
                if edge_id != blocked and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return in_cycle


def _selected_aromatic_bridges(edges: list[tuple[int, int]], labels: np.ndarray) -> list[int]:
    """Return class-4 edges that are bridges in the selected aromatic graph.

    Candidate-cycle membership is only a necessary condition for aromaticity:
    a candidate shortcut can close a cycle even when the final selected
    aromatic graph contains a dangling bridge.  This graph-only check operates
    after MILP selection and has no access to reference bonds.
    """
    aromatic = [(edge_id, edge) for edge_id, edge in enumerate(edges) if int(labels[edge_id]) == 4]
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for local_id, (_edge_id, (left, right)) in enumerate(aromatic):
        adjacency.setdefault(left, []).append((right, local_id))
        adjacency.setdefault(right, []).append((left, local_id))
    bridges = []
    for local_id, (edge_id, (start, target)) in enumerate(aromatic):
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            if node == target:
                break
            for neighbour, blocked in adjacency.get(node, []):
                if blocked != local_id and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if target not in seen:
            bridges.append(edge_id)
    return bridges


def _repair_aromatic_bridges(edges: list[tuple[int, int]], labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Demote selected aromatic bridges to their best non-aromatic class."""
    repaired = labels.copy()
    non_aromatic = np.asarray((0, 1, 2, 3, 5), dtype=np.int64)
    while True:
        bridges = _selected_aromatic_bridges(edges, repaired)
        if not bridges:
            return repaired
        for edge_id in bridges:
            repaired[edge_id] = int(non_aromatic[scores[edge_id, non_aromatic].argmax()])


def _charge_states(z: int, state_catalogue: dict[int, tuple[tuple[int, float], ...]] | None = None):
    """(formal charge, maximum total valence) organic state catalogue.

    Values are *capacities*, not equalities (see ``_atom_states``), so the
    neutral entry for each element carries its highest neutral valence.

    Neutral hypervalent P and S must be present here.  Without them the MILP
    cannot represent a phosphate, phosphonate, sulfone, sulfonamide or sulfate
    at formal charge 0: reaching valence 5 (P) or 6 (S) would require charge +2,
    and because total charge is conserved exactly, that +2 has to be paid for
    somewhere else in the molecule.  When it cannot be, the optimizer instead
    demotes the P=O / S=O bonds to single.

    That was measured, not assumed.  On the 488-molecule PDBBind pose set, of
    the 145 true-double-predicted-single edges:

        69  on neutral P centres whose reference valence sum is 5.0
        21  on neutral S centres whose reference valence sum is 6.0

    i.e. 90/145 of that error class, and the refiner's own top-ranked class was
    already correct for 55 of the 145 -- the loss happened in the decoder, not
    in the model.
    """
    if state_catalogue is not None and int(z) in state_catalogue:
        return tuple((int(q), float(vmax)) for q, vmax in state_catalogue[int(z)])
    if z == 7:   # N-, neutral N, N+
        return ((-1, 2.0), (0, 3.0), (1, 4.0))
    if z == 8:   # O-, neutral O, O+
        return ((-1, 1.0), (0, 2.0), (1, 3.0))
    if z == 16:  # sulfide/sulfoxide/sulfone all occur at formal charge 0
        return ((-1, 1.0), (0, 6.0), (1, 3.0), (2, 6.0))
    if z == 15:  # phosphine (3) and phosphate/phosphonate (5), both neutral
        return ((0, 5.0), (1, 4.0), (2, 5.0))
    return ((0, float(MAX_VALENCE[int(z)])),)


def _atom_states(
    z: int,
    explicit_h: bool = False,
    state_catalogue: dict[int, tuple[tuple[int, float], ...]] | None = None,
):
    """Return feasible ``(charge, implicit-H count, total-valence cap)`` states.

    These are deliberately *capacities*, rather than an equality constraint:
    aromatic 1.5 bond labels and hypervalent motifs make a universal exact
    valence equation incorrect.  A final RDKit sanitization check determines
    whether a decoded proposal is chemically representable.
    """
    # For an all-atom generator every hydrogen is already represented by a
    # node and must be selected through an H-X edge.  Giving those atoms an
    # additional implicit-H choice would double count valence.
    states = []
    for q, vmax in _charge_states(z, state_catalogue=state_catalogue):
        for h in (0,) if explicit_h else range(min(4, int(vmax)) + 1):
            states.append((int(q), int(h), float(vmax)))
    return tuple(states)


def _kekulizable(elems: torch.Tensor, edges: list[tuple[int, int]], labels: np.ndarray,
                 charges: np.ndarray, implicit_h: np.ndarray | None = None,
                 all_atoms_explicit_h: bool = False) -> bool:
    """Check a reconstructed heavy-atom graph with RDKit sanitization.

    The historical name is kept for compatibility.  The check now includes
    Kekulization *and* formal-valence sanitization after attaching the decoded
    implicit hydrogens, so an accepted proposal is a usable molecular graph.
    """
    return build_sanitized_mol(
        elems, edges, labels, charges, implicit_h,
        all_atoms_explicit_h=all_atoms_explicit_h,
    ) is not None


def build_sanitized_mol(
    elems: torch.Tensor,
    edges: list[tuple[int, int]],
    labels: np.ndarray,
    charges: np.ndarray,
    implicit_h: np.ndarray | None = None,
    coord: torch.Tensor | None = None,
    all_atoms_explicit_h: bool = False,
):
    """Build a sanitized RDKit molecule from a decoded heavy-atom graph.

    This is the boundary between BondNet's graph decision and a downstream
    generator: it deliberately uses only decoded graph states, never reference
    bonds. ``None`` denotes an invalid graph rather than a repairable result.
    """
    try:
        from rdkit import Chem

        rw = Chem.RWMol()
        for atom_idx, (z, q) in enumerate(zip(elems.detach().cpu().tolist(), charges.tolist())):
            atom = Chem.Atom(int(z))
            atom.SetFormalCharge(int(q))
            if all_atoms_explicit_h:
                atom.SetNoImplicit(True)
            elif implicit_h is not None:
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(int(implicit_h[atom_idx]))
            rw.AddAtom(atom)
        type_map = {
            1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC,
            # Delocalized groups remain a single-bond graph representation;
            # formal charge states encode their non-local electron assignment.
            5: Chem.BondType.SINGLE,
        }
        aromatic_atoms = set()
        for (i, j), cls in zip(edges, labels):
            if int(cls) == 0:
                continue
            rw.AddBond(int(i), int(j), type_map[int(cls)])
            if int(cls) == 4:
                aromatic_atoms.update((int(i), int(j)))
        mol = rw.GetMol()
        for idx in aromatic_atoms:
            mol.GetAtomWithIdx(idx).SetIsAromatic(True)
        for (i, j), cls in zip(edges, labels):
            if int(cls) == 4:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)
        Chem.Kekulize(mol, clearAromaticFlags=False)
        Chem.SanitizeMol(mol)
        if coord is not None:
            conformer = Chem.Conformer(int(elems.numel()))
            for index, xyz in enumerate(coord.detach().cpu().tolist()):
                conformer.SetAtomPosition(index, xyz)
            mol.AddConformer(conformer, assignId=True)
        return mol
    except Exception:
        return None


def forcefield_basin_displacement(mol) -> float | None:
    """Return mean heavy-atom displacement after a short classical minimization.

    The input molecule is never mutated.  ``None`` means that neither MMFF94s
    nor UFF can parameterize the molecule, so callers must not interpret it as
    a poor geometry score.  This is intentionally a posterior ranking signal,
    not a new hard validity constraint.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        work = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        n_source_atoms = mol.GetNumAtoms()
        conformer = work.GetConformer()
        before = np.array([list(conformer.GetAtomPosition(i)) for i in range(n_source_atoms)], dtype=float)
        heavy = [i for i in range(n_source_atoms) if work.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if not heavy:
            return None
        ff = None
        if AllChem.MMFFHasAllMoleculeParams(work):
            props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(work, props)
        if ff is None and AllChem.UFFHasAllMoleculeParams(work):
            ff = AllChem.UFFGetMoleculeForceField(work)
        if ff is None:
            return None
        ff.Minimize(maxIts=200)
        after = np.array([list(conformer.GetAtomPosition(i)) for i in range(n_source_atoms)], dtype=float)
        return float(np.linalg.norm(after - before, axis=1)[heavy].mean())
    except Exception:
        return None


# Mean log-probabilities are bounded well above this, so it dominates any real
# score while staying finite for the MILP objective.
_DELOCALIZED_FORBIDDEN_SCORE = -1e6


def decode_top_n(logits, edge_index, elems, total_charge=0, n_solutions=8, **kwargs):
    """Enumerate the N best-scoring feasible graphs, best first.

    Rationale: 52% of this decoder's residual edge errors sit on symmetric
    terminal-oxy groups whose bond-order assignment is fixed by a hydrogen that
    the input does not contain, and the correct graph lies within a two-edge
    flip of the argmax for 87% of molecules.  A single answer therefore
    discards recoverable information; a short ranked list does not.

    Implemented with the MILP's existing no-good cuts: each accepted solution is
    appended to ``forbidden_edge_labelings``, which constrains the next solve to
    differ from it in at least one edge.  Solutions are therefore distinct by
    construction and come out in non-increasing objective order.

    Note that distinct *labelings* are not the same as distinct *molecules* --
    several of them will be alternative notations for one species.  Collapse
    them with ``scripts/classify_candidate_ambiguity.py`` before presenting a
    candidate list, or the output will look far more uncertain than it is.

    Returns a list of ``(JointDecodeResult, per_unique_edge_labels)``.  The
    second element is the length-m array in the decoder's internal unique-edge
    order, which is what the no-good cut consumes.
    """
    require_kekulizable = bool(kwargs.pop("require_kekulizable", False))
    max_attempts = int(kwargs.pop("max_attempts", max(4 * n_solutions, n_solutions)))
    if logits.numel() == 0 or n_solutions < 1:
        return []
    keys, rows, unique, _ = _undirected_scores(logits, edge_index)
    forbidden = list(kwargs.pop("forbidden_edge_labelings", None) or [])

    solutions = []
    # A no-good cut is also a cheap lazy chemical constraint: if a top-scoring
    # graph cannot be represented by RDKit, exclude that *complete* labeling
    # and ask the same MILP for the next hypothesis.  This is deliberately a
    # post-solve check instead of encoding Hückel/Kekule logic in the MILP,
    # which would be non-local and fragile for fused heteroaromatics.
    for _ in range(max_attempts):
        result = joint_decode(logits, edge_index, elems, total_charge,
                              forbidden_edge_labelings=forbidden or None, **kwargs)
        if not result.solved:
            break
        # joint_decode returns labels over directed rows; the cut needs one
        # label per undirected edge, in `unique` order.
        labels = result.labels.detach().cpu()
        per_edge = np.asarray([int(labels[rows[key][0]]) for key in unique], dtype=np.int64)
        if any(np.array_equal(per_edge, previous) for previous in forbidden):
            # The solver returned an already-excluded labeling; the cut did not
            # bind. Stop rather than loop forever emitting duplicates.
            break
        forbidden.append(per_edge)
        if require_kekulizable:
            # Heavy-only deployment does not observe H coordinates.  Match the
            # output construction convention: let RDKit infer compatible H
            # counts from the decoded graph and formal charges rather than
            # rejecting an otherwise valid graph for one arbitrary H state.
            usable = _kekulizable(
                elems, unique, per_edge,
                result.formal_charge.detach().cpu().numpy(), implicit_h=None,
            )
            if not usable:
                continue
        solutions.append((result, per_edge))
        if len(solutions) >= n_solutions:
            break
    return solutions


def _gate_delocalized_to_trained_elements(scores, unique, elems):
    """Forbid class 5 on edges the class-5 label was never defined for.

    ``mark_symmetric_delocalized`` (training) only ever writes class 5 onto a
    (C|N)--O edge whose O is terminal.  The six-class head nonetheless emits
    class 5 on chemically similar but untrained centres -- most of all the
    symmetric terminal-oxy groups of phosphorus and sulfur (phosphate,
    phosphonate, sulfone, sulfonate), where both the training labels and the
    CCD reference keep conventional single/double bonds.

    On the 488-molecule PDBBind set this is the single largest error source:
    true 1->predicted 5 (143 edges) plus true 2->predicted 5 (139 edges) is
    51.6% of all mislabelled bonds, and molecules containing such a group fail
    strict exact match at 63.2% versus 36.3% for molecules without one.

    The gate applies the *necessary element condition* of the training rule --
    a class-5 edge must be (C|N)--O -- which needs no knowledge of the decoded
    graph and so cannot circularly depend on the prediction being gated.  It
    deliberately does not check terminality; the existing
    ``decode_delocalized_oxy_groups`` promotion rule handles the sufficient
    condition on the other side.

    Scores are mean log-probabilities, so a large *finite* penalty is used
    rather than ``-inf``: the MILP objective at the end of this function is
    built as ``-scores``, and an infinity there would make the program
    unsolvable instead of merely making class 5 unattractive.
    """
    z = [int(v) for v in elems.detach().cpu().tolist()]
    gated = scores.copy()
    for edge_id, (i, j) in enumerate(unique):
        zi, zj = z[i], z[j]
        centre_oxygen = ((zi in (6, 7) and zj == 8) or (zj in (6, 7) and zi == 8))
        if not centre_oxygen:
            gated[edge_id, 5] = _DELOCALIZED_FORBIDDEN_SCORE
    return gated


def joint_decode(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    elems: torch.Tensor,
    total_charge: int = 0,
    formal_charge: torch.Tensor | None = None,
    charge_logits: torch.Tensor | None = None,
    implicit_h_logits: torch.Tensor | None = None,
    charge_weight: float = 0.25,
    implicit_h_weight: float = 0.25,
    time_limit: float = 2.0,
    explicit_h: bool = False,
    aromatic_constraints: bool = False,
    aromatic_bridge_repair: bool = False,
    forbidden_edge_labelings: list[np.ndarray] | None = None,
    forbid_aromatic: bool = False,
    restrict_delocalized: bool = False,
    state_catalogue: dict[int, tuple[tuple[int, float], ...]] | None = None,
) -> JointDecodeResult:
    """Jointly optimize bond classes and atomic charge states with a MILP.

    The selected graph obeys one class per pair, per-atom valence upper bounds
    selected by its formal-charge state, and exact global charge conservation.
    A failed MILP or Kekulization is reported to the caller so that the sample
    can be rejected rather than silently returned as a false-valid molecule.
    """
    if logits.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=logits.device)
        return JointDecodeResult(empty, torch.zeros_like(elems), torch.zeros_like(elems), True, True, 0.0, 0)
    if logits.shape[-1] != 6:
        raise ValueError("joint decoder expects six bond classes")
    keys, rows, unique, scores = _undirected_scores(logits, edge_index)
    if restrict_delocalized:
        scores = _gate_delocalized_to_trained_elements(scores, unique, elems)
    m, n_classes, n_atoms = len(unique), scores.shape[1], int(elems.numel())
    local = scores.argmax(axis=1)
    local_valence = np.zeros(n_atoms, dtype=np.float64)
    for cls, (i, j) in zip(local, unique):
        local_valence[i] += VALENCE_BOND_ORDER[cls]
        local_valence[j] += VALENCE_BOND_ORDER[cls]
    naive_limit = np.asarray([MAX_VALENCE[int(z)] for z in elems.detach().cpu().tolist()])
    local_overvalent = int(np.sum(local_valence > naive_limit + 1e-6))

    supplied_charge = (formal_charge.detach().cpu().numpy().astype(np.int64)
                       if formal_charge is not None else None)
    state_sets = []
    for atom, z in enumerate(elems.detach().cpu().tolist()):
        choices = _atom_states(
            int(z), explicit_h=explicit_h, state_catalogue=state_catalogue
        )
        if supplied_charge is not None:
            fixed = tuple(state for state in choices if state[0] == int(supplied_charge[atom]))
            if fixed:
                choices = fixed
        state_sets.append(choices)
    local_charge = (supplied_charge if supplied_charge is not None
                    else np.zeros(n_atoms, dtype=np.int64))
    local_limits = np.asarray([
        next((vmax for q, _h, vmax in choices if q == int(local_charge[a])), max(v for _, _, v in choices))
        for a, choices in enumerate(state_sets)
    ])
    # Do not rewrite a high-confidence graph that already satisfies the full
    # supplied-charge constraints and aromatic feasibility. The joint program
    # is a repair operator, not a gratuitous alternative graph generator.
    if (not forbidden_edge_labelings and supplied_charge is not None and np.all(local_valence <= local_limits + 1e-6)
            and int(local_charge.sum()) == int(total_charge)
            and _kekulizable(elems, unique, local, local_charge,
                             all_atoms_explicit_h=explicit_h)):
        labels = torch.empty(len(keys), dtype=torch.long, device=logits.device)
        for e, key in enumerate(unique):
            labels[rows[key]] = int(local[e])
        return JointDecodeResult(
            labels, torch.tensor(local_charge, dtype=torch.long, device=logits.device),
            torch.zeros(n_atoms, dtype=torch.long, device=logits.device),
            True, True, float(scores[np.arange(m), local].sum()), local_overvalent,
        )

    state_offsets, states = [], []
    for choices in state_sets:
        state_offsets.append(len(states))
        states.extend(choices)
    n_state = len(states)
    n_edge_vars = m * n_classes
    n_aromatic_atom_vars = n_atoms if aromatic_constraints else 0
    n_vars = n_edge_vars + n_state + n_aromatic_atom_vars
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        # Edge one-hot, charge-state one-hot, global charge, atom valence,
        # and optionally aromatic degree coherence at each atom.
        base_rows = m + n_atoms + 1 + n_atoms
        n_aromatic_rows = 2 * n_atoms if aromatic_constraints else 0
        forbidden = forbidden_edge_labelings or []
        n_rows = base_rows + n_aromatic_rows + len(forbidden)
        constraints = lil_matrix((n_rows, n_vars), dtype=np.float64)
        lb = np.full(n_rows, -np.inf, dtype=np.float64)
        ub = np.full(n_rows, np.inf, dtype=np.float64)
        aromatic_start = n_edge_vars + n_state
        aromatic_min_start = base_rows
        aromatic_max_start = base_rows + n_atoms
        aromatic_eligible = {6, 7, 8, 15, 16}
        cycle_edges = _candidate_cycle_edges(unique, n_atoms) if aromatic_constraints else set()
        for e, (i, j) in enumerate(unique):
            cols = np.arange(e * n_classes, (e + 1) * n_classes)
            constraints[e, cols] = 1.0
            lb[e] = ub[e] = 1.0
            constraints[m + 1 + n_atoms + i, cols] += VALENCE_BOND_ORDER
            constraints[m + 1 + n_atoms + j, cols] += VALENCE_BOND_ORDER
            if aromatic_constraints:
                aromatic_col = e * n_classes + 4
                constraints[aromatic_min_start + i, aromatic_col] += 1.0
                constraints[aromatic_min_start + j, aromatic_col] += 1.0
                constraints[aromatic_max_start + i, aromatic_col] += 1.0
                constraints[aromatic_max_start + j, aromatic_col] += 1.0
        for atom, z in enumerate(elems.detach().cpu().tolist()):
            begin = n_edge_vars + state_offsets[atom]
            end = begin + len(state_sets[atom])
            constraints[m + atom, begin:end] = 1.0
            lb[m + atom] = ub[m + atom] = 1.0
            for col, (_q, h, vmax) in zip(range(begin, end), state_sets[atom]):
                # Sum of decoded bond orders (+ implicit H in heavy-only mode)
                # is bounded by the selected charge-state valence capacity.
                constraints[m + 1 + n_atoms + atom, col] += float(h - vmax)
            ub[m + 1 + n_atoms + atom] = 0.0
            for col, (charge, _h, _vmax) in zip(range(begin, end), state_sets[atom]):
                constraints[m + n_atoms, col] = charge
        lb[m + n_atoms] = ub[m + n_atoms] = float(total_charge)

        if aromatic_constraints:
            # y_i indicates participation in an aromatic system.  A selected
            # aromatic edge forces both endpoints to y=1; a participating atom
            # must have 2--3 aromatic neighbours (2 for a simple ring, 3 for
            # a fused junction).  This rejects dangling/non-ring aromaticity.
            for atom in range(n_atoms):
                y_col = aromatic_start + atom
                constraints[aromatic_min_start + atom, y_col] = -2.0
                lb[aromatic_min_start + atom] = 0.0
                constraints[aromatic_max_start + atom, y_col] = -3.0
                ub[aromatic_max_start + atom] = 0.0

        # A no-good cut excludes one complete undirected edge labeling while
        # leaving charge/implicit-H states free.  Re-solving after a rejected
        # RDKit proposal therefore enumerates the next highest-scoring graph,
        # providing an open-source analogue of a solver solution pool.
        forbidden_start = base_rows + n_aromatic_rows
        for cut_id, labels_to_exclude in enumerate(forbidden):
            labels_array = np.asarray(labels_to_exclude, dtype=np.int64)
            if labels_array.shape != (m,):
                raise ValueError("forbidden edge labeling has incompatible shape")
            row = forbidden_start + cut_id
            for edge_id, cls in enumerate(labels_array.tolist()):
                if not 0 <= cls < n_classes:
                    raise ValueError("forbidden edge labeling contains invalid class")
                constraints[row, edge_id * n_classes + cls] = 1.0
            ub[row] = float(m - 1)

        objective = np.zeros(n_vars, dtype=np.float64)
        objective[:n_edge_vars] = -scores.reshape(-1)
        for atom, z in enumerate(elems.detach().cpu().tolist()):
            begin = n_edge_vars + state_offsets[atom]
            state_set = state_sets[atom]
            if charge_logits is not None:
                atom_logp = torch.log_softmax(charge_logits[atom].detach(), dim=-1).cpu().numpy()
                for col, (charge, h, _vmax) in zip(range(begin, begin + len(state_set)), state_set):
                    score = charge_weight * atom_logp[int(np.clip(charge, -2, 2)) + 2]
                    if implicit_h_logits is not None:
                        h_logp = torch.log_softmax(implicit_h_logits[atom].detach(), dim=-1).cpu().numpy()
                        score += implicit_h_weight * h_logp[h]
                    objective[col] = -score
            else:
                for col, (charge, h, _vmax) in zip(range(begin, begin + len(state_set)), state_set):
                    score = -charge_weight * abs(charge)
                    if implicit_h_logits is not None:
                        h_logp = torch.log_softmax(implicit_h_logits[atom].detach(), dim=-1).cpu().numpy()
                        score += implicit_h_weight * h_logp[h]
                    objective[col] = -score
        upper_bound = np.ones(n_vars, dtype=np.float64)
        if aromatic_constraints:
            atomic_numbers = elems.detach().cpu().tolist()
            for edge_id, (i, j) in enumerate(unique):
                if ((i, j) not in cycle_edges
                        or atomic_numbers[i] not in aromatic_eligible
                        or atomic_numbers[j] not in aromatic_eligible):
                    upper_bound[edge_id * n_classes + 4] = 0.0
        if forbid_aromatic:
            # Sanitization-aware fallback: retain the model's probability for
            # every other class but require an explicit single/double Kekule
            # hypothesis. This is only used after the normal aromatic proposal
            # failed, never as the primary decoder.
            upper_bound[np.arange(m) * n_classes + 4] = 0.0
        result = milp(
            c=objective, integrality=np.ones(n_vars), bounds=Bounds(0.0, upper_bound),
            constraints=LinearConstraint(constraints.tocsr(), lb, ub),
            options={"time_limit": float(time_limit)},
        )
        if not result.success or result.x is None:
            raise RuntimeError("joint MILP did not find a feasible solution")
        classes = result.x[:n_edge_vars].reshape(m, n_classes).argmax(axis=1)
        if aromatic_bridge_repair:
            classes = _repair_aromatic_bridges(unique, classes, scores)
        charge = np.zeros(n_atoms, dtype=np.int64)
        implicit_h = np.zeros(n_atoms, dtype=np.int64)
        for atom, z in enumerate(elems.detach().cpu().tolist()):
            begin = n_edge_vars + state_offsets[atom]
            choice = result.x[begin:begin + len(state_sets[atom])].argmax()
            charge[atom], implicit_h[atom], _ = state_sets[atom][int(choice)]
        kekulizable = _kekulizable(
            elems, unique, classes, charge, implicit_h,
            all_atoms_explicit_h=explicit_h,
        )
        labels = torch.empty(len(keys), dtype=torch.long, device=logits.device)
        for e, key in enumerate(unique):
            labels[rows[key]] = int(classes[e])
        return JointDecodeResult(
            labels, torch.tensor(charge, dtype=torch.long, device=logits.device),
            torch.tensor(implicit_h, dtype=torch.long, device=logits.device),
            True, kekulizable, float(-result.fun), local_overvalent,
        )
    except Exception:
        labels = torch.empty(len(keys), dtype=torch.long, device=logits.device)
        for e, key in enumerate(unique):
            labels[rows[key]] = int(local[e])
        return JointDecodeResult(
            labels, torch.zeros(n_atoms, dtype=torch.long, device=logits.device),
            torch.zeros(n_atoms, dtype=torch.long, device=logits.device),
            False, False, float("-inf"), local_overvalent,
        )


@torch.enable_grad()
def project_coordinates(
    coord: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    elems: torch.Tensor,
    steps: int = 40,
    max_displacement: float = 0.15,
    bond_weight: float = 10.0,
    repulsion_weight: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """Project coordinates minimally towards the decoded local bond geometry.

    It minimizes displacement plus bond-length and close-contact penalties; it
    never assumes that a reference conformer is the unique correct coordinate
    label.  The result is intended as an optional post-processing proposal.
    """
    if coord.numel() == 0:
        return coord, {"rms_displacement": 0.0, "energy": 0.0}
    # Pairwise labels are directed in the caller; use one representative pair.
    keys, rows, unique, _ = _undirected_scores(
        torch.nn.functional.one_hot(labels, num_classes=6).float() * 20.0,
        edge_index,
    )
    pair_labels = torch.tensor([int(labels[rows[key][0]]) for key in unique], device=coord.device)
    pairs = torch.tensor(unique, dtype=torch.long, device=coord.device)
    original = coord.detach()
    delta = torch.zeros_like(original, requires_grad=True)
    opt = torch.optim.Adam((delta,), lr=0.02)
    order = torch.tensor(BOND_ORDER, dtype=coord.dtype, device=coord.device)
    # Generic covalent target: 1.54 A for single C-C-like bonds, shortened by
    # 0.18 A per added order.  It is deliberately a soft local prior.
    target = (1.72 - 0.18 * order[pair_labels]).clamp(0.95, 1.55)
    bond_mask = pair_labels > 0
    for _ in range(int(steps)):
        proposed = original + delta
        vec = proposed[pairs[:, 1]] - proposed[pairs[:, 0]]
        dist = vec.norm(dim=-1).clamp_min(1e-5)
        displacement = delta.square().mean()
        bond_loss = ((dist[bond_mask] - target[bond_mask]).square().mean()
                     if bool(bond_mask.any()) else displacement * 0.0)
        nonbond = ~bond_mask
        repulsion = (torch.relu(1.05 - dist[nonbond]).square().mean()
                     if bool(nonbond.any()) else displacement * 0.0)
        energy = displacement + bond_weight * bond_loss + repulsion_weight * repulsion
        opt.zero_grad(set_to_none=True)
        energy.backward()
        opt.step()
        with torch.no_grad():
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta.mul_((max_displacement / norm).clamp_max(1.0))
            delta.sub_(delta.mean(dim=0, keepdim=True))
    projected = (original + delta.detach()).to(coord.dtype)
    return projected, {
        "rms_displacement": float(delta.detach().square().mean().sqrt()),
        "energy": float(energy.detach()),
    }


class RepairabilityCalibrator:
    """Small Platt-style calibrator for accepting versus rejecting a repair."""

    def __init__(self):
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    @staticmethod
    def features(edge_logits: torch.Tensor, labels: torch.Tensor,
                 result: JointDecodeResult, projection: dict | None = None) -> torch.Tensor:
        prob = torch.softmax(edge_logits.detach(), dim=-1)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(-1).mean()
        selected = prob.gather(1, labels.view(-1, 1)).mean()
        return torch.tensor([
            float(entropy), float(selected), float(result.local_overvalent_atoms),
            float(not result.solved), float(not result.kekulizable),
            float((projection or {}).get("rms_displacement", 0.0)),
        ], dtype=torch.float32)

    def fit(self, x: torch.Tensor, success: torch.Tensor, steps: int = 300, lr: float = 0.05):
        x = x.float()
        y = success.float()
        self.weight = torch.zeros(x.shape[1], requires_grad=True)
        self.bias = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam((self.weight, self.bias), lr=lr)
        for _ in range(int(steps)):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ self.weight + self.bias, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        self.weight = self.weight.detach(); self.bias = self.bias.detach()
        return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("fit the repairability calibrator before prediction")
        return torch.sigmoid(x.float() @ self.weight + self.bias)
