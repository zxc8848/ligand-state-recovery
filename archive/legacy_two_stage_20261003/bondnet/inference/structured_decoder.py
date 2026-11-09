"""Global constrained decoder for the six-class structured bond refiner.

It selects one class per *undirected* candidate pair while enforcing upper bounds
on the total incident bond order of every atom.  This removes asymmetric directed
edge decisions and prevents locally plausible but globally over-valent graphs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


# 0=nobond, 1=single, 2=double, 3=triple, 4=aromatic, 5=delocalized
BOND_ORDER = np.asarray((0.0, 1.0, 2.0, 3.0, 1.5, 1.5), dtype=np.float64)
MAX_VALENCE = defaultdict(lambda: 4.0, {
    1: 1.0, 5: 3.0, 6: 4.0, 7: 4.0, 8: 3.0, 9: 1.0,
    14: 4.0, 15: 5.0, 16: 6.0, 17: 1.0, 35: 1.0, 53: 1.0,
})


def decode_valence_constrained(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    elems: torch.Tensor,
    repair_confidence_threshold: float = 0.80,
    charge_logits: torch.Tensor | None = None,
    implicit_h_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode directed six-class logits with a per-molecule valence ILP.

    Args:
        logits: (E, 6) directed class logits.
        edge_index: (E, 2) directed candidate pairs, for one molecule.
        elems: (N,) atomic numbers for that molecule.

    Returns:
        (E,) long class labels. If SciPy's MILP solver is unavailable or the
        optimization fails, it safely falls back to symmetric local argmax.
    """
    if logits.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=logits.device)
    if logits.shape[-1] != 6:
        raise ValueError("structured decoder expects six-class logits")

    keys = [tuple(sorted((int(i), int(j)))) for i, j in edge_index.detach().cpu().tolist()]
    groups = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    unique = list(groups)
    logp = torch.log_softmax(logits.detach(), dim=-1).cpu()
    scores = np.stack([logp[groups[key]].mean(dim=0).numpy() for key in unique])

    # Symmetric local prediction remains a robust fallback and is also used when
    # SciPy is intentionally absent on a lightweight inference installation.
    local = scores.argmax(axis=1)

    # The constrained objective must be a *repair*, not a gratuitous global
    # rewrite.  In particular, a highly confident local graph that already
    # satisfies every valence bound can lose exact-match accuracy when the ILP
    # trades a tiny log-probability difference across otherwise valid edges.
    local_valence = np.zeros(int(elems.numel()), dtype=np.float64)
    for cls, (i, j) in zip(local, unique):
        local_valence[i] += BOND_ORDER[cls]
        local_valence[j] += BOND_ORDER[cls]
    limits = np.asarray(
        [MAX_VALENCE[int(z)] for z in elems.detach().cpu().tolist()], dtype=np.float64
    )
    if charge_logits is not None:
        # Predicted charge selects a tighter, chemistry-aware limit only for
        # common charge-sensitive atoms.  The confidence gate below keeps this
        # auxiliary signal from overriding a confident local graph.
        charges = charge_logits.detach().argmax(dim=-1).cpu().numpy() - 2
        zs = elems.detach().cpu().numpy()
        for atom, (z, q) in enumerate(zip(zs, charges)):
            if z == 7:       # neutral N typically has BO sum <= 3; N+ <= 4
                limits[atom] = 4.0 if q > 0 else 3.0
            elif z == 8:     # O-, O, O+ permit 1, 2, 3 respectively
                limits[atom] = float(np.clip(2 + q, 1, 3))
    if implicit_h_logits is not None:
        h_count = implicit_h_logits.detach().argmax(dim=-1).cpu().numpy()
        # Bond-order capacity remaining after the predicted implicit hydrogens.
        # A confidence-gated repair below prevents an uncertain H prediction
        # from rewriting a confident heavy-atom graph.
        limits = np.maximum(limits - h_count, 0.0)
    selected_probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    selected_probability /= selected_probability.sum(axis=1, keepdims=True)
    bond_probability = selected_probability[np.arange(len(local)), local][local > 0]
    # Element-only valence limits cannot represent every formally charged or
    # hypervalent motif.  Do not overwrite a confident local graph merely
    # because it violates this deliberately coarse prior; activate the repair
    # only for an over-valent *and uncertain* prediction.
    confident = bond_probability.size == 0 or float(bond_probability.mean()) >= repair_confidence_threshold
    if np.all(local_valence <= limits + 1e-6) or confident:
        pred = torch.empty(len(keys), dtype=torch.long, device=logits.device)
        for edge_id, key in enumerate(unique):
            pred[groups[key]] = int(local[edge_id])
        return pred
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        m, n_classes = scores.shape
        n_atoms = int(elems.numel())
        n_vars = m * n_classes
        constraints = lil_matrix((m + n_atoms, n_vars), dtype=np.float64)
        lower = np.full(m + n_atoms, -np.inf, dtype=np.float64)
        upper = np.full(m + n_atoms, np.inf, dtype=np.float64)
        for edge_id, (i, j) in enumerate(unique):
            cols = np.arange(edge_id * n_classes, (edge_id + 1) * n_classes)
            constraints[edge_id, cols] = 1.0
            lower[edge_id] = upper[edge_id] = 1.0
            constraints[m + i, cols] += BOND_ORDER
            constraints[m + j, cols] += BOND_ORDER
        for atom, z in enumerate(elems.detach().cpu().tolist()):
            upper[m + atom] = MAX_VALENCE[int(z)]

        result = milp(
            c=-scores.reshape(-1), integrality=np.ones(n_vars),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(constraints.tocsr(), lower, upper),
            options={"time_limit": 2.0},
        )
        if result.success and result.x is not None:
            local = result.x.reshape(m, n_classes).argmax(axis=1)
    except Exception:
        pass

    pred = torch.empty(len(keys), dtype=torch.long, device=logits.device)
    for edge_id, key in enumerate(unique):
        pred[groups[key]] = int(local[edge_id])
    return pred


def decode_delocalized_oxy_groups(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    elems: torch.Tensor,
    pred: torch.Tensor,
    threshold: float = 0.20,
) -> torch.Tensor:
    """Jointly assign two terminal X--O edges to ``delocalized``.

    The transform is prediction-only: a C/N centre must have exactly two
    terminal O neighbours in the currently decoded graph, and both edges must
    carry non-trivial class-5 probability. It never uses reference bonds.
    """
    if logits.numel() == 0:
        return pred
    probs = torch.softmax(logits.detach(), dim=-1).cpu()
    keys = [tuple(sorted((int(i), int(j)))) for i, j in edge_index.detach().cpu().tolist()]
    rows = {}
    for idx, key in enumerate(keys):
        rows.setdefault(key, []).append(idx)
    pair_prob = {key: probs[idxs].mean(dim=0) for key, idxs in rows.items()}
    pair_label = {key: int(pred[idxs[0]]) for key, idxs in rows.items()}
    adj = {}
    for (i, j), label in pair_label.items():
        if label > 0:
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)
    out = pred.clone()
    z = elems.detach().cpu().tolist()
    for center, nbrs in adj.items():
        if int(z[center]) not in (6, 7):
            continue
        terminal_o = [o for o in nbrs if int(z[o]) == 8 and len(adj.get(o, ())) == 1]
        if len(terminal_o) != 2:
            continue
        group_keys = [tuple(sorted((center, o))) for o in terminal_o]
        if min(float(pair_prob[key][5]) for key in group_keys) < threshold:
            continue
        for key in group_keys:
            out[rows[key]] = 5
    return out
