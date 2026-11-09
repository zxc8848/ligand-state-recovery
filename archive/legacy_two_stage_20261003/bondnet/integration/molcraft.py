"""Generator-agnostic XYZ adapter used by the MolCraftDiffusion integration.

MolCraftDiffusion writes one XYZ file per generated coordinate sample.  This
module turns that coordinate-only output into the tensor interface consumed by
the BondNet refiner without depending on MolCraft's Python environment.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..model.bond_type_gnn import compute_bonded_geometry
from ..model.bondnet import _symmetrize_logits


def read_xyz(path: str | Path, keep_hydrogens: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a standard XYZ file, optionally dropping explicit hydrogens.

    ``keep_hydrogens=True`` is the all-atom mode for coordinate generators
    such as the default MolCraft EDM checkpoint.  ``False`` preserves the
    earlier heavy-atom-only bridge for generators that do not emit H atoms.
    """
    from rdkit import Chem

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {path}") from exc
    if len(lines) < n_atoms + 2:
        raise ValueError(f"Truncated XYZ file: {path}")
    periodic_table = Chem.GetPeriodicTable()
    elems, coord = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ row in {path}: {line!r}")
        z = int(periodic_table.GetAtomicNumber(parts[0]))
        if z <= 0:
            raise ValueError(f"Unknown XYZ element {parts[0]!r} in {path}")
        if z == 1 and not keep_hydrogens:
            continue
        elems.append(z)
        coord.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not elems:
        raise ValueError(f"No atoms remain after reading {path}")
    return torch.tensor(elems, dtype=torch.long), torch.tensor(coord, dtype=torch.float32)


def read_heavy_xyz(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible heavy-atom XYZ reader."""
    return read_xyz(path, keep_hydrogens=False)


def radius_knn_candidate_edges(
    coord: torch.Tensor,
    radius: float = 3.0,
    knn: int = 12,
    max_neighbors: int = 15,
) -> torch.Tensor:
    """Build a symmetric Radius ∪ kNN graph with a bounded degree.

    This is intentionally more conservative than a Delaunay-only graph: it
    retains spatially close candidates, recovers stretched local neighbours,
    and remains stable when a diffusion sample contains atom collisions.
    """
    n_atoms = int(coord.shape[0])
    if n_atoms < 2:
        return torch.empty((0, 2), dtype=torch.long, device=coord.device)
    distance = torch.cdist(coord, coord)
    candidates: set[tuple[int, int]] = set()
    for i in range(n_atoms):
        nearest = torch.argsort(distance[i])
        for j in nearest[1:1 + min(int(knn), n_atoms - 1)].tolist():
            candidates.add((min(i, int(j)), max(i, int(j))))
        # ``torch.cdist`` can return a tiny positive floating-point value on
        # its diagonal.  Exclude the diagonal explicitly: relying on
        # ``distance > 0`` can otherwise create a self-loop in sparse
        # radius-only graphs, which later reaches RDKit as an illegal bond.
        in_radius = (distance[i] <= radius) & (distance[i] > 0)
        in_radius[i] = False
        for j in torch.where(in_radius)[0].tolist():
            candidates.add((min(i, int(j)), max(i, int(j))))
    # A global nearest-first pass ensures the degree cap holds at both ends.
    degree = [0] * n_atoms
    kept: list[tuple[int, int]] = []
    for i, j in sorted(candidates, key=lambda pair: float(distance[pair[0], pair[1]])):
        if degree[i] >= max_neighbors or degree[j] >= max_neighbors:
            continue
        kept.append((i, j))
        degree[i] += 1
        degree[j] += 1
    if not kept:
        return torch.empty((0, 2), dtype=torch.long, device=coord.device)
    undirected = torch.tensor(kept, dtype=torch.long, device=coord.device)
    return torch.cat((undirected, undirected.flip(-1)), dim=0)


def all_atom_candidate_edges(
    elems: torch.Tensor,
    coord: torch.Tensor,
    radius: float = 3.0,
    knn: int = 12,
    max_neighbors: int = 15,
    h_cutoff: float = 2.0,
) -> torch.Tensor:
    """Build the explicit-H candidate graph used by the training featurizer.

    Heavy-heavy pairs use the robust Radius ∪ kNN graph.  Hydrogen nodes only
    connect to nearby heavy atoms; H-H edges are never candidates.  This keeps
    all generated H coordinates available to the decoder without allowing a
    combinatorial cloud of physically meaningless H-H assignments.
    """
    device = coord.device
    heavy_idx = torch.where(elems != 1)[0]
    h_idx = torch.where(elems == 1)[0]
    parts: list[torch.Tensor] = []
    if heavy_idx.numel() >= 2:
        local = radius_knn_candidate_edges(
            coord[heavy_idx], radius=radius, knn=knn, max_neighbors=max_neighbors,
        )
        if local.numel():
            parts.append(heavy_idx[local])
    if heavy_idx.numel() and h_idx.numel():
        distance = torch.cdist(coord[h_idx], coord[heavy_idx])
        h_local, heavy_local = torch.where(distance <= float(h_cutoff))
        if h_local.numel():
            h_to_heavy = torch.stack((h_idx[h_local], heavy_idx[heavy_local]), dim=-1)
            parts.extend((h_to_heavy, h_to_heavy.flip(-1)))
    if not parts:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)


def make_coordinate_batch(
    elems: torch.Tensor,
    coord: torch.Tensor,
    total_charge: int,
    radius: float = 3.0,
    knn: int = 12,
    max_neighbors: int = 15,
    candidate_edge_index: torch.Tensor | None = None,
) -> dict:
    edge_index = (candidate_edge_index if candidate_edge_index is not None
                  else radius_knn_candidate_edges(coord, radius, knn, max_neighbors))
    edge_diff, edge_dist = compute_bonded_geometry(coord, edge_index)
    return {
        "elems": elems,
        "coord": coord,
        "edge_index": edge_index,
        "edge_diff": edge_diff,
        "edge_dist": edge_dist,
        "train_edge_mask": torch.ones(edge_index.shape[0], dtype=torch.bool, device=coord.device),
        "num_atoms_per_mol": torch.tensor([int(elems.numel())], dtype=torch.long, device=coord.device),
        "molecular_total_charge": torch.tensor([int(total_charge)], dtype=torch.long, device=coord.device),
    }


def _build_refiner_batch(stage1, batch: dict, args,
                         refiner_heavy_only: bool = False) -> dict:
    """Stage-1-to-refiner bridge that needs no reference bond labels.

    Existing structured refiner checkpoints were trained to read out only
    heavy-heavy edges.  In explicit-H mode, H-X connectivity is instead
    supplied by the Stage-1 head that was trained on those candidate edges.
    Keeping this split at inference avoids asking the refiner for logits on a
    supervision regime it has never seen.
    """
    _, _, edge_feat = stage1.backbone(
        batch["elems"], batch["coord"], batch["edge_index"],
        batch["edge_diff"], batch["edge_dist"],
    )
    conn_logits = stage1.stage1(edge_feat)
    conn_prob = torch.sigmoid(conn_logits)
    edge_index = batch["edge_index"]
    train_mask = batch["train_edge_mask"].bool()
    # Single-stage inference must mirror single-stage training: full candidate
    # message graph, Stage-1 connectivity feature zeroed. Gated behind an
    # explicit flag so the default two-stage path is untouched.
    single_stage = bool(getattr(args, "single_stage", False))
    if single_stage:
        elems_all = batch["elems"]
        hh = (elems_all[edge_index[:, 0]] != 1) & (elems_all[edge_index[:, 1]] != 1)
        message_mask = train_mask & hh
        conn_logits = torch.zeros_like(conn_logits)
        conn_prob = torch.zeros_like(conn_prob)
    else:
        threshold = float(getattr(args, "conn_threshold", 0.5))
        message_mask = (conn_prob >= threshold) & train_mask
        retain_threshold = getattr(args, "retain_threshold", None)
        if retain_threshold is not None:
            message_mask |= (conn_prob >= float(retain_threshold)) & train_mask
    message_ei = edge_index[message_mask]
    message_diff, message_dist = compute_bonded_geometry(batch["coord"], message_ei)
    readout_mask = train_mask
    if refiner_heavy_only:
        elems = batch["elems"]
        readout_mask = readout_mask & (elems[edge_index[:, 0]] != 1) & (elems[edge_index[:, 1]] != 1)
    readout_ei = edge_index[readout_mask]
    readout_diff, readout_dist = compute_bonded_geometry(batch["coord"], readout_ei)
    extra = torch.stack((
        conn_logits[readout_mask].clamp(-8, 8) / 8.0,
        conn_prob[readout_mask],
    ), dim=-1)
    return {
        "message_ei": message_ei,
        "message_diff": message_diff,
        "message_dist": message_dist,
        "readout_ei": readout_ei,
        "readout_diff": readout_diff,
        "readout_dist": readout_dist,
        "extra": extra,
        "readout_conn_prob": conn_prob[readout_mask],
        "readout_mask": readout_mask,
        "all_conn_logits": conn_logits,
        "all_conn_prob": conn_prob,
    }


@torch.no_grad()
def build_refiner_inference_batch(stage1, batch: dict, args,
                                 refiner_heavy_only: bool = False) -> dict:
    """No-grad Stage-1-to-refiner bridge used by ordinary inference."""
    return _build_refiner_batch(stage1, batch, args, refiner_heavy_only)


def build_refiner_differentiable_batch(stage1, batch: dict, args,
                                      refiner_heavy_only: bool = False) -> dict:
    """Differentiable Stage-1-to-refiner bridge for coordinate auxiliary loss.

    Callers must supply a *fixed* candidate edge index, normally built from the
    clean training conformer.  The connectivity threshold still selects a
    discrete message-passing topology, but gradients flow through all selected
    geometric features and logits back to ``batch['coord']``.  This function
    is intentionally separate from inference to make that piecewise-constant
    topology choice explicit.
    """
    return _build_refiner_batch(stage1, batch, args, refiner_heavy_only)


def forward_coordinate_refiner(model, batch: dict, built: dict, return_aux: bool = True):
    atom_mol_index = torch.zeros(batch["elems"].shape[0], dtype=torch.long, device=batch["elems"].device)
    output = model(
        batch["elems"],
        built["message_ei"], built["message_diff"], built["message_dist"],
        built["readout_ei"], built["readout_diff"], built["readout_dist"],
        readout_extra=built["extra"], coord=batch["coord"], return_aux=return_aux,
        molecular_total_charge=batch.get("molecular_total_charge"),
        atom_mol_index=atom_mol_index if model.total_charge_condition else None,
    )
    if not return_aux:
        return _symmetrize_logits(output, built["readout_ei"])
    logits, aux = output
    return _symmetrize_logits(logits, built["readout_ei"]), aux


def stage1_hydrogen_logits(conn_logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Map Stage-1 H-X connectivity scores to six structured bond classes.

    Hydrogen can only form a single covalent bond in the supported organic
    element vocabulary.  We therefore retain Stage-1's learned binary evidence
    for ``nobond`` versus ``single`` and rule out double/triple/aromatic/
    delocalized H-X labels.  The result is symmetric over directed edge pairs.
    """
    logits = torch.full((conn_logits.shape[0], 6), -20.0,
                        dtype=conn_logits.dtype, device=conn_logits.device)
    logits[:, 0] = -conn_logits
    logits[:, 1] = conn_logits
    return _symmetrize_logits(logits, edge_index)


def uncertainty_error_mask(logits: torch.Tensor, edge_index: torch.Tensor, n_atoms: int,
                           max_atoms: int = 8) -> list[int]:
    """Return high-uncertainty atoms for a downstream inpainting mask.

    This is deliberately an error *proposal*, not a claim that the selected
    atoms are ground-truth errors.  It is useful for an initial local resampling
    experiment and can later be calibrated against real diffusion trajectories.
    """
    if logits.numel() == 0:
        return list(range(min(n_atoms, max_atoms)))
    prob = torch.softmax(logits.detach(), dim=-1)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(-1)
    score = torch.zeros(n_atoms, dtype=entropy.dtype, device=entropy.device)
    count = torch.zeros_like(score)
    score.index_add_(0, edge_index[:, 0], entropy)
    count.index_add_(0, edge_index[:, 0], torch.ones_like(entropy))
    score = score / count.clamp_min(1)
    k = min(int(max_atoms), n_atoms)
    return torch.topk(score, k=k).indices.detach().cpu().tolist()
