"""
Stage 2 bond-type GNN.

This model runs on the bonded topology predicted by Stage 1. Hydrogens can stay
as ordinary nodes in the message graph while readout is restricted to the bonds
the caller asks it to classify.
"""

import torch
from torch import nn

from .painn_backbone import PainnMessage, PainnUpdate, sinc_expansion
from ..utils.local_geometry import compute_local_geometry_features

_NORM_EPS = 1e-8       # used for float32 safe_norm
_SINC_EPS = 1e-4       # used for sinc division — must be representable in float16
                       # (float16 min normal ~6e-5; 1e-8 underflows to 0 under AMP)


def safe_norm(x: torch.Tensor, dim, keepdim: bool = False, eps: float = _NORM_EPS):
    """Numerically stable L2 norm with an epsilon inside the square root."""
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def sanitize_hidden(x: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    """Keep PaiNN hidden states finite under aggressive coordinate noise."""
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp_(
        min=-limit,
        max=limit,
    )


def limit_vector_norm(node_vector: torch.Tensor, limit: float) -> torch.Tensor:
    """Clip each equivariant vector channel by its 3D norm."""
    if limit <= 0.0 or node_vector.numel() == 0:
        return node_vector
    vec_norm = safe_norm(node_vector, dim=1, keepdim=True)
    scale = (limit / vec_norm.clamp_min(_NORM_EPS)).clamp_max(1.0)
    return node_vector * scale



class BondTypeGNN(nn.Module):
    """Bond type classifier on a bonded molecular topology."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 3,
        edge_embedding_size: int = 16,
        bond_cutoff: float = 3.0,
        num_bond_types: int = 4,
        dropout: float = 0.1,
        vector_norm_limit=3,
        random_node_feat_dim: int = 0,
        random_node_feat_std: float = 1.0,
        readout_extra_dim: int = 0,
        readout_feature_mode: str = "none",
        predict_formal_charge: bool = False,
        predict_delocalized_center: bool = False,
        predict_implicit_h: bool = False,
        predict_hybridization: bool = False,
        total_charge_condition: bool = False,
        backbone: str = "painn",
    ):
        super().__init__()
        # Retain this keyword solely to load existing PaiNN checkpoints whose
        # saved training configuration contains ``backbone='painn'``.
        if str(backbone).lower() != "painn":
            raise ValueError("BondNet 2.0 supports the PaiNN refiner only")
        self.backbone = "painn"
        self.bond_cutoff = bond_cutoff
        self.edge_embedding_size = edge_embedding_size
        self.random_node_feat_dim = random_node_feat_dim
        self.random_node_feat_std = random_node_feat_std
        self.readout_extra_dim = int(readout_extra_dim)
        self.readout_feature_mode = str(readout_feature_mode).lower()
        self.predict_formal_charge = bool(predict_formal_charge)
        self.predict_delocalized_center = bool(predict_delocalized_center)
        self.predict_implicit_h = bool(predict_implicit_h)
        self.predict_hybridization = bool(predict_hybridization)
        self.total_charge_condition = bool(total_charge_condition)
        if self.readout_feature_mode not in {"none", "vector", "angle", "vector_geometry"}:
            raise ValueError(
                "readout_feature_mode must be one of: none, vector, angle, vector_geometry"
            )

        self.vector_norm_limit = vector_norm_limit

        self.atom_embedding = nn.Embedding(119, hidden_size)
        # The molecular net charge is available in essentially every
        # charge-conditioned 3D diffusion model.  It is broadcast to all atoms
        # in that generated sample, never leaking reference local charges.
        self.total_charge_embedding = (
            nn.Embedding(9, hidden_size) if self.total_charge_condition else None
        )
        self.random_node_proj = (
            nn.Linear(random_node_feat_dim, hidden_size, bias=False)
            if random_node_feat_dim > 0 else None
        )

        self.msg_layers = nn.ModuleList([
            PainnMessage(hidden_size, edge_embedding_size, bond_cutoff)
            for _ in range(num_layers)
        ])
        self.upd_layers = nn.ModuleList([
            PainnUpdate(hidden_size) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # ``vector`` adds eight rotation-invariant summaries of the final PaiNN
        # vector channels.  ``angle`` adds endpoint geometry descriptors
        # [planarity, mean-angle-cosine, degree] for both atoms.
        # ``vector_geometry`` deliberately retains the existing vector PaiNN
        # signal and appends those local descriptors; it is the learned
        # counterpart of Knodle's planarity/angle checks, not a hard rule.
        mode_dim = {"none": 0, "vector": 8, "angle": 6, "vector_geometry": 14}[self.readout_feature_mode]
        edge_in = 2 * hidden_size + edge_embedding_size + self.readout_extra_dim + mode_dim
        self.cls_head = nn.Sequential(
            nn.Linear(edge_in, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, num_bond_types),
        )
        # Atom-state heads are auxiliary during training and become inputs to
        # the constrained decoder at inference. Formal charges are clipped to
        # {-2,-1,0,+1,+2}; the central class is therefore index 2.
        self.charge_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                          nn.Linear(hidden_size // 2, 5))
            if self.predict_formal_charge else None
        )
        self.delocalized_center_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                          nn.Linear(hidden_size // 2, 1))
            if self.predict_delocalized_center else None
        )
        self.implicit_h_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                          nn.Linear(hidden_size // 2, 5))
            if self.predict_implicit_h else None
        )
        # Hybridization labels are SP, SP2, SP3, aromatic.  The resulting
        # probabilities supply a *soft*, learned compatibility contribution to
        # every edge class rather than a decoder constraint.  This keeps valid
        # unusual chemistry representable while making sp2/aromatic evidence
        # available to the bond classifier.
        self.hybridization_head = (
            nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                          nn.Linear(hidden_size // 2, 4))
            if self.predict_hybridization else None
        )
        self.hybrid_bond_compat = (
            nn.Parameter(torch.zeros(4, 4, num_bond_types))
            if self.predict_hybridization else None
        )

    def forward(
        self,
        elems: torch.Tensor,
        bond_edge_index: torch.Tensor,
        bond_edge_diff: torch.Tensor,
        bond_edge_dist: torch.Tensor,
        readout_edge_index: torch.Tensor = None,
        readout_edge_diff: torch.Tensor = None,
        readout_edge_dist: torch.Tensor = None,
        readout_extra: torch.Tensor = None,
        coord: torch.Tensor = None,
        return_aux: bool = False,
        molecular_total_charge: torch.Tensor = None,
        atom_mol_index: torch.Tensor = None,
    ) -> torch.Tensor:
        if readout_edge_index is None:
            readout_edge_index = bond_edge_index
            readout_edge_diff = bond_edge_diff
            readout_edge_dist = bond_edge_dist
        if readout_edge_index.shape[0] == 0:
            return torch.zeros(0, self.cls_head[-1].out_features, device=elems.device)

        s = self.atom_embedding(elems)
        if self.total_charge_embedding is not None:
            if molecular_total_charge is None or atom_mol_index is None:
                raise ValueError("molecular_total_charge and atom_mol_index are required for charge conditioning")
            q = molecular_total_charge.clamp(-4, 4).long() + 4
            s = s + self.total_charge_embedding(q[atom_mol_index])
        if self.random_node_proj is not None:
            rand_feat = torch.randn(
                elems.shape[0],
                self.random_node_feat_dim,
                device=elems.device,
                dtype=s.dtype,
            ) * self.random_node_feat_std
            s = s + self.random_node_proj(rand_feat)
        v = torch.zeros(s.shape[0], 3, s.shape[1], device=s.device, dtype=s.dtype)

        if bond_edge_index.shape[0] > 0:
            for msg, upd, norm in zip(self.msg_layers, self.upd_layers, self.norms):
                s, v = msg(s, v, bond_edge_index, bond_edge_diff, bond_edge_dist)
                # s, v = upd(s, v)
                s = norm(s)
                v = v.clamp(-1e4, 1e4)   # prevent vector feature explosion across layers

                s= sanitize_hidden(s)
                v = sanitize_hidden(v)
                s,v = upd(s,v)
                s= sanitize_hidden(s)
                v = sanitize_hidden(v)
                v = limit_vector_norm(v, self.vector_norm_limit)


        s_i = s[readout_edge_index[:, 0]]
        s_j = s[readout_edge_index[:, 1]]
        rbf = sinc_expansion(readout_edge_dist, self.edge_embedding_size, self.bond_cutoff)
        parts = [s_i, s_j, rbf]
        if self.readout_feature_mode in {"vector", "vector_geometry"}:
            # All quantities below are invariant to a global rotation: norms,
            # channelwise vector dot products, and projections on the bond axis.
            vi = v[readout_edge_index[:, 0]]
            vj = v[readout_edge_index[:, 1]]
            rhat = readout_edge_diff / readout_edge_dist.clamp_min(_NORM_EPS).unsqueeze(-1)
            ni = safe_norm(vi, dim=1)
            nj = safe_norm(vj, dim=1)
            dot = (vi * vj).sum(dim=1)
            proj_i = (vi * rhat.unsqueeze(-1)).sum(dim=1)
            proj_j = (vj * rhat.unsqueeze(-1)).sum(dim=1)
            vector_extra = torch.stack(
                (
                    ni.mean(-1), ni.std(-1, unbiased=False),
                    nj.mean(-1), nj.std(-1, unbiased=False),
                    dot.mean(-1), dot.std(-1, unbiased=False),
                    0.5 * (proj_i.mean(-1) + proj_j.mean(-1)),
                    0.5 * (proj_i.std(-1, unbiased=False) + proj_j.std(-1, unbiased=False)),
                ),
                dim=-1,
            )
            parts.append(vector_extra)
        if self.readout_feature_mode in {"angle", "vector_geometry"}:
            if coord is None:
                raise ValueError("coord is required for geometry-aware readout")
            local_geom = compute_local_geometry_features(
                coord, bond_edge_index, bond_edge_dist, self.bond_cutoff
            )
            parts.append(torch.cat((
                local_geom[readout_edge_index[:, 0]],
                local_geom[readout_edge_index[:, 1]],
            ), dim=-1).to(dtype=s.dtype))
        if self.readout_extra_dim:
            if readout_extra is None:
                readout_extra = torch.zeros(
                    readout_edge_index.shape[0],
                    self.readout_extra_dim,
                    device=s.device,
                    dtype=s.dtype,
                )
            if readout_extra.shape != (readout_edge_index.shape[0], self.readout_extra_dim):
                raise ValueError(
                    'readout_extra must have shape '
                    f'({readout_edge_index.shape[0]}, {self.readout_extra_dim}), '
                    f'got {tuple(readout_extra.shape)}'
                )
            parts.append(readout_extra.to(dtype=s.dtype))
        logits = self.cls_head(torch.cat(parts, dim=-1))
        hybrid_logits = None
        if self.hybridization_head is not None:
            hybrid_logits = self.hybridization_head(s)
            hybrid_prob = torch.softmax(hybrid_logits, dim=-1)
            # Symmetrising makes an undirected chemical compatibility table
            # independent of the arbitrary readout-edge orientation.
            compat_table = 0.5 * (
                self.hybrid_bond_compat + self.hybrid_bond_compat.transpose(0, 1)
            )
            p_i = hybrid_prob[readout_edge_index[:, 0]]
            p_j = hybrid_prob[readout_edge_index[:, 1]]
            logits = logits + torch.einsum("ei,ijk,ej->ek", p_i, compat_table, p_j)
        if not return_aux:
            return logits
        aux = {}
        if self.charge_head is not None:
            aux["charge_logits"] = self.charge_head(s)
        if self.delocalized_center_head is not None:
            aux["delocalized_center_logits"] = self.delocalized_center_head(s).squeeze(-1)
        if self.implicit_h_head is not None:
            aux["implicit_h_logits"] = self.implicit_h_head(s)
        if hybrid_logits is not None:
            aux["hybridization_logits"] = hybrid_logits
        return logits, aux


def compute_bonded_geometry(coord: torch.Tensor, bond_edge_index: torch.Tensor):
    diff = coord[bond_edge_index[:, 1]] - coord[bond_edge_index[:, 0]]
    dist = diff.norm(dim=-1)
    return diff, dist
