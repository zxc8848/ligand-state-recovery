"""
Gaussian noise augmentation for BondNet training.

Adds isotropic Gaussian noise to atom coordinates and recomputes
edge_diff and edge_dist in-place.  edge_index (connectivity topology)
is unchanged — only the geometric quantities are updated.
"""

import torch
from typing import Dict, Optional

from ..utils.local_geometry import compute_local_geometry_features


class GaussianNoiseAugment:
    """
    Apply Gaussian noise to 3D atom coordinates.

    Args:
        sigma:      Noise standard deviation (Å).  σ=0.05 simulates typical
                    EDM/GeoLDM generation error.
        sigma_min:  If provided and sigma_max is also set, σ is sampled
                    uniformly in [sigma_min, sigma_max] per molecule.
        sigma_max:  Upper bound for random σ sampling.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        cutoff: float = 5.0,
        clean_prob: float = 0.0,
    ):
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cutoff = cutoff
        self.clean_prob = clean_prob

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply noise augmentation to a sample dict in-place.

        Args:
            sample: Feature dict returned by MoleculeFeaturizer.featurize().
                    Must contain 'coord', 'edge_index', 'edge_diff', 'edge_dist'.

        Returns:
            The same dict with 'coord', 'edge_diff', 'edge_dist' updated.
        """
        if self.clean_prob > 0.0 and torch.rand(()) < self.clean_prob:
            return sample

        coord = sample['coord']  # (N, 3)

        # Sample σ from range if specified; otherwise use fixed sigma
        sigma = self.sigma
        if self.sigma_min is not None and self.sigma_max is not None:
            sigma = float(
                torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
            )

        # σ=0 → no-op (keeps clean coordinates in the training distribution)
        if sigma == 0.0:
            return sample

        # Perturb coordinates
        noise = torch.randn_like(coord) * sigma
        new_coord = coord + noise

        # Recompute geometric edge features
        edge_index = sample['edge_index']   # (E, 2)
        i_idx = edge_index[:, 0]
        j_idx = edge_index[:, 1]
        new_edge_diff = new_coord[j_idx] - new_coord[i_idx]
        new_edge_dist = new_edge_diff.norm(dim=-1)

        sample = dict(sample)  # shallow copy to avoid mutating original
        sample['coord'] = new_coord
        sample['edge_diff'] = new_edge_diff
        sample['edge_dist'] = new_edge_dist
        if 'local_geom' in sample:
            sample['local_geom'] = compute_local_geometry_features(
                new_coord,
                edge_index,
                new_edge_dist,
                self.cutoff,
            )
        return sample

    def augment_batch(
        self,
        batch: Dict,
        sigma: Optional[float] = None,
        per_molecule: bool = True,
        clean_prob: float = 0.0,
    ) -> Dict:
        """
        Augment a collated batch dict.  Uses the same σ for the whole batch.
        """
        coord = batch['coord']
        if per_molecule and 'num_atoms_per_mol' in batch:
            counts = batch['num_atoms_per_mol'].to(device=coord.device)
            n_mols = int(counts.shape[0])
            if self.sigma_min is not None and self.sigma_max is not None:
                sigmas = torch.empty(
                    n_mols, device=coord.device, dtype=coord.dtype
                ).uniform_(self.sigma_min, self.sigma_max)
            else:
                s = sigma if sigma is not None else self.sigma
                sigmas = torch.full(
                    (n_mols,), float(s), device=coord.device, dtype=coord.dtype
                )
            if clean_prob > 0.0:
                clean_mask = torch.rand(n_mols, device=coord.device) < clean_prob
                sigmas = sigmas.masked_fill(clean_mask, 0.0)
            atom_sigma = torch.repeat_interleave(sigmas, counts).unsqueeze(-1)
            noise = torch.randn_like(coord) * atom_sigma
        else:
            s = sigma if sigma is not None else self.sigma
            if clean_prob > 0.0 and torch.rand((), device=coord.device) < clean_prob:
                s = 0.0
            noise = torch.randn_like(coord) * s
        new_coord = coord + noise

        edge_index = batch['edge_index']
        i_idx = edge_index[:, 0]
        j_idx = edge_index[:, 1]
        new_edge_diff = new_coord[j_idx] - new_coord[i_idx]
        new_edge_dist = new_edge_diff.norm(dim=-1)

        batch = dict(batch)
        batch['coord'] = new_coord
        batch['edge_diff'] = new_edge_diff
        batch['edge_dist'] = new_edge_dist
        if 'local_geom' in batch:
            batch['local_geom'] = compute_local_geometry_features(
                new_coord,
                edge_index,
                new_edge_dist,
                self.cutoff,
            )
        return batch
