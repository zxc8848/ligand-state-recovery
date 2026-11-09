"""Bond-axis length augmentation matched to real PDBBind pose geometry.

Why isotropic Gaussian noise is the wrong augmentation here
-----------------------------------------------------------
``GaussianNoiseAugment`` displaces every atom by an isotropic Gaussian.  That
perturbs all bonds roughly equally and symmetrically.  Real experimental /
docked poses do not distort that way.  Measured on the reference sets in this
repo (CCD ideal coordinates vs. PDBBind ligand poses, heavy atoms only):

    bond          CCD ideal sd    PDBBind sd    ratio    mean shift
    C-O single       0.039           0.061      x1.55      -0.003
    C=O double       0.005           0.034      x6.36      +0.025
    C-C single       0.023           0.041      x1.77      -0.001
    C=C double       0.028           0.076      x2.74      +0.055
    C-N single       0.052           0.066      x1.27      -0.001
    C=N double       0.018           0.049      x2.77      +0.030

Double bonds are hit five to six times harder than single bonds, and they drift
*longer* -- toward single-bond territory.  The consequence is visible in class
separability: in CCD ideal geometry essentially no double bond falls inside the
single-bond length range, whereas in PDBBind poses 20% (C-O), 27% (C-C) and
56% (C-N) of doubles do.

This is exactly the observed failure mode.  On the 488-molecule pose set the
six-class refiner recovers single bonds at 0.968 recall and aromatic at 0.997,
but **double bonds at only 0.710** -- while scoring 0.995 double F1 on its own
in-distribution test set.  The model learned "C=O means length 1.214 +- 0.005",
a discriminator that simply does not survive contact with real poses.

Training with more isotropic noise cannot fix this, because isotropic noise
widens single and double bonds alike and so never teaches the model that a
1.27 A C-O bond can still be a double bond.

What this augmentation does
---------------------------
For each true bond, both endpoints are displaced *along the bond axis* by a
sampled amount, changing the bond length while leaving the rest of the local
geometry far more intact than isotropic jitter would.  The per-bond-order
distributions are taken from the table above, so the augmented training
distribution reproduces the measured pose distortion rather than a generic one.

The displacement is applied symmetrically (each endpoint moves by half the
delta in opposite directions) so the bond midpoint, and therefore the molecule's
centre of mass, is preserved.

Labels are unchanged: a stretched double bond is still a double bond.  That is
the entire point -- it is the supervision signal that teaches length tolerance.

Caveats to keep in mind when reporting
--------------------------------------
* This uses ground-truth bond orders to decide how much to stretch, so it is a
  training-time-only augmentation.  It leaks nothing at inference.
* Because bonds share atoms, sequential per-bond displacement is not exactly
  achievable; deltas are accumulated per atom and applied once, so the realised
  length change is approximate for atoms in dense environments. Use
  ``verify_realised_shift`` to measure how close the realisation lands.
* Stretching doubles toward single-bond lengths necessarily makes the two
  classes less separable *in training*. In-distribution double F1 is expected to
  drop somewhat. That is a real trade, and it must be checked on both the
  in-distribution test split and the pose set rather than assumed to be free.
"""

from __future__ import annotations

from typing import Dict

import torch

# (mean shift in A, extra sd in A) per bond class, derived from the
# CCD-ideal vs PDBBind-pose comparison documented above. Values are the
# additional distortion to inject on top of already-ideal geometry.
_DEFAULT_SHIFT = {
    1: (0.000, 0.030),   # single: modest broadening only
    2: (0.035, 0.045),   # double: systematic lengthening + large broadening
    3: (0.010, 0.020),   # triple: rare; keep conservative
    4: (0.005, 0.030),   # aromatic: behaves close to single
    5: (0.020, 0.035),   # delocalized: between single and double
}


class BondAxisLengthAugment:
    """Stretch/compress true bonds along their axis with class-dependent stats.

    Args:
        shift_table: {bond_class: (mean_shift_A, sd_A)}. Defaults to the
            measured CCD-ideal -> PDBBind-pose distortion.
        scale: Global multiplier on both mean shift and sd. ``scale=0``
            disables the augmentation, which makes ablation trivial.
        apply_prob: Probability of augmenting a given molecule at all.
        cutoff: Neighbour cutoff used when recomputing edge features.
    """

    def __init__(
        self,
        shift_table: Dict[int, tuple] | None = None,
        scale: float = 1.0,
        apply_prob: float = 0.5,
        cutoff: float = 3.0,
    ):
        self.shift_table = dict(shift_table or _DEFAULT_SHIFT)
        self.scale = float(scale)
        self.apply_prob = float(apply_prob)
        self.cutoff = float(cutoff)

    def _atom_displacement(self, coord, edge_index, labels, true_mask, generator=None):
        """Accumulate per-atom displacement from each true bond's axial delta."""
        displacement = torch.zeros_like(coord)
        if edge_index.numel() == 0:
            return displacement

        src, dst = edge_index[:, 0], edge_index[:, 1]
        # Each undirected bond appears twice; handle i<j once to avoid
        # double-counting the same physical bond.
        forward = true_mask & (src < dst)
        if not bool(forward.any()):
            return displacement

        idx = forward.nonzero(as_tuple=True)[0]
        i, j = src[idx], dst[idx]
        classes = labels[idx]

        delta = torch.zeros(len(idx), device=coord.device, dtype=coord.dtype)
        for bond_class, (mean_shift, sd) in self.shift_table.items():
            selected = classes == bond_class
            if not bool(selected.any()):
                continue
            count = int(selected.sum())
            noise = torch.randn(count, device=coord.device, dtype=coord.dtype,
                                generator=generator)
            delta[selected] = self.scale * (mean_shift + sd * noise)

        axis = coord[j] - coord[i]
        length = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = axis / length
        half = 0.5 * delta.unsqueeze(-1) * unit
        # Move endpoints apart (or together) about the bond midpoint.
        displacement.index_add_(0, i, -half)
        displacement.index_add_(0, j, half)
        return displacement

    def __call__(self, sample: Dict, generator=None) -> Dict:
        if self.scale == 0.0:
            return sample
        if self.apply_prob < 1.0 and float(torch.rand((), generator=generator)) >= self.apply_prob:
            return sample

        required = ("coord", "edge_index", "bond_labels", "true_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(
                f"BondAxisLengthAugment needs {missing} in the sample; it must run where "
                "ground-truth bond classes are available (training only)."
            )

        coord = sample["coord"]
        displacement = self._atom_displacement(
            coord, sample["edge_index"], sample["bond_labels"], sample["true_mask"], generator
        )
        sample["coord"] = coord + displacement
        _recompute_edges(sample)
        return sample


def _recompute_edges(sample: Dict) -> None:
    """Refresh edge_diff / edge_dist after coordinates move."""
    coord, edge_index = sample["coord"], sample["edge_index"]
    if edge_index.numel() == 0:
        return
    diff = coord[edge_index[:, 1]] - coord[edge_index[:, 0]]
    sample["edge_diff"] = diff
    sample["edge_dist"] = diff.norm(dim=-1)


def verify_realised_shift(before, after, edge_index, labels, true_mask):
    """Mean realised length change per bond class, for sanity-checking.

    Because bonds share atoms, the requested per-bond delta is only approximately
    realised. This reports what actually happened so the augmentation can be
    tuned against the target table instead of trusted blindly.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    forward = true_mask & (src < dst)
    idx = forward.nonzero(as_tuple=True)[0]
    i, j = src[idx], dst[idx]
    length_before = (before[j] - before[i]).norm(dim=-1)
    length_after = (after[j] - after[i]).norm(dim=-1)
    change = length_after - length_before
    out = {}
    for bond_class in labels[idx].unique().tolist():
        selected = labels[idx] == bond_class
        out[int(bond_class)] = {
            "n": int(selected.sum()),
            "mean_change": float(change[selected].mean()),
            "sd_change": float(change[selected].std()) if int(selected.sum()) > 1 else 0.0,
        }
    return out
