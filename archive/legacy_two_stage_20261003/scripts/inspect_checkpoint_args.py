#!/usr/bin/env python3
r"""Print the saved training args of a refiner checkpoint.

Used as a preflight before scripts/run_seed_variance.bat so the multi-seed COMMON
recipe can be confirmed against the released checkpoint before a multi-hour run.

Usage:
  python scripts/inspect_checkpoint_args.py checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt
"""

from __future__ import annotations

import sys

KEYS = [
    "num_layers", "edge_embedding_size", "bond_cutoff", "readout_feature_mode",
    "hard_negative_weight", "hard_negative_distance", "hard_negative_confidence",
    "retain_threshold", "topology_fp_rate", "topology_fn_rate", "teacher_message_prob",
    "focal_gamma", "lr", "weight_decay", "batch_size", "dropout",
    "noise_min", "noise_max", "noise_clean_prob", "total_charge_condition",
    "structured_six_class", "charge_aux_weight", "implicit_h_aux_weight", "epochs", "seed",
]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: inspect_checkpoint_args.py <checkpoint.pt>")
    import torch
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)
    print(f"saved training args for {sys.argv[1]}:")
    for k in KEYS:
        if k in args:
            print(f"  {k} = {args[k]}")
    extra = sorted(set(args) - set(KEYS))
    if extra:
        print("  (other saved keys: " + ", ".join(extra) + ")")


if __name__ == "__main__":
    main()
