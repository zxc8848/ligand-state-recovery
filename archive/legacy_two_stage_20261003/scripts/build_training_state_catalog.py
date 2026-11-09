"""Derive decoder atom-state capacities from the training split only.

The catalogue contains one maximum observed total-valence capacity for each
``(atomic number, formal charge)`` pair.  Total valence is computed from the
reference heavy-atom bond labels plus the stored implicit-hydrogen count.  The
held-out validation and test splits are used only to report catalogue coverage;
they never change the catalogue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


BOND_CAPACITY = {0: 1, 1: 2, 2: 3, 3: 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit_split", default="train")
    parser.add_argument("--min_count", type=int, default=1)
    return parser.parse_args()


def sample_states(sample):
    load = [0] * int(sample["num_atoms"])
    for (source, _target), label in zip(
        sample["edge_index_bond"].tolist(), sample["bond_type_full"].tolist()
    ):
        load[int(source)] += BOND_CAPACITY[int(label)]
    for atom, (z, charge, hydrogens) in enumerate(
        zip(
            sample["elems"].tolist(),
            sample["formal_charge"].tolist(),
            sample["implicit_h_count"].tolist(),
        )
    ):
        yield int(z), int(charge), int(load[atom] + int(hydrogens))


def main():
    args = parse_args()
    cache_path = Path(args.cache)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    samples = payload["samples"]

    fitted_counts = Counter()
    split_sizes = Counter()
    for sample in samples:
        split = str(sample.get("split", "train"))
        split_sizes[split] += 1
        if split != args.fit_split:
            continue
        fitted_counts.update(sample_states(sample))

    by_element_charge = defaultdict(list)
    for (z, charge, valence), count in fitted_counts.items():
        if count >= args.min_count:
            by_element_charge[(z, charge)].append((valence, count))

    catalogue = {}
    for (z, charge), observations in sorted(by_element_charge.items()):
        vmax = max(valence for valence, _count in observations)
        support = sum(count for _valence, count in observations)
        catalogue.setdefault(str(z), []).append(
            {"charge": charge, "max_valence": vmax, "support": support}
        )

    fitted = {
        (int(z), int(state["charge"])): int(state["max_valence"])
        for z, states in catalogue.items()
        for state in states
    }
    coverage = {}
    for split in sorted(split_sizes):
        atoms = covered = 0
        missing = Counter()
        for sample in samples:
            if str(sample.get("split", "train")) != split:
                continue
            for z, charge, valence in sample_states(sample):
                atoms += 1
                capacity = fitted.get((z, charge))
                if capacity is not None and valence <= capacity:
                    covered += 1
                else:
                    missing[(z, charge, valence)] += 1
        coverage[split] = {
            "molecules": int(split_sizes[split]),
            "atoms": atoms,
            "covered_atoms": covered,
            "coverage": covered / atoms if atoms else None,
            "missing_states": [
                {"atomic_number": z, "charge": q, "valence": v, "count": count}
                for (z, q, v), count in missing.most_common()
            ],
        }

    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    report = {
        "schema": "bondnet.training_state_catalog.v1",
        "source_cache": str(cache_path.resolve()),
        "source_sha256": digest,
        "fit_split": args.fit_split,
        "min_count": args.min_count,
        "bond_capacity": BOND_CAPACITY,
        "catalogue": catalogue,
        "heldout_coverage": coverage,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
