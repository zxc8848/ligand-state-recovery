"""Independent saved-output invariants; no model forward and no locked data."""
import gzip
import json

import study as S


def main():
    protocol = S.freeze()
    receipts = []
    for condition, (_, expected_n) in S.CONDITIONS.items():
        cache = S.torch.load(S.CACHE_ROOT/f"{condition}.pt", map_location="cpu", weights_only=False)
        rows = {r["id"]: r for r in cache["rows"]}
        for seed in S.SEEDS:
            root = S.HERE/"evaluations_v3"/f"seed{seed}"/condition
            done = S.read(root/"COMPLETE.json")
            assert done["protocol_sha256"] == protocol and done["n"] == expected_n
            with gzip.open(root/"predictions.json.gz", "rt", encoding="utf-8") as f:
                predictions = json.load(f)
            assert predictions["protocol_sha256"] == protocol
            records = predictions["records"]
            assert len(records) == expected_n and {r["name"] for r in records} == set(rows)
            flags = {}
            for arm in S.ARMS:
                v = S.read(root/f"verdicts_{arm}.json")
                flags[arm] = {r["name"]: r for r in v["per_molecule"]}
                assert set(flags[arm]) == set(rows)
            native_build_checks = 0
            candidate_state_checks = 0
            for r in records:
                row = rows[r["name"]]
                pool = r["pool"]
                assert S.digest(pool) == r["pool_sha256"]
                assert pool["pairs"] == row["pairs"].tolist()
                n = len(pool["bond_labels"])
                assert 1 <= n <= 48 and len(pool["bscore"]) == n
                for labels in pool["bond_labels"]:
                    assert len(labels) == len(pool["pairs"]) and set(labels) <= set(range(5))
                for arm in S.ARMS:
                    item = r[arm]
                    assert len(item["candidates"]) == n
                    chosen = max(range(n), key=lambda i: (
                        item["candidates"][i]["good"], item["candidates"][i]["score"]))
                    assert item["selected_index"] == chosen
                    assert item["labels"] == pool["bond_labels"][chosen]
                    for i, c in enumerate(item["candidates"]):
                        assert len(c["atom_states"]) == len(row["z"])
                        assert all(0 <= a < 25 for a in c["atom_states"])
                        assert sum(a//5-2 for a in c["atom_states"]) == row["Q"]
                        assert abs(c["score"]-pool["bscore"][i]-c["ascore"]) < 1e-6
                        candidate_state_checks += 1
                    c = item["candidates"][chosen]
                    smi, good = S.J.build_species(row,
                        S.torch.tensor(item["labels"]), S.torch.tensor(c["atom_states"]))
                    assert smi == c["smiles"] and good == c["good"]
                    native = bool(good and smi == row["smiles"])
                    assert native == item["native"] == flags[arm][r["name"]]["native"]
                    assert good == flags[arm][r["name"]]["constructible"]
                    target = row["atom_target"]
                    atoms = S.torch.tensor(c["atom_states"])
                    assert bool(S.torch.equal(atoms//5, target//5)) == item["q_exact"]
                    assert bool(S.torch.equal(atoms%5, target%5)) == item["h_exact"]
                    assert bool(S.torch.equal(atoms, target)) == item["atom_state_exact"]
                    native_build_checks += 1
            receipt = {"seed": seed, "condition": condition, "n": expected_n,
                       "candidate_charge_and_score_checks": candidate_state_checks,
                       "selected_native_rebuild_checks": native_build_checks,
                       "same_saved_pool_and_bscores": True}
            receipts.append(receipt)
            S.log(f"Verified saved outputs {seed} {condition}")
    S.save(S.HERE/"OUTPUT_VERIFICATION.json", {
        "protocol_sha256": protocol, "cells": receipts,
        "all_fixed_populations_included": True,
        "no_new_model_forward_or_locked_evaluation": True})


if __name__ == "__main__":
    S.RDLogger.DisableLog("rdApp.*")
    main()
