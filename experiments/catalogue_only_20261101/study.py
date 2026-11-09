"""Paired development-only removal of inference-time neural atom logits."""
import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/prob_mask_20261002"
sys.path.insert(0, str(OLD))
import joint as J
import jointt as T
import evaluation_common as E
from cluster_bootstrap import identity_key, mcnemar
from pose_localmap import write_dumps

SEEDS = (20261002, 20261003, 20261004)
CONDITIONS = {
    "pose_clean": ("pdbbind_sigma0p00", 488),
    "pose_noise010": ("pdbbind_sigma0p10", 488),
    "ccd_ideal": ("ccd_sigma0p00", 1000),
}
ARMS = ("neural_catalogue", "catalogue_only")
ENDPOINTS = ("native", "strict", "unified_resonance", "smiles_species")
SOURCES = OLD / "joint_v1/sources.pt"
CACHE_ROOT = ROOT / "experiments/mask_control_20261101/eval_features"
AUDIT = ROOT / "experiments/mask_control_20261101/IDENTITY_AUDIT.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def log(text):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {text}", flush=True)


def reference(condition):
    return ROOT / "work/paper2_noise_ccd_probe" / CONDITIONS[condition][0] / "reference_for_scoring.sdf"


def historical(seed, condition):
    return OLD / "joint_v7" / f"seed{seed}" / CONDITIONS[condition][0] / "pose_localmap_abl.json"


def checkpoint(seed):
    return OLD / "joint_v5" / f"seed{seed}" / "best.pt"


def fingerprints():
    paths = [Path(__file__), HERE / "README.md", HERE / "PREFLIGHT_DIAGNOSTIC.md", SOURCES, AUDIT]
    paths += [OLD / x for x in ("joint.py", "jointt.py", "probe.py",
                               "evaluation_common.py", "pose_localmap.py",
                               "cluster_bootstrap.py")]
    legacy = ROOT / "archive/legacy_two_stage_20261003/scripts"
    paths += [legacy / x for x in ("benchmark_templatefree_motifs.py",
                                  "classify_candidate_ambiguity.py",
                                  "evaluate_species_recovery.py",
                                  "evaluate_candidate_sets.py")]
    for condition in CONDITIONS:
        paths += [CACHE_ROOT / f"{condition}.pt", reference(condition)]
        paths += [historical(s, condition) for s in SEEDS]
    paths += [checkpoint(s) for s in SEEDS]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    spec = {
        "schema": "catalogue-only-atom-logit-control.v3",
        "scope": "original released heads; development only; no locked inference",
        "seeds": SEEDS, "conditions": CONDITIONS, "arms": ARMS,
        "change": "neural atom logits = 0; normalized catalogue prior only",
        "same_within_pair": ["candidate content/order", "bond scores", "catalogue",
                             "sum(q)=Q solver", "constructibility rule", "ranking"],
        "candidate_budget": {"fraction": .10, "cap": 8, "topk": 2, "beam": 48},
        "selection": "lexicographic max(buildable, bond_logprob + normalized_atom_logprob)",
        "delocalized_rewriting": False, "endpoints": ENDPOINTS,
        "holm_family": 36, "bootstrap_reps": 2000,
        "historical_audit": "record selected bond/buildability/native differences and unordered signatures; do not filter samples on output parity",
        "sensitivity": "saved known-source exclusions; descriptive only",
        "fingerprints": fingerprints(),
    }
    # JSON normalization makes tuples and lists equivalent on reload.
    spec = json.loads(json.dumps(spec))
    path = HERE / "PROTOCOL_v3.json"
    if path.exists():
        old = read(path)
        if old["spec"] != spec:
            raise ValueError("Protocol/source/input changed; refusing to reuse outputs")
    else:
        save(path, {"frozen_utc": datetime.now(timezone.utc).isoformat(), "spec": spec})
    return sha(path)


def blinded(row):
    # Only these non-target fields reach prediction or catalogue lookup.
    return {k: row[k] for k in ("z", "pairs", "feat", "node_feat", "base", "Q")} | {
        "labels": torch.zeros(len(row["pairs"]), dtype=torch.long),
        "atom_target": torch.full((len(row["z"]),), 10, dtype=torch.long),
    }


def beam(blog):
    base = blog.argmax(-1)
    k = min(8, max(1, round(.10 * len(base))))
    flagged = torch.topk(-blog.gather(1, base[:, None]).squeeze(1), k).indices.tolist()
    choices = list(itertools.product(*[torch.topk(blog[e], 2).indices.tolist() for e in flagged]))
    choices.sort(key=lambda c: -sum(float(blog[e, v]) for e, v in zip(flagged, c)))
    candidates = []
    for choice in choices[:48]:
        labels = base.clone()
        for e, v in zip(flagged, choice):
            labels[e] = v
        candidates.append(labels)
    assert torch.equal(candidates[0], base)
    return candidates, k


def decode(row, candidates, bscore, logits, device):
    counts = [len(row["z"])] * len(candidates)
    state = torch.full((sum(counts),), J.AMASK, device=device, dtype=torch.long)
    q = torch.full((len(counts),), int(row["Q"]), device=device, dtype=torch.long)
    assigned = J.constrained_atoms(logits, state, counts, q,
                                   np.random.default_rng(11), greedy=True)
    parts = assigned.cpu().split(counts)
    probs = logits.log_softmax(-1).cpu().split(counts)
    details = []
    for labels, atoms, lp, bs in zip(candidates, parts, probs, bscore):
        if int((atoms // 5 - 2).sum()) != int(row["Q"]):
            raise AssertionError("Charge conservation failed")
        smi, good = J.build_species(row, labels.cpu(), atoms)
        atom_score = float(lp.gather(1, atoms[:, None]).sum())
        details.append({"atom_states": atoms.tolist(), "smiles": smi, "good": bool(good),
                        "ascore": atom_score, "score": bs + atom_score})
    picked = max(range(len(details)), key=lambda i: (details[i]["good"], details[i]["score"]))
    return {"selected_index": picked, "candidates": details}


@torch.no_grad()
def infer(net, source_row, catalogue, device):
    row = blinded(source_row)
    one = J.pack([row], device)
    bl, _ = net(one, torch.full_like(one["labels"], J.BMASK),
                torch.full_like(one["atarget"], J.AMASK))
    blog = bl.log_softmax(-1)
    candidates, flagged = beam(blog)
    bscores = [float(blog.gather(1, c[:, None]).sum()) for c in candidates]
    batch = J.pack([row] * len(candidates), device)
    states = torch.cat(candidates)
    _, neural = net(batch, states, torch.full_like(batch["atarget"], J.AMASK))
    prior = J.catalogue_logits(batch, states, catalogue)
    pooled = [c.cpu().tolist() for c in candidates]
    shared = {"pairs": row["pairs"].tolist(), "bond_labels": pooled, "bscore": bscores}
    result = {"pool": shared, "pool_sha256": digest(shared), "flagged": flagged}
    result[ARMS[0]] = decode(row, candidates, bscores, neural + prior, device)
    result[ARMS[1]] = decode(row, candidates, bscores, torch.zeros_like(neural) + prior, device)
    return result


def selftest():
    source = {"z": torch.tensor([6, 6]), "truth": {(0, 1): 1},
              "atom_target": torch.tensor([13, 13])}
    cat = J.make_catalogue([source])
    b = {"z": source["z"], "pairs": torch.tensor([[0, 1]]), "feat": torch.zeros(1, 1)}
    lp = J.catalogue_logits(b, torch.tensor([1]), cat)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-6)
    row = {"z": source["z"], "pairs": b["pairs"], "Q": 0}
    d = decode(row, [torch.tensor([1])], [0.], lp, "cpu")
    assert d["candidates"][0]["good"] and d["candidates"][0]["smiles"] == "CC"
    uniform = torch.full_like(lp, -np.log(25))
    assert torch.allclose((lp + uniform).log_softmax(-1), lp.log_softmax(-1), atol=2e-6)
    for q in (-1, 0, 1):
        row["Q"] = q
        d = decode(row, [torch.tensor([1])], [0.], lp, "cpu")
        assert sum(s // 5 - 2 for s in d["candidates"][0]["atom_states"]) == q
    log("Self-tests passed: normalization, uniform-logit equivalence, build, charge")


def check_parity(pred, row, old):
    if pred["pool"]["pairs"] != old["pairs"]:
        raise AssertionError(f"{row['id']}: pair order mismatch")
    details = pred[ARMS[0]]["candidates"]
    archive = old["candidates_detail"]
    if len(details) != len(archive):
        raise AssertionError(f"{row['id']}: beam size mismatch")
    labels = pred["pool"]["bond_labels"]
    # Historical development beams lack candidate labels. Near-tied bond scores
    # can swap the enumeration order across GPU runs; compare the available
    # unordered signatures, not unrelated candidates at the same array index.
    new_signatures = sorted(
        (sum(x != y for x, y in zip(c, labels[0])), d["good"],
         d["smiles"] == row["smiles"], d["score"])
        for c, d in zip(labels, details))
    old_signatures = sorted((a["dist"], a["good"], a["correct"], a["score"]) for a in archive)
    selected = labels[pred[ARMS[0]]["selected_index"]]
    picked = details[pred[ARMS[0]]["selected_index"]]
    signatures_match = all(d[:3] == a[:3] for d, a in zip(new_signatures, old_signatures))
    return {
        "same_selected_B": selected == old["localmap_labels"],
        "same_selected_buildability": picked["good"] == old["localmap_valid"],
        "same_selected_native": (picked["smiles"] == row["smiles"]) == old["localmap_species"],
        "same_unordered_candidate_outcomes": signatures_match,
        "max_unordered_score_difference": max(abs(d[3]-a[3]) for d, a in
                                              zip(new_signatures, old_signatures))
                                           if signatures_match else None,
    }


def targets(pred, row):
    pred.update(name=row["id"], Q=int(row["Q"]), z=row["z"].tolist())
    for arm in ARMS:
        item = pred[arm]
        chosen = item["candidates"][item["selected_index"]]
        atoms = torch.tensor(chosen["atom_states"])
        item["labels"] = pred["pool"]["bond_labels"][item["selected_index"]]
        item["native"] = bool(chosen["good"] and chosen["smiles"] == row["smiles"])
        item["constructible"] = chosen["good"]
        item["q_exact"] = bool(torch.equal(atoms // 5, row["atom_target"] // 5))
        item["h_exact"] = bool(torch.equal(atoms % 5, row["atom_target"] % 5))
        item["atom_state_exact"] = bool(torch.equal(atoms, row["atom_target"]))
    return pred


def cell(seed, condition, catalogue, device, protocol, smoke=False):
    root = HERE / ("smoke_v3" if smoke else "evaluations_v3") / f"seed{seed}" / condition
    done = root / "COMPLETE.json"
    if done.exists():
        if read(done)["protocol_sha256"] != protocol:
            raise ValueError("Completed cell belongs to another protocol")
        log(f"skip completed {seed} {condition}")
        return
    root.mkdir(parents=True, exist_ok=True)
    blob = torch.load(CACHE_ROOT / f"{condition}.pt", map_location="cpu", weights_only=False)
    rows = blob["rows"][:4] if smoke else blob["rows"]
    if len(blob["rows"]) != CONDITIONS[condition][1] or sha(reference(condition)) != blob["reference_sha256"]:
        raise ValueError("Feature/reference population or bytes changed")
    archived = {r["name"]: r for r in read(historical(seed, condition))["records"]}
    if {r["id"] for r in blob["rows"]} != set(archived):
        raise ValueError("Archived and cached populations differ")
    net, _ = T.load_joint(checkpoint(seed), device)
    records = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        pred = infer(net, row, catalogue, device)
        pred["historical_audit"] = check_parity(pred, row, archived[row["id"]])
        records.append(targets(pred, row))
        if (i + 1) % 50 == 0:
            log(f"{seed} {condition}: {i+1}/{len(rows)} paired inference complete")
    scoring_sdf = reference(condition)
    if smoke:
        refs, _ = E.reference_population(scoring_sdf)
        scoring_sdf = root / "reference.sdf"
        writer = Chem.SDWriter(str(scoring_sdf))
        for row in rows:
            writer.write(refs[row["id"]])
        writer.close()
    base = {r["id"]: [{"name": r["id"], "i": i, "j": j,
                      "z_i": int(r["z"][i]), "z_j": int(r["z"][j]), "true": c}
                     for (i, j), c in sorted(r["truth"].items())] for r in rows}
    export = []
    for r in records:
        export.append({"name": r["name"], "pairs": r["pool"]["pairs"], "z": r["z"],
                       **{arm: r[arm]["labels"] for arm in ARMS}})
    write_dumps(export, base, "atom_control", root, 0., log=log,
                selectors=[(arm, arm) for arm in ARMS])
    for arm in ARMS:
        old_path = list(sys.path)
        try:
            verdict = E.score_dump(root / f"pose_{arm}_atom_control_fullgraph.jsonl",
                                   scoring_sdf, expected_n=len(rows))
        finally:
            sys.path[:] = old_path
        indexed = {r["name"]: r[arm] for r in records}
        for r in verdict["per_molecule"]:
            r.update({k: indexed[r["name"]][k] for k in
                      ("native", "constructible", "q_exact", "h_exact", "atom_state_exact")})
        for metric in ("native", "constructible", "q_exact", "h_exact", "atom_state_exact"):
            verdict[metric] = statistics.mean(r[metric] for r in verdict["per_molecule"])
        save(root / f"verdicts_{arm}.json", verdict)
    with gzip.open(root / "predictions.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"protocol_sha256": protocol, "records": records}, handle, separators=(",", ":"))
    save(done, {"protocol_sha256": protocol, "n": len(rows),
                "historical_audit": {
                    key: sum(not r["historical_audit"][key] for r in records)
                    for key in ("same_selected_B", "same_selected_buildability",
                                "same_selected_native", "same_unordered_candidate_outcomes")},
                "same_pool_and_bscores_within_pair": True,
                "seconds": time.perf_counter()-started})
    log(f"completed {seed} {condition}: n={len(rows)}, {time.perf_counter()-started:.1f}s")
    del net
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def intervals(diffs, clusters):
    rng = np.random.default_rng(11)
    ds = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(ds), (2000, len(ds)))
    mol = np.quantile(ds[idx].mean(1), [.025, .975]).tolist()
    sums = np.array([ds[c].sum() for c in clusters])
    sizes = np.array([len(c) for c in clusters])
    idx = rng.integers(0, len(clusters), (2000, len(clusters)))
    cluster = np.quantile(sums[idx].sum(1) / sizes[idx].sum(1), [.025, .975]).tolist()
    return {"molecule_ci95": mol, "identity_cluster_ci95": cluster}


def summarize(protocol):
    audit = read(AUDIT)
    result = {"scope": "paired development atom-logit ablation, no locked inference",
              "protocol_sha256": protocol, "cells": {}, "contrasts": [], "tests": []}
    extra = ("constructible", "q_exact", "h_exact", "atom_state_exact")
    for condition in CONDITIONS:
        refs, _ = E.reference_population(reference(condition))
        names = sorted(refs)
        grouped = collections.defaultdict(list)
        for i, n in enumerate(names):
            grouped[identity_key(refs[n])].append(i)
        clusters = list(grouped.values())
        excluded = set(audit["conditions"][condition]["sensitivity_exclusions"])
        subset = [n for n in names if n not in excluded]
        flags = {}
        for seed in SEEDS:
            for arm in ARMS:
                path = HERE / "evaluations_v3" / f"seed{seed}" / condition / f"verdicts_{arm}.json"
                v = read(path)
                flags[seed, arm] = {r["name"]: r for r in v["per_molecule"]}
                if set(flags[seed, arm]) != set(names):
                    raise ValueError("Verdict population drift")
        for arm in ARMS:
            entry = {}
            for metric in ENDPOINTS + extra:
                vals = [statistics.mean(flags[s, arm][n][metric] for n in names) for s in SEEDS]
                entry[metric] = {"per_seed": vals, "mean": statistics.mean(vals), "sd": statistics.stdev(vals)}
            entry["source_excluded"] = {"n": len(subset), "metrics": {}}
            for metric in ENDPOINTS:
                vals = [statistics.mean(flags[s, arm][n][metric] for n in subset) for s in SEEDS]
                entry["source_excluded"]["metrics"][metric] = {
                    "per_seed": vals, "mean": statistics.mean(vals), "sd": statistics.stdev(vals)}
            result["cells"][f"{condition}/{arm}"] = entry
        for metric in ENDPOINTS:
            per_seed = []
            for seed in SEEDS:
                # Positive delta means the learned neural atom logits help.
                pairs = [(int(flags[seed, ARMS[1]][n][metric]),
                          int(flags[seed, ARMS[0]][n][metric])) for n in names]
                ds = np.array([y-x for x, y in pairs]); per_seed.append(ds)
                result["tests"].append({"condition": condition, "seed": seed,
                    "endpoint": metric, "neural_minus_catalogue": float(ds.mean()),
                    "gained": sum(y and not x for x, y in pairs),
                    "lost": sum(x and not y for x, y in pairs),
                    "raw_p": mcnemar(pairs), **intervals(ds, clusters)})
            mean = np.mean(per_seed, axis=0)
            result["contrasts"].append({"condition": condition, "endpoint": metric,
                "mean_neural_minus_catalogue": float(mean.mean()),
                "seed_sd": float(np.std([v.mean() for v in per_seed], ddof=1)),
                **intervals(mean, clusters)})
    assert len(result["tests"]) == 36
    running = 0.
    for i, test in enumerate(sorted(result["tests"], key=lambda t: t["raw_p"])):
        running = max(running, min(1., (36-i)*test["raw_p"]))
        test["holm_p_36"] = running
    result["holm_family_size"] = 36
    save(HERE / "SUMMARY.json", result)
    lines = ["# Neural atom logits versus catalogue-only", "",
             "Original released bond model and paired candidate pools; development only.",
             "Percentages: mean +/- SD over the three original head seeds.", "",
             "| Condition | Arm | Native | Strict | Resonance | Graph SMILES |",
             "|---|---|---:|---:|---:|---:|"]
    for key, v in result["cells"].items():
        condition, arm = key.split("/")
        numbers = [f'{100*v[e]["mean"]:.2f} +/- {100*v[e]["sd"]:.2f}' for e in ENDPOINTS]
        lines.append("| " + " | ".join([condition, arm] + numbers) + " |")
    lines += ["", "Positive differences below favour neural atom logits.", "",
              "| Condition | Native difference pp | Identity-cluster CI pp | Seed-wise Holm p |",
              "|---|---:|---|---|"]
    for c in result["contrasts"]:
        if c["endpoint"] != "native":
            continue
        tests = [t for t in result["tests"] if t["condition"] == c["condition"] and t["endpoint"] == "native"]
        ci = [100*x for x in c["identity_cluster_ci95"]]
        lines.append(f'| {c["condition"]} | {100*c["mean_neural_minus_catalogue"]:+.2f} | '
                     f'[{ci[0]:+.2f}, {ci[1]:+.2f}] | '
                     + ", ".join(f'{t["holm_p_36"]:.4g}' for t in tests) + " |")
    lines += ["", "Intervals condition on these trained models; they do not include all training randomness.",
              "Indexed q/h exactness is representation-sensitive; native identity is non-stereochemical.",
              "Source-excluded subset results and all 36 tests are retained in SUMMARY.json."]
    (HERE / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    log("Wrote SUMMARY.json and RESULTS.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "freeze", "smoke", "run", "summarize"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    RDLogger.DisableLog("rdApp.*")
    if args.command == "selftest":
        selftest(); return
    protocol = freeze()
    if args.command == "freeze":
        log(f"Frozen protocol: {protocol}"); return
    if args.command in ("smoke", "run"):
        catalogue = J.make_catalogue(torch.load(SOURCES, map_location="cpu", weights_only=False)["train"])
        seeds = SEEDS[:1] if args.command == "smoke" else SEEDS
        save(HERE / "STATUS.json", {"stage": args.command.upper(), "protocol_sha256": protocol})
        for seed in seeds:
            for condition in CONDITIONS:
                cell(seed, condition, catalogue, args.device, protocol, smoke=args.command=="smoke")
        if args.command == "smoke":
            log("All three smoke conditions passed archived-output parity"); return
    summarize(protocol)
    output = subprocess.run([sys.executable, "-B", str(OLD/"frozen_manifest.py")],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    if freeze() != protocol:
        raise ValueError("Protocol changed during evaluation")
    save(HERE / "FINAL_VERIFICATION.json", {"protocol_and_input_fingerprints_unchanged": True,
        "original_frozen_manifest": output.stdout.strip(), "no_locked_inference": True})
    save(HERE / "STATUS.json", {"stage": "COMPLETE", "protocol_sha256": protocol})
    log(output.stdout.strip())


if __name__ == "__main__":
    main()
