"""Post hoc source-excluded sensitivity from existing unified development flags."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT/"experiments/prob_mask_20261002"
sys.path.insert(0, str(CORE))
import evaluation_common as E
from cluster_bootstrap import identity_key

BASE = ROOT/"work/evaluation_fix_v3/development/unified"
AUDIT = ROOT/"experiments/mask_control_20261101/IDENTITY_AUDIT.json"
POSE_HITS = CORE/"joint_pose/geom_overlap_names.json"
CCD_HITS = ROOT/"data/ccd_control_identity_hits.json"
SEEDS = (20261002, 20261003, 20261004)
CONDITIONS = ("pdbbind_sigma0p00", "pdbbind_sigma0p05", "pdbbind_sigma0p10",
              "pdbbind_sigma0p15", "ccd_sigma0p00", "ccd_sigma0p10")
METRICS = ("strict", "unified_resonance", "smiles_species")
EXPECTED_FULL_RES = {
    "pdbbind_sigma0p00": (.8299, .7971), "pdbbind_sigma0p05": (.8169, .7029),
    "pdbbind_sigma0p10": (.7712, .1660), "pdbbind_sigma0p15": (.7008, .0287),
    "ccd_sigma0p00": (.8210, .8810), "ccd_sigma0p10": (.7553, .1840),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(ds, groups):
    rng = np.random.default_rng(11)
    ds = np.asarray(ds, dtype=float)
    ids = rng.integers(0, len(ds), (2000, len(ds)))
    molecule = np.quantile(ds[ids].mean(1), [.025, .975]).tolist()
    sums = np.array([ds[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    ids = rng.integers(0, len(groups), (2000, len(groups)))
    cluster = np.quantile(sums[ids].sum(1)/sizes[ids].sum(1), [.025, .975]).tolist()
    return {"molecule_ci95": molecule, "identity_cluster_ci95": cluster}


def main():
    paths = [Path(__file__), HERE/"README.md", AUDIT, POSE_HITS, CCD_HITS,
             CORE/"evaluation_common.py", CORE/"cluster_bootstrap.py"]
    for c in CONDITIONS:
        paths += [BASE/c/"knodle.json"]
        paths += [BASE/c/f"new_seed{s}.json" for s in SEEDS]
        paths += [ROOT/"work/paper2_noise_ccd_probe"/c/"reference_for_scoring.sdf"]
    manifest = {
        "scope": "post hoc development sensitivity; saved flags only",
        "systems": "original released D selector and fixed Knodle binary",
        "no_new_hypothesis_tests": True, "bootstrap_reps": 2000,
        "intervals": "unadjusted; conditional on fixed trained models",
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    save(HERE/"ANALYSIS_MANIFEST.json", manifest)
    audit = read(AUDIT)
    pose_hits, ccd_hits = read(POSE_HITS), read(CCD_HITS)
    output = {"manifest_sha256": sha(HERE/"ANALYSIS_MANIFEST.json"), "conditions": {}}
    for c in CONDITIONS:
        is_pose = c.startswith("pdbbind")
        full_n, subset_n = (488, 407) if is_pose else (1000, 926)
        key = "pose_clean" if is_pose else "ccd_ideal"
        expected_exclusion = set(audit["conditions"][key]["sensitivity_exclusions"])
        refs, _ = E.reference_population(ROOT/"work/paper2_noise_ccd_probe"/c/"reference_for_scoring.sdf")
        names = sorted(refs)
        assert len(names) == full_n
        known = ((set(pose_hits["exact_smiles"]) | set(pose_hits["skeleton"])) if is_pose
                 else (set(ccd_hits["plain"]) | set(ccd_hits["stereo"])))
        derived = (known | set(audit["conditions"][key]["direct_completion_train_hits"])) & set(names)
        assert derived == expected_exclusion
        selected = [n for n in names if n not in expected_exclusion]
        assert len(selected) == subset_n
        kd, knodle = E.load_verdicts(BASE/c/"knodle.json")
        models = [E.load_verdicts(BASE/c/f"new_seed{s}.json")[1] for s in SEEDS]
        assert set(knodle) == set(names) and all(set(m) == set(names) for m in models)
        result = {"full_n": full_n, "subset_n": subset_n,
                  "excluded_names": sorted(expected_exclusion), "metrics": {}}
        groups = collections.defaultdict(list)
        for i, name in enumerate(selected):
            groups[identity_key(refs[name])].append(i)
        clusters = list(groups.values())
        result["subset_identity_clusters"] = len(clusters)
        for metric in METRICS:
            full_models = [statistics.mean(m[n][metric] for n in names) for m in models]
            subset_models = [statistics.mean(m[n][metric] for n in selected) for m in models]
            full_kn = statistics.mean(knodle[n][metric] for n in names)
            subset_kn = statistics.mean(knodle[n][metric] for n in selected)
            differences = [np.array([int(m[n][metric])-int(knodle[n][metric]) for n in selected])
                           for m in models]
            result["metrics"][metric] = {
                "full": {"ours_per_seed": full_models, "ours_mean": statistics.mean(full_models),
                         "ours_sd": statistics.stdev(full_models), "knodle": full_kn},
                "subset": {"ours_per_seed": subset_models, "ours_mean": statistics.mean(subset_models),
                           "ours_sd": statistics.stdev(subset_models), "knodle": subset_kn,
                           "mean_difference": statistics.mean(subset_models)-subset_kn,
                           "per_seed_paired_intervals": [intervals(d, clusters) for d in differences],
                           "seed_averaged_paired_intervals": intervals(np.mean(differences, axis=0), clusters)}
            }
        check = result["metrics"]["unified_resonance"]["full"]
        for observed, expected in zip((check["ours_mean"], check["knodle"]), EXPECTED_FULL_RES[c]):
            assert abs(observed-expected) <= 5.1e-5, (c, observed, expected)
        output["conditions"][c] = result
    save(HERE/"SUMMARY.json", output)
    lines = ["# Source-excluded sensitivity of the original graph comparison", "",
             "Saved original D (+delocalized) and Knodle predictions, not new inference.",
             "Post hoc descriptive analysis; no new p-values or correction family.",
             "Percentages as original-head mean +/- SD; Knodle is fixed.", "",
             "| Condition | n | Ours resonance | Knodle | Difference pp | Cluster CI pp |",
             "|---|---:|---:|---:|---:|---|"]
    for c, r in output["conditions"].items():
        v = r["metrics"]["unified_resonance"]["subset"]
        lo, hi = [100*x for x in v["seed_averaged_paired_intervals"]["identity_cluster_ci95"]]
        lines.append(f'| {c} | {r["subset_n"]} | {100*v["ours_mean"]:.2f} +/- {100*v["ours_sd"]:.2f} | '
                     f'{100*v["knodle"]:.2f} | {100*v["mean_difference"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    for metric in METRICS:
        lines += ["", f"## {metric}", "",
                  "| Condition | Full ours | Full Knodle | Subset ours | Subset Knodle |",
                  "|---|---:|---:|---:|---:|"]
        for c, r in output["conditions"].items():
            v = r["metrics"][metric]
            values = [v["full"]["ours_mean"],v["full"]["knodle"],
                      v["subset"]["ours_mean"],v["subset"]["knodle"]]
            lines.append("| " + " | ".join([c]+[f"{100*x:.2f}" for x in values]) + " |")
    (HERE/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    assert manifest["input_sha256"] == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(HERE/"VERIFICATION.json", {"all_inputs_unchanged": True,
        "all_full_resonance_rates_match_original_manuscript_rounding": True,
        "all_six_populations_and_fixed_exclusion_names_checked": True,
        "no_model_inference_or_locked_evaluation": True})
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
