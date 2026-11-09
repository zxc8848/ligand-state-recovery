"""Reuse completed v2 paired predictions only when scientific functions/inputs match."""
import ast
import gzip
import json
import shutil
from pathlib import Path

import study as S


def function_ast(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names}


def main():
    protocol = S.freeze()
    names = {"blinded", "beam", "decode", "infer", "targets", "intervals"}
    assert function_ast(S.HERE/"history/study_v2.py", names) == function_ast(S.HERE/"study.py", names)
    old = S.read(S.HERE/"PROTOCOL_v2.json")["spec"]["fingerprints"]
    new = S.read(S.HERE/"PROTOCOL_v3.json")["spec"]["fingerprints"]
    excluded = {str((S.HERE/x).relative_to(S.ROOT)) for x in ("study.py", "README.md")}
    for key, value in old.items():
        if key not in excluded:
            assert new[key] == value, f"Input changed: {key}"
    for source in sorted((S.HERE/"evaluations").glob("seed*/*/COMPLETE.json")):
        old_done = S.read(source)
        assert old_done["protocol_sha256"] == S.sha(S.HERE/"PROTOCOL_v2.json")
        seed = int(source.parent.parent.name[4:])
        condition = source.parent.name
        target = S.HERE/"evaluations_v3"/f"seed{seed}"/condition
        if (target/"COMPLETE.json").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        inputs = {}
        for path in source.parent.iterdir():
            if path.is_file() and path.name not in ("COMPLETE.json", "predictions.json.gz"):
                shutil.copy2(path, target/path.name)
                inputs[path.name] = S.sha(path)
        with gzip.open(source.parent/"predictions.json.gz", "rt", encoding="utf-8") as f:
            predictions = json.load(f)
        blob = S.torch.load(S.CACHE_ROOT/f"{condition}.pt", map_location="cpu", weights_only=False)
        rows = {r["id"]: r for r in blob["rows"]}
        archive = {r["name"]: r for r in S.read(S.historical(seed, condition))["records"]}
        for record in predictions["records"]:
            record["historical_audit"] = S.check_parity(record, rows[record["name"]], archive[record["name"]])
        predictions["source_protocol_sha256"] = predictions["protocol_sha256"]
        predictions["protocol_sha256"] = protocol
        with gzip.open(target/"predictions.json.gz", "wt", encoding="utf-8") as f:
            json.dump(predictions, f, separators=(",", ":"))
        done = dict(old_done, protocol_sha256=protocol,
                    migrated_from=str(source.parent.relative_to(S.ROOT)),
                    historical_audit={k: sum(not r["historical_audit"][k] for r in predictions["records"])
                    for k in ("same_selected_B", "same_selected_buildability",
                              "same_selected_native", "same_unordered_candidate_outcomes")})
        S.save(target/"MIGRATION_RECEIPT.json", {
            "source_complete_sha256": S.sha(source),
            "source_prediction_sha256": S.sha(source.parent/"predictions.json.gz"),
            "inference_function_asts_identical": sorted(names),
            "non_reporting_inputs_identical": True, "copied_verdict_file_hashes": inputs,
            "source_protocol_sha256": S.sha(S.HERE/"PROTOCOL_v2.json"),
            "target_protocol_sha256": protocol})
        S.save(target/"COMPLETE.json", done)
        S.log(f"Migrated unchanged paired cell {seed} {condition}")


if __name__ == "__main__":
    main()
