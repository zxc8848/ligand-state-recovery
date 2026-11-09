"""Read-only diagnostic for an archived-score mismatch; no result files written."""
import sys
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study as S
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

torch.set_num_threads(4)
name = sys.argv[1] if len(sys.argv) > 1 else "2avq"
condition = sys.argv[2] if len(sys.argv) > 2 else "pose_clean"
blob = torch.load(S.CACHE_ROOT/f"{condition}.pt", weights_only=False, map_location="cpu")
row = next(r for r in blob["rows"] if r["id"] == name)
old = next(r for r in S.read(S.historical(20261002, condition))["records"] if r["name"] == name)
catalogue = S.J.make_catalogue(torch.load(S.SOURCES, weights_only=False, map_location="cpu")["train"])
net, _ = S.T.load_joint(S.checkpoint(20261002), "cuda")
pred = S.infer(net, row, catalogue, "cuda")
selected = pred[S.ARMS[0]]["selected_index"]
print("selected", selected, "native", pred[S.ARMS[0]]["candidates"][selected]["smiles"] == row["smiles"],
      "old_native", old["localmap_species"], "same_bonds",
      pred["pool"]["bond_labels"][selected] == old["localmap_labels"], "Q", row["Q"])
for i, (new, archive) in enumerate(zip(pred[S.ARMS[0]]["candidates"], old["candidates_detail"])):
    if abs(new["score"]-archive["score"]) > 0.002:
        print(json.dumps({"candidate": i, "old": archive, "new": new}, indent=2))
with torch.no_grad():
    rows = [row, S.blinded(row)]
    outputs = []
    for r in rows:
        b = S.J.pack([r], "cuda")
        outputs.append(net(b, torch.full_like(b["labels"], S.J.BMASK),
                           torch.full_like(b["atarget"], S.J.AMASK)))
    print("target_placeholder_logit_difference",
          [float((a-b).abs().max()) for a,b in zip(*outputs)])
    model, args = S.J.p.load_backbone(S.ROOT/"checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt","cuda")
    refs, _ = S.E.reference_population(S.reference(condition))
    fresh = S.J.make_row(refs[name], name, -1, "diagnostic")
    fresh = S.J.p.geometry(fresh, 0., np.random.default_rng(11), args["cutoff"])
    fresh = S.J.extract(model, [fresh], "cuda")[0]
    print("fresh_feature_max_diff", {k: float((fresh[k]-row[k]).abs().max())
                                   for k in ("feat","node_feat","base")})
