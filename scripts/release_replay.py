"""Stage exact historical inputs, or verify saved locked predictions, without inference."""
import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = Path('experiments/prob_mask_20261002')
ARCHIVE = Path('experiments/milp_native_completion_20261102')
FROZEN = Path('results/FROZEN_MANIFEST_V3_1_2026-10-05.json')
SEEDS = (20261002, 20261003, 20261004)
LOCKED_SDF = Path('data/pdbbind_pose_locked500.sdf')
POP = Path('data/pdbbind_pose_locked500_population_v3_1.json')
HISTORY_EXTRAS = (
    'results/FROZEN_CONFIG_2026-10-04.md',
    'work/evaluation_fix_v3/end_to_end/PASSED.json',
    str(CORE / 'joint_v6/joint_v1/pdbbind_sigma0p00/species_localmap_deloc.json').replace('\\', '/'),
    str(CORE / 'joint_v6/joint_v1/pdbbind_sigma0p00/pose_delocalize.json').replace('\\', '/'),
    'work/paper2_noise_ccd_probe/pdbbind_sigma0p00/reference_for_scoring.sdf',
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def bound(rel, base):
    path = base / rel
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'Unsafe path: {rel}')
    return path


def checked(path, want):
    if not path.is_file() or sha(path) != want:
        raise ValueError(f'Missing or hash-mismatched input: {path}')
    return path


def stage_history(authorized, out, inventory):
    if out.exists():
        raise FileExistsError(out)
    manifest = read(ROOT / FROZEN)['files']
    paths = {}
    for rel, want in manifest.items():
        candidates = (bound(rel, ROOT/'historical_v3_1/overlay'), bound(rel, ROOT), bound(rel, authorized))
        src = next((p for p in candidates if p.is_file() and sha(p) == want), None)
        if src is None:
            raise ValueError(f'No exact original file for {rel}')
        paths[rel] = src
    for rel in HISTORY_EXTRAS:
        local = bound(rel, ROOT)
        if local.is_file():
            paths[rel] = local
        else:
            paths[rel] = checked(bound(rel, authorized), inventory[rel]['sha256'])
    paths[FROZEN.as_posix()] = ROOT / FROZEN
    out.mkdir(parents=True, exist_ok=False)
    for rel, src in paths.items():
        dst = bound(rel, out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f'Isolated historical tree: {len(manifest)} verified frozen inputs plus preflight helpers.')
    print('Original workspace and manifests were not modified. Do not submit this tree as a public archive.')


def audit_saved(authorized, inventory):
    import torch
    from rdkit import Chem, RDLogger, rdBase

    torch.set_num_threads(4)
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, str(ROOT/CORE))
    import joint as J
    from evaluation_common import score_dump
    from population_manifest import validate_population

    def get(rel):
        rel = Path(rel).as_posix()
        return checked(bound(rel, authorized), inventory[rel]['sha256'])

    ref_file = get(LOCKED_SDF)
    pop_file = get(POP)
    refs, excluded = validate_population(ref_file, pop_file)
    assert len(refs) == 491 and len(excluded) == 9
    target = {n: J.make_row(m, n, -1, 'release_audit') for n, m in refs.items()}
    graph_sets = {'archived': CORE/'locked_eval/old_per_edge.jsonl',
                  'knodle': CORE/'locked_eval/knodle_per_edge.jsonl'}
    for seed in SEEDS:
        graph_sets[str(seed)] = CORE/f'locked_eval/seed{seed}/pose_localmap_deloc_locked_fullgraph.jsonl'
    graphs = {}
    for name, rel in graph_sets.items():
        scores = score_dump(get(rel), ref_file, expected_n=491, population_manifest=pop_file)
        rows = scores['per_molecule']
        graphs[name] = {k:scores[k] for k in ('strict','unified_resonance','smiles_species')}
        graphs[name]['typed_true_edge_recall'] = sum(r['true_edge_hits'] for r in rows)/sum(r['true_edge_count'] for r in rows)

    def rebuild(name, pairs, labels, states):
        r = target[name]
        row = {'z':r['z'], 'pairs':torch.tensor(pairs, dtype=torch.long)}
        return J.build_species(row, torch.tensor(labels, dtype=torch.long), torch.tensor(states, dtype=torch.long))

    native = {}
    for seed in SEEDS:
        rows = read(get(CORE/f'locked_eval/seed{seed}/pose_localmap_locked.json'))['records']
        assert len(rows) == 491 and {r['name'] for r in rows} == set(refs)
        n = 0
        for r in rows:
            name = r['name']
            c = r['candidates_detail'][r['selected_indices']['localmap']]
            smi, good = rebuild(name,r['pairs'],c['bond_labels'],c['atom_states'])
            ok = bool(good and smi == target[name]['smiles'])
            assert (smi,good,ok) == (c['smiles'],c['good'],c['correct'])
            n += ok
        native[str(seed)] = n/491

    lines = get(ARCHIVE/'evaluations/locked_clean/predictions.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(x) for x in lines if x]
    assert len(rows) == 491 and {r['name'] for r in rows} == set(refs)
    old = 0
    for r in rows:
        name, f = r['name'], r['final']
        ok = False
        if f['complete']:
            states = [(q+2)*5+h for q,h in zip(f['q'],f['h'])]
            smi, good = rebuild(name,f['pairs'],f['labels'],states)
            good = bool(good and f['sum_correct'] and f['in_vocabulary'])
            assert (smi,good) == (f['smiles'],f['good'])
            ok = bool(good and smi == target[name]['smiles'])
        assert ok == f['native']
        old += ok

    current = statistics.mean(native.values())
    assert round(100*current,2) == 82.89 and round(100*old/491,2) == 74.34
    assert round(100*statistics.mean(graphs[str(s)]['unified_resonance'] for s in SEEDS),2) == 83.44
    assert round(100*graphs['knodle']['unified_resonance'],2) == 78.62
    return {'scope':'saved-prediction rescoring only, no model inference or training',
            'rdkit':rdBase.rdkitVersion, 'n':491, 'raw_n':500, 'native_current':current,
            'native_current_per_seed':native, 'native_archived_completed':old/491,
            'graph_endpoints':graphs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage-history','audit-saved'])
    p.add_argument('--authorized-root', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inventory = {r['path']:r for r in read(a.inventory)['main']}
    if a.action == 'stage-history':
        stage_history(a.authorized_root, a.output, inventory)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        result = audit_saved(a.authorized_root, inventory)
        a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
