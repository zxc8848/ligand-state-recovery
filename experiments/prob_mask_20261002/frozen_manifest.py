"""Freeze/check exact source, checkpoint and input bytes without model inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/FROZEN_MANIFEST_V3_1_2026-10-05.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def paths():
    exp = ROOT/'experiments/prob_mask_20261002'
    critical = [ROOT/'results/FROZEN_CONFIG_V3_2026-10-04.md',
                ROOT/'results/FROZEN_CONFIG_V3_1_2026-10-05.md',
                ROOT/'data/pdbbind_pose_locked500_population_v3_1.json',
                ROOT/'data/pdbbind_pose_locked500.sdf',
                ROOT/'data/pdbbind_pose_locked500_freeze.json',
                ROOT/'checkpoints/diverse80k_heavy_stage1/best.pt',
                ROOT/'checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt',
                ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt',
                exp/'joint_v1/sources.pt', exp/'joint_v1/best.pt']
    critical += [exp/f'joint_v5/{s}/best.pt' for s in
                 ('seed20261002', 'seed20261003', 'seed20261004')]
    code = list(exp.glob('*.py')) + list((exp/'joint_pose').glob('*.py'))
    code += list((ROOT/'bondnet').rglob('*.py'))
    code += list((ROOT/'archive/legacy_two_stage_20261003/scripts').glob('*.py'))
    code += list((ROOT/'archive/legacy_two_stage_20261003/bondnet').rglob('*.py'))
    code += [ROOT/'scripts'/p for p in
             ('run_locked_eval.ps1', 'run_unified_resonance.ps1', 'run_knodle_locked.sh')]
    return sorted(set(critical+code))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--manifest', default=str(DEFAULT))
    a = ap.parse_args()
    manifest = Path(a.manifest)
    actual = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths()}
    if a.write:
        if manifest.exists(): raise ValueError('Manifest already exists; use a new explicit version/path')
        manifest.write_text(json.dumps({'version': '3.1', 'files': actual}, indent=1), encoding='utf-8')
        print('FROZEN', len(actual), 'files:', manifest)
    else:
        expected = json.loads(manifest.read_text(encoding='utf-8'))['files']
        changed = [p for p in set(actual)|set(expected) if actual.get(p) != expected.get(p)]
        if changed: raise ValueError(f'Frozen bytes differ: {sorted(changed)}')
        print('Exact hash manifest OK:', len(actual), 'files')


if __name__ == '__main__': main()
