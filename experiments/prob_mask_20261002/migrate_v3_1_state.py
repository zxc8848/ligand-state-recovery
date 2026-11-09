"""Explicit metadata-only transition for the raw500/eligible491 bug fix."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from frozen_manifest import ROOT, DEFAULT, paths, sha
from population_manifest import validate_population

ALLOWED_CHANGES = {
    'experiments/prob_mask_20261002/' + name for name in (
        'evaluation_common.py', 'unified_resonance.py', 'paired_verdicts.py',
        'native_ladder_stats.py', 'pose_ablate_search.py', 'pose_localmap.py',
        'frozen_manifest.py', 'verify_frozen_config.py', 'test_eval_v3.py',
        'smoke_pipeline_v3.py')
} | {'scripts/run_locked_eval.ps1'}
ALLOWED_ADDITIONS = {
    'experiments/prob_mask_20261002/population_manifest.py',
    'experiments/prob_mask_20261002/migrate_v3_1_state.py',
    'results/FROZEN_CONFIG_V3_1_2026-10-05.md',
    'data/pdbbind_pose_locked500_population_v3_1.json',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    old_path = ROOT/'results/FROZEN_MANIFEST_V3_2026-10-04.json'
    state_path = ROOT/'results/LOCKED_EVAL_STATE_V3.json'
    backup = ROOT/'results/LOCKED_EVAL_STATE_V3.pre_v3_1_20261005.json'
    receipt = ROOT/'results/LOCKED_EVAL_MIGRATION_V3_1_20261005.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))['files']
    new = json.loads(DEFAULT.read_text(encoding='utf-8'))['files']
    current = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths()}
    if current != new:
        raise ValueError('New frozen manifest does not match current bytes')
    removed = set(old)-set(new)
    added = set(new)-set(old)
    changed = {p for p in old.keys() & new.keys() if old[p] != new[p]}
    if removed or added-ALLOWED_ADDITIONS or changed-ALLOWED_CHANGES:
        raise ValueError(f'Unexpected change: removed={sorted(removed)}, '
                         f'added={sorted(added-ALLOWED_ADDITIONS)}, '
                         f'changed={sorted(changed-ALLOWED_CHANGES)}')
    state = json.loads(state_path.read_text(encoding='utf-8-sig'))
    if (state['manifest_sha256'] != sha(old_path) or state['status'] != 'FAILED'
            or state.get('main_complete') or state.get('knodle_complete')):
        raise ValueError('State is not the interrupted, incomplete v3 run')
    sdf = ROOT/'data/pdbbind_pose_locked500.sdf'
    pop = ROOT/'data/pdbbind_pose_locked500_population_v3_1.json'
    refs, excluded = validate_population(sdf, pop)
    if (len(refs), len(excluded)) != (491, 9) or sha(sdf) != state['locked_set_sha256']:
        raise ValueError('Locked input/population differs')
    print('Migration checks passed: weights/inputs and unrelated frozen sources unchanged; '
          'raw500/eligible491; no prediction/scoring performed')
    if not a.apply:
        print('Check only. Use --apply for the explicit state transition.')
        return
    if backup.exists() or receipt.exists():
        raise ValueError('Migration backup/receipt already exists; do not overwrite')
    backup.write_bytes(state_path.read_bytes())
    data = {
        'version': '3.1', 'migrated_at_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Correct raw-record count vs pre-existing element-eligible count',
        'old_manifest_sha256': sha(old_path), 'new_manifest_sha256': sha(DEFAULT),
        'changed_source_files': sorted(changed), 'added_files': sorted(added),
        'raw_n': 500, 'eligible_n': 491, 'population_sha256': sha(pop),
        'old_state_backup': str(backup), 'old_state_sha256': sha(backup),
        'is_untouched_test': False, 'no_locked_predictions_scored_during_migration': True,
    }
    receipt.write_text(json.dumps(data, indent=1), encoding='utf-8')
    state.update(manifest_sha256=sha(DEFAULT), previous_manifest_sha256=sha(old_path),
                 status='FAILED_READY_FOR_V3_1_RESUME', migration_receipt=str(receipt),
                 population_manifest=str(pop), raw_n=500, eligible_n=491,
                 is_untouched_test=False)
    tmp = state_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=1), encoding='utf-8')
    tmp.replace(state_path)
    print('State migrated with historical backup and receipt:', receipt)


if __name__ == '__main__':
    main()
