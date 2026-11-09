"""Record diagnostic code changes WITHOUT overwriting the frozen manifest.

The frozen manifest fixes the code state in which the locked benchmark was read.
Adding diagnostics after that read changes file hashes. Updating the manifest in
place would silently destroy the only record of which code produced the locked
numbers, so this script never writes to it. It emits a separate diagnostic
manifest that references the frozen one by digest and enumerates every
difference and its declared purpose. Registration is not a proof of default-output
equivalence or a replacement for reviewing the code changes.

    python diagnostic_manifest.py --frozen results/FROZEN_MANIFEST_V3_1_2026-10-05.json \
        --output results/DIAGNOSTIC_MANIFEST_<date>.json [--check]

--check exits non-zero if a changed file is not in the declared allow-list below,
or a new file in the frozen source inventory is undeclared. The original manifest
is never rewritten.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Every entry states what changed and why a released number cannot move. A file
# not listed here will fail --check.
DECLARED = {
    'experiments/prob_mask_20261002/pose_localmap.py': (
        'Added --noise_seed (default 11, the value every published condition used) '
        'and --permute_seed (default 0 = no renumbering). Added localmap_states, '
        'localmap_smiles and permutation to each record. Defaults reproduce the '
        'released decode exactly; verify_frozen_config.py asserts both defaults.'),
    'experiments/prob_mask_20261002/verify_frozen_config.py': (
        'Added assertions that permute_seed == 0 and noise_seed == 11. The file '
        'checks the configuration; it does not participate in inference.'),
    'experiments/prob_mask_20261002/reference_selfscore.py': (
        'New. Measures the graph-SMILES endpoint self-score on reference graphs. '
        'Runs no model and writes no prediction.'),
    'experiments/prob_mask_20261002/atom_order_invariance.py': (
        'New. Compares decodes under atom renumbering. Pure analysis of saved runs.'),
    'experiments/prob_mask_20261002/external_state_baseline.py': (
        'New. Completes an external bond graph with the training catalogue and the '
        'charge solver. Reads no checkpoint and touches no released output.'),
    'experiments/prob_mask_20261002/noise_seed_variance.py': (
        'New. Aggregates decodes across noise seeds. Pure analysis of saved runs.'),
    'experiments/prob_mask_20261002/diagnostic_manifest.py': (
        'New. This file; checks registered changes and scans the original source '
        'inventory for undeclared additions. Registration does not certify '
        'default-output equivalence.'),
    'experiments/prob_mask_20261002/kekule_adapter.py': (
        'New. Failed graph-matching control used only by the post hoc external '
        'completion diagnostic. Not used for original released inference; '
        'matching legality is not chemical validity.'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--output', default='')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    frozen_path = Path(a.frozen)
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    files = frozen['files'] if 'files' in frozen else frozen
    frozen_digest = digest(frozen_path)

    changed, missing, added = [], [], []
    for rel, want in sorted(files.items()):
        p = ROOT/rel.replace('\\', '/')
        if not p.exists():
            missing.append(rel); continue
        got = digest(p)
        if got != want:
            changed.append({'path': rel, 'frozen_sha256': want, 'current_sha256': got,
                            'declared_reason': DECLARED.get(rel.replace('\\', '/'))})
    # Do not scan DECLARED alone: that silently misses undeclared new files.
    # Reuse exactly the source/critical-input inventory of the frozen checker.
    import frozen_manifest
    inventory = {str(p.relative_to(ROOT)).replace('\\', '/'): p
                 for p in frozen_manifest.paths()}
    frozen_keys = {k.replace('\\', '/') for k in files}
    for rel, p in sorted(inventory.items()):
        if rel not in frozen_keys and p.exists():
            added.append({'path': rel, 'current_sha256': digest(p),
                          'declared_reason': DECLARED.get(rel)})

    undeclared = [c['path'] for c in changed + added if not c['declared_reason']]
    print(f'frozen manifest {frozen_path.name}  sha256 {frozen_digest[:16]}...')
    print(f'  entries {len(files)}   changed {len(changed)}   '
          f'missing {len(missing)}   added-since {len(added)}')
    for c in changed:
        mark = '  ' if c['declared_reason'] else '  UNDECLARED '
        print(f'{mark}CHANGED  {c["path"]}')
    for c in added:
        print(f'  ADDED    {c["path"]}')
    if missing:
        for rel in missing:
            print(f'  MISSING  {rel}')

    out = {
        'schema': 'bondnet.diagnostic-manifest.v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'note': ('Diagnostic additions made AFTER the locked benchmark was read. '
                 'The frozen manifest is referenced by digest and is not modified. '
                 'It remains the record of the code state that produced the locked '
                 'numbers. Declared reasons record purpose, not a proof that '
                 'default outputs are equivalent.'),
        'frozen_manifest': {'path': str(frozen_path), 'sha256': frozen_digest,
                            'entries': len(files)},
        'changed': changed, 'added_since_freeze': added, 'missing': missing,
    }
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')
        print('the frozen manifest was NOT modified')

    if a.check:
        if undeclared or missing:
            print(f'\nFAIL: {len(undeclared)} undeclared change(s), '
                  f'{len(missing)} missing file(s)')
            raise SystemExit(1)
        print('\nOK: every inventory difference is declared; frozen manifest unchanged')
        print('Registration alone does not certify default-output equivalence.')


if __name__ == '__main__':
    main()
