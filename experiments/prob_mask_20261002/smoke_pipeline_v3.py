"""Development-only end-to-end smoke; never loads the locked test set."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(ROOT/'work/evaluation_fix_v3/end_to_end'))
    a = ap.parse_args()
    d = Path(a.outdir); d.mkdir(parents=True, exist_ok=True)
    sdf = d/'development10.sdf'; writer = Chem.SDWriter(str(sdf))
    count = 0
    for mol in Chem.SDMolSupplier(str(ROOT/'data/pdbbind_ccd_pose500.sdf'), removeHs=False):
        if mol is not None:
            writer.write(mol); count += 1
            if count == 10: break
    writer.close()
    if count != 10: raise ValueError('Need 10 development references')
    from population_manifest import population
    from evaluation_common import ALLOWED
    pop = d/'development10_population.json'
    pop.write_text(json.dumps(population(sdf, ALLOWED), indent=1), encoding='utf-8')

    def run(script, *args):
        subprocess.run([sys.executable, str(HERE/script), *map(str, args)], cwd=ROOT, check=True)
    run('test_eval_v3.py')
    run('pose_ablate_search.py', '--sdf', sdf, '--reference_from_sdf', '--device', 'cpu',
        '--population_manifest', pop, '--joint_ckpt', HERE/'joint_v5/seed20261002/best.pt', '--outdir', d, '--tag', 'smoke')
    beam = d/'pose_localmap_smoke.json'
    run('native_ladder_stats.py', '--beam', beam, '--reference_sdf', sdf,
        '--expected_n', 10, '--population_manifest', pop, '--output', d/'native_ladder.json')
    for sel in ('greedy', 'bondonly', 'localmap'):
        run('unified_resonance.py', '--per_edge', d/f'pose_{sel}_smoke_fullgraph.jsonl',
            '--reference_sdf', sdf, '--expected_n', 10, '--population_manifest', pop, '--output', d/f'{sel}.json')
    run('paired_verdicts.py', '--old_verdicts', d/'greedy.json', '--new_verdicts', d/'localmap.json',
        '--reference_sdf', sdf, '--expected_n', 10, '--population_manifest', pop, '--output', d/'paired.json')
    run('ablate_aggregate.py', '--runs', beam, '--labels', 'smoke', '--output', d/'ladder.json')
    run('holm_family.py', '--inputs', d/'paired.json', d/'native_ladder.json',
        '--expected_tests', 5, '--output', d/'holm.json')
    modules = ['pose_ablate_search.py','pose_localmap.py','joint.py','jointt.py','probe.py',
               'evaluation_common.py','unified_resonance.py','paired_verdicts.py',
               'cluster_bootstrap.py','native_ladder_stats.py','ablate_aggregate.py',
               'holm_family.py','test_eval_v3.py','smoke_pipeline_v3.py','population_manifest.py']
    hashes = {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in modules}
    (d/'PASSED.json').write_text(json.dumps({'development_only': True, 'n': 10,
        'passed': True, 'source_sha256': hashes}, indent=1), encoding='utf-8')
    print('END-TO-END DEVELOPMENT SMOKE PASSED', d)


if __name__ == '__main__': main()
