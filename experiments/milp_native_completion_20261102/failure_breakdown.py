"""Descriptive output-availability attribution; not a new primary comparison."""
import json
import statistics
import study as S


def main():
    digest=S.check();out={'plan_sha256':digest,'scope':'post hoc descriptive; output-conditioned subsets are not causal ablations','conditions':{}}
    for condition in S.CONDITIONS:
        rows=[json.loads(line) for line in (S.HERE/f'evaluations/{condition}/predictions.jsonl').read_text().splitlines() if line.strip()]
        old={r['name']:r for r in rows};per_head=[]
        for seed in S.SEEDS:
            current={r['name']:r for r in S.read(S.current(condition,seed))['records']}
            native={};built={}
            for name,r in current.items():
                if condition=='locked_clean':c=r['candidates_detail'][r['selected_indices']['localmap']];native[name]=bool(c['correct']);built[name]=bool(c['good'])
                else:c=r['neural_catalogue'];native[name]=bool(c['native']);built[name]=bool(c['constructible'])
            both=[n for n in old if old[n]['final']['good'] and built[n]]
            current_only=[n for n in old if native[n] and not old[n]['final']['native']]
            baseline_only=[n for n in old if old[n]['final']['native'] and not native[n]]
            per_head.append({'seed':seed,'n':len(old),'baseline_buildable':sum(r['final']['good'] for r in old.values()),
                'current_buildable':sum(built.values()),'both_buildable_n':len(both),
                'baseline_native_on_both_buildable':sum(old[n]['final']['native'] for n in both)/len(both) if both else None,
                'current_native_on_both_buildable':sum(native[n] for n in both)/len(both) if both else None,
                'current_only_total':len(current_only),'current_only_baseline_failed_build':sum(not old[n]['final']['good'] for n in current_only),
                'current_only_both_buildable':sum(old[n]['final']['good'] and built[n] for n in current_only),
                'baseline_only_total':len(baseline_only),'baseline_only_current_failed_build':sum(not built[n] for n in baseline_only),
                'baseline_only_both_buildable':sum(old[n]['final']['good'] and built[n] for n in baseline_only)})
        out['conditions'][condition]={'per_head':per_head,
            'mean_current_only_on_baseline_build_failure':statistics.mean(r['current_only_baseline_failed_build'] for r in per_head),
            'mean_baseline_only_on_current_build_failure':statistics.mean(r['baseline_only_current_failed_build'] for r in per_head),
            'mean_current_only_both_buildable':statistics.mean(r['current_only_both_buildable'] for r in per_head),
            'mean_baseline_only_both_buildable':statistics.mean(r['baseline_only_both_buildable'] for r in per_head)}
    S.save(S.HERE/'FAILURE_BREAKDOWN.json',out);S.check();print(json.dumps(out,indent=2))


if __name__=='__main__':main()
