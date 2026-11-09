"""Post-outcome exploratory analysis. Never changes frozen study artifacts."""
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
import joint as J
from verify_outputs import build

SEEDS = (20261002, 20261003, 20261004)
ARMS = ('neural_constrained', 'catalogue_constrained', 'neural_unconstrained')
GROUPS = ('cation', 'anion', 'separated_neutral', 'uncharged_neutral')
REPS = 2000
OUT = HERE/'exploratory_conditional'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sufficient(record, arm):
    a = record['arms'][arm]
    truth = np.array(record['target_q']) + 2
    pred = np.array(a['atom_states'])//5
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    assert bool(np.all(truth == pred)) == a['q_exact']
    assert sum(pred-2) == a['qsum']
    assert (a['qsum'] == record['Q']) == a['charge_sum_correct']
    # Native equality necessarily implies the same total formal charge.
    assert not a['native'] or a['charge_sum_correct']
    return cm


def f1(cm, classes):
    den = cm.sum(0) + cm.sum(1)
    values = np.divide(2*np.diag(cm), den, out=np.zeros(5, dtype=float), where=den != 0)
    return float(values[classes].mean())


def measure(records, arm, indices, classes):
    if not len(indices):
        return {'n': 0, 'native': None, 'q_exact': None, 'charge_sum_correct': None,
                'q_macro_f1': None}
    chosen = [records[i] for i in indices]
    cm = sum((sufficient(r, arm) for r in chosen), np.zeros((5, 5), dtype=np.int64))
    return {'n': len(chosen), **{k: statistics.mean(int(r['arms'][arm][k]) for r in chosen)
                               for k in ('native', 'q_exact', 'charge_sum_correct')},
            'q_macro_f1': f1(cm, classes), 'q_confusion_minus2_to_plus2': cm.tolist()}


def paired_bootstrap(records, indices, metric, conditional=False, classes=None):
    """Per-head paired molecule bootstrap, fixed truth-present class set.

    Conditional analysis resamples the observed S only. It does not account for
    choosing S or turn that selected-subset contrast into a causal estimand.
    """
    if not len(indices):
        return None
    chosen = [records[i] for i in indices]
    rng = np.random.default_rng(20261102)
    draws = rng.integers(0, len(chosen), size=(REPS, len(chosen)))
    if metric == 'q_macro_f1':
        if classes is None:
            classes = np.flatnonzero(sum((sufficient(r, ARMS[0]) for r in chosen),
                                        np.zeros((5, 5), dtype=int)).sum(1))
        cms = {a: np.stack([sufficient(r, a) for r in chosen]) for a in (ARMS[0], ARMS[2])}
        delta = []
        for ids in draws:
            delta.append(f1(cms[ARMS[0]][ids].sum(0), classes) -
                         f1(cms[ARMS[2]][ids].sum(0), classes))
    else:
        values = np.array([int(r['arms'][ARMS[0]][metric])-int(r['arms'][ARMS[2]][metric])
                           for r in chosen])
        delta = values[draws].mean(1)
    return {'ci95': np.quantile(delta, [.025, .975]).tolist(), 'reps': REPS,
            'unit': 'paired molecule', 'scope': 'observed common S only' if conditional else 'all molecules',
            'training_uncertainty_included': False}


def analyze_scope(records):
    all_indices = np.arange(len(records))
    S = np.array([i for i, r in enumerate(records) if r['arms'][ARMS[2]]['charge_sum_correct']], dtype=int)
    classes = np.flatnonzero(sum((sufficient(r, ARMS[0]) for r in records), np.zeros((5, 5), int)).sum(1))
    out = {'n': len(records), 'S_n': len(S), 'S_fraction': len(S)/len(records),
           'S_names': [records[i]['name'] for i in S], 'truth_present_q_classes': (classes-2).tolist(),
           'all': {}, 'common_S': {}}
    for arm in ARMS:
        out['all'][arm] = measure(records, arm, all_indices, classes)
        out['common_S'][arm] = measure(records, arm, S, classes)
    out['contrasts_constrained_minus_unconstrained'] = {}
    for scope, ids in (('all', all_indices), ('common_S', S)):
        item = {}
        for metric in ('native', 'q_exact', 'q_macro_f1'):
            left, right = out[scope][ARMS[0]][metric], out[scope][ARMS[2]][metric]
            item[metric] = {'delta': None if left is None else left-right,
                            'bootstrap': paired_bootstrap(records, ids, metric, scope == 'common_S', classes)}
        item['native_rescued'] = sum(not records[i]['arms'][ARMS[2]]['native'] and
                                    records[i]['arms'][ARMS[0]]['native'] for i in ids)
        item['native_harmed'] = sum(records[i]['arms'][ARMS[2]]['native'] and
                                   not records[i]['arms'][ARMS[0]]['native'] for i in ids)
        item['selected_candidate_changed'] = sum(records[i]['arms'][ARMS[0]]['selected_index'] !=
                                                records[i]['arms'][ARMS[2]]['selected_index'] for i in ids)
        out['contrasts_constrained_minus_unconstrained'][scope] = item
    # Exact decomposition of the *unconstrained* native probability.
    free = out['all'][ARMS[2]]['native']
    cond = out['common_S'][ARMS[2]]['native']
    assert abs(free - (out['S_fraction'] * cond if cond is not None else 0)) < 1e-12
    return out


def support(z, graph, q, h, catalogue):
    exact, coarse, by_z = catalogue
    neighbors = [[] for _ in z]
    for (i, j), label in graph.items():
        neighbors[i].append((z[j], label)); neighbors[j].append((z[i], label))
    result = []
    for i, zi in enumerate(z):
        key = (zi, tuple(sorted(neighbors[i])))
        ck = (zi, tuple(sorted(c for _, c in neighbors[i])))
        level = 'exact' if key in exact else 'coarse' if ck in coarse else 'element' if zi in by_z else 'uniform'
        counts = exact[key] if level == 'exact' else coarse[ck] if level == 'coarse' else by_z.get(zi, np.ones(25))
        state = (q[i]+2)*5+h[i]
        result.append({'atom': i, 'z': zi, 'q': q[i], 'h': h[i], 'backoff': level,
                       'target_state_empirical_count': float(counts[state]),
                       'target_q_empirical_count': float(np.asarray(counts).reshape(5, 5)[q[i]+2].sum())})
    return result


def diagnose(record, row, feat, catalogue):
    z = row['z'].tolist(); pairs = feat['pairs'].tolist()
    ref_labels = [row['truth'].get(tuple(p), 0) for p in pairs]
    assert {tuple(p): c for p, c in zip(pairs, ref_labels) if c} == row['truth']
    trueq, trueh = record['target_q'], record['target_h']
    out = {'name': record['name'], 'component_id': row['component_id'], 'Q': record['Q'],
           'target_smiles': record['target_smiles'], 'reference_context_support': support(z, row['truth'], trueq, trueh, catalogue),
           'arms': {}}
    for arm in ARMS:
        a = record['arms'][arm]
        predq = [s//5-2 for s in a['atom_states']]; predh = [s%5 for s in a['atom_states']]
        graph = {tuple(p): c for p, c in zip(pairs, a['labels']) if c}
        controls = {}
        # Eight oracle replacement controls. Not re-decoding, not model scores,
        # not an additive causal decomposition. Reference identity is used here.
        for B in ('pred', 'ref'):
            for q in ('pred', 'ref'):
                for h in ('pred', 'ref'):
                    qs = predq if q == 'pred' else trueq
                    hs = predh if h == 'pred' else trueh
                    smi, good = build(z, pairs, a['labels'] if B == 'pred' else ref_labels,
                                      [(qi+2)*5+hi for qi, hi in zip(qs, hs)])
                    controls[f'B_{B}_q_{q}_h_{h}'] = {'constructible': bool(good),
                        'native': bool(good and smi == record['target_smiles'])}
        assert controls['B_pred_q_pred_h_pred']['native'] == a['native']
        assert controls['B_ref_q_ref_h_ref']['native']
        out['arms'][arm] = {'native': a['native'], 'strict': a['strict'], 'q_exact': a['q_exact'],
                            'h_exact': a['h_exact'], 'constructible': a['constructible'],
                            'qsum': a['qsum'], 'selected_index': a['selected_index'],
                            'q_error_atoms': [i for i in range(len(z)) if trueq[i] != predq[i]],
                            'h_error_atoms': [i for i in range(len(z)) if trueh[i] != predh[i]],
                            'q_on_reference_charged_atoms': [{'atom': i, 'z': z[i], 'truth_q': trueq[i], 'pred_q': predq[i]}
                                                             for i in range(len(z)) if trueq[i]],
                            'oracle_replacement_controls': controls,
                            'predicted_context_target_support': support(z, graph, trueq, trueh, catalogue)}
    return out


def selftest():
    def r(name, cq, uq, cn, un, ce, ue):
        def a(q, n, e):
            return {'atom_states': [(x+2)*5 for x in q], 'selected_index': 0, 'qsum': sum(q), 'charge_sum_correct': sum(q) == 0,
                    'native': n, 'q_exact': e}
        return {'name': name, 'Q': 0, 'target_q': [1, -1],
                'arms': {ARMS[0]: a(cq, cn, ce), ARMS[1]: a(cq, cn, ce), ARMS[2]: a(uq, un, ue)}}
    records = [r('a', [1, -1], [0, 0], True, False, True, False),
               r('b', [0, 0], [1, -1], False, True, False, True),
               r('c', [1, -1], [1, 0], True, False, True, False)]
    x = analyze_scope(records)
    assert x['S_names'] == ['a', 'b']
    assert x['contrasts_constrained_minus_unconstrained']['common_S']['native_rescued'] == 1
    assert x['contrasts_constrained_minus_unconstrained']['common_S']['native_harmed'] == 1
    assert x['contrasts_constrained_minus_unconstrained']['common_S']['native']['delta'] == 0
    assert abs(x['all'][ARMS[2]]['native'] - 1/3) < 1e-12
    assert f1(np.eye(5, dtype=int), np.arange(5)) == 1
    assert paired_bootstrap(records, np.array([], dtype=int), 'native', True) is None


def pct(value):
    return 'NA' if value is None else f'{100*value:.2f}'


def report(result):
    lines = ['# Post-outcome exploratory charge analysis', '',
             'Saved predictions only; no training/inference. Not preregistered before outcomes. No new p-values; original 30-test Holm family unchanged.', '',
             'S is defined separately for each head by the unconstrained output satisfying sum(q)=Q. Both arms are evaluated on exactly the same S. Selection bias remains; this is not a fixed-B causal control.', '',
             '## Conditional comparison (percent, each head separately)', '',
             '| Group | Seed | S / n | Native C / U on S | q-vector exact C / U on S | Native rescued / harmed | Changed B candidate on S |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in ('pooled', *GROUPS):
        for seed in SEEDS:
            x = result['cells'][group][str(seed)]
            c, u = x['common_S'][ARMS[0]], x['common_S'][ARMS[2]]
            d = x['contrasts_constrained_minus_unconstrained']['common_S']
            lines.append(f"| {group} | {seed} | {x['S_n']}/{x['n']} | {pct(c['native'])} / {pct(u['native'])} | {pct(c['q_exact'])} / {pct(u['q_exact'])} | {d['native_rescued']} / {d['native_harmed']} | {d['selected_candidate_changed']} |")
    lines += ['', '## All-sample charge metrics (mean ± sample SD across 3 heads, percent)', '',
              '| Group | Arm | sum(q)=Q | Charge macro-F1 | Native (secondary here) |', '|---|---|---:|---:|---:|']
    for group in ('pooled', *GROUPS):
        for arm in ARMS:
            values = [result['cells'][group][str(s)]['all'][arm] for s in SEEDS]
            def fmt(k):
                v = [x[k] for x in values]
                return f'{100*statistics.mean(v):.2f} ± {100*statistics.stdev(v):.2f}'
            lines.append(f"| {group} | {arm} | {fmt('charge_sum_correct')} | {fmt('q_macro_f1')} | {fmt('native')} |")
    lines += ['', '## Charge-separated neutral diagnostics', '',
              'Counts below are per head (23 molecules), not 69 independent molecules. Replacing a predicted block by reference labels is an oracle compatibility probe, not a deployable score or additive attribution.', '',
              '| Arm | Seed | Native | Strict B | q exact | h exact | Native if ref B only | Native if ref q only | Native if ref h only | Native if ref q+h | Native if ref B+q | Native if ref B+h |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            rows = [r['arms'][arm] for r in result['separated_diagnostics'][str(seed)]]
            vals = [sum(r[k] for r in rows) for k in ('native', 'strict', 'q_exact', 'h_exact')]
            keys = ('B_ref_q_pred_h_pred', 'B_pred_q_ref_h_pred', 'B_pred_q_pred_h_ref',
                    'B_pred_q_ref_h_ref', 'B_ref_q_ref_h_pred', 'B_ref_q_pred_h_ref')
            vals += [sum(r['oracle_replacement_controls'][k]['native'] for r in rows) for k in keys]
            lines.append('| '+ ' | '.join([arm, str(seed), *map(str, vals)])+' |')
    lines += ['', '## Interpretation limits', '',
              '- Charge macro-F1 uses classes present in each full reference stratum, fixed during molecule bootstrap; confusion matrices retain all five classes.',
              '- Bootstrap CIs are per head, conditional on that head and observed S; they do not pool 3×270 as independent molecules.',
              '- Indexed q-vector/h exactness penalizes resonance or atom-symmetry localization differences; native can be correct even when indexed vectors differ.',
              '- Catalogue support is empirical training frequency in the actual exact/coarse/element backoff context. Zero count is not impossibility: the original prior adds a positive pseudocount.',
              '- Substituting reference q/h into an incorrect B may be chemically inconsistent. Oracle controls reveal compatibility, not uniquely attributable root causes.',
              '- This is CCD ideal geometry with curated reference states, not measured protonation or real bound-pose validation.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    RDLogger.DisableLog('rdApp.*')
    selftest()
    original = json.loads((HERE/'PROTOCOL.json').read_text())
    files = [HERE/'PROTOCOL.json', HERE/'SUMMARY.json', HERE/'VERIFICATION.json', HERE/'selected.pt', HERE/'features.pt',
             ROOT/'experiments/prob_mask_20261002/joint_v1/sources.pt', HERE/'verify_outputs.py']
    files += [HERE/f'evaluations/seed{s}/{f}' for s in SEEDS for f in ('predictions.json.gz', 'DONE.json')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for rel, digest in original['spec']['sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    plan = {'status': 'post-outcome exploratory; not confirmatory preregistration',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'original_protocol_sha256': sha(HERE/'PROTOCOL.json'),
            'analysis_sha256': sha(Path(__file__)), 'inputs_sha256': before,
            'S_rule': 'unconstrained selected q sums to reference Q; identical per-head S for both arms',
            'metrics': ['all-population charge conservation', 'all-population charge macro-F1',
                        'native on common S', 'indexed full q-vector exact on common S'],
            'bootstrap': {'reps': REPS, 'seed': 20261102, 'unit': 'paired molecule, separately per head',
                          'F1_classes': 'fixed truth-present classes from original scope'},
            'hypothesis_tests': 0, 'original_Holm_family_modified': False,
            'separated_diagnostics': 'all 23 cases, all 3 heads and arms; 2x2x2 reference/predicted B/q/h oracle rebuilds; catalogue context empirical support'}
    plan_path = OUT/'ANALYSIS_PLAN.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        plan['created_utc'] = old['created_utc']
        assert plan == old, 'Exploratory analysis drift; use a new version, do not overwrite provenance'
    else:
        save(plan_path, plan)
    data = {}
    for seed in SEEDS:
        folder = HERE/f'evaluations/seed{seed}'
        done = json.loads((folder/'DONE.json').read_text())
        assert done['sha256'] == sha(folder/'predictions.json.gz')
        with gzip.open(folder/'predictions.json.gz', 'rt') as f:
            raw = json.load(f)
        assert raw['protocol_sha256'] == sha(HERE/'PROTOCOL.json')
        data[seed] = sorted(raw['records'], key=lambda r: r['name'])
    assert all([r['name'] for r in data[s]] == [r['name'] for r in data[SEEDS[0]]] for s in SEEDS)
    result = {'status': plan['status'], 'analysis_plan_sha256': sha(plan_path), 'cells': {}, 'separated_diagnostics': {}}
    for group in ('pooled', *GROUPS):
        result['cells'][group] = {str(s): analyze_scope([r for r in data[s] if group == 'pooled' or r['bucket'] == group]) for s in SEEDS}
        print('analyzed', group, flush=True)
    rows = {r['id']: r for r in torch.load(HERE/'selected.pt', weights_only=False, map_location='cpu')}
    feats = {r['id']: r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    sources = torch.load(ROOT/'experiments/prob_mask_20261002/joint_v1/sources.pt', weights_only=False, map_location='cpu')
    catalogue = J.make_catalogue(sources['train'])
    for s in SEEDS:
        result['separated_diagnostics'][str(s)] = [diagnose(r, rows[r['name']], feats[r['name']], catalogue)
                                                for r in data[s] if r['bucket'] == 'separated_neutral']
        assert len(result['separated_diagnostics'][str(s)]) == 23
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in files}
    assert len(json.loads((HERE/'SUMMARY.json').read_text())['tests']) == 30
    result['original_inputs_unchanged'] = True
    save(OUT/'SUMMARY.json', result)
    report(result)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
