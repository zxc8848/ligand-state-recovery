"""Does a delocalized (class-5) output mode still buy anything after the coupled
second pass?

The archived two-stage decoder emitted class 5 on 282 edges of the 488 poses, all
on reference single/double edges, and the resonance-aware criterion counts that as
representation-equivalent. But that criterion *already* accepts a permutation of
bond orders among a centre's terminal oxygens, so class 5 can only earn credit
where the decoder gets the group's bond-order *multiset* wrong.

Which multiset is "wrong" is a file convention, not a universal rule, and it
depends on the centre element. Taken from the joint training split only:

  C, 2 terminal O : (1,2) in 279/280 groups
  N, 2 terminal O : (1,2) in 453/461
  P, 2 terminal O : (1,2) in 11/11;   P, 3 terminal O : (1,1,2) in 10/10
  S, 2 terminal O : (2,2) in 673/673; S, 3 terminal O : (1,2,2) in 21/21

So a symmetric (2,2) assignment is *correct* on sulfur and wrong on carbon. The
rule below re-expresses a group as delocalized only when its predicted multiset
differs from the training-canonical multiset for that (centre element, number of
terminal oxygens). The table comes from training labels, group eligibility comes
from the predicted adjacency, and no reference label of the evaluated molecule is
consulted.
"""
import argparse
import collections
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--per_edge', default=str(HERE/'joint_pose/pose_localmap_refedges.jsonl'))
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--output', default=str(HERE/'joint_pose/pose_localmap_deloc_refedges.jsonl'))
    a.add_argument('--report', default=str(HERE/'joint_pose/pose_delocalize.json'))
    a.add_argument('--min_support', type=int, default=5)
    return a.parse_args()


def canonical_multisets(sources, min_support):
    """{(centre Z, n terminal O): canonical multiset} from the training split."""
    rows = torch.load(sources, map_location='cpu', weights_only=False)['train']
    obs = collections.defaultdict(collections.Counter)
    for r in rows:
        z = r['z'].tolist()
        adj = collections.defaultdict(set)
        for (i, j), _ in r['truth'].items():
            adj[i].add(j); adj[j].add(i)
        for centre in range(len(z)):
            if z[centre] not in (6, 7, 15, 16):
                continue
            term = [n for n in adj[centre] if z[n] == 8 and len(adj[n]) == 1]
            if len(term) < 2:
                continue
            labels = tuple(sorted(r['truth'].get(tuple(sorted((centre, o))), 0)
                                  for o in term))
            obs[(z[centre], len(term))][labels] += 1
    table = {}
    support = {}
    for key, counter in obs.items():
        total = sum(counter.values())
        if total < min_support:
            continue
        best, n = counter.most_common(1)[0]
        table[key] = best
        support[f'{key[0]}:{key[1]}'] = {'canonical': list(best), 'n': total,
                                         'purity': n/total}
    return table, support


def main():
    a = parse_args()
    table, support = canonical_multisets(a.sources, a.min_support)

    per = collections.defaultdict(list)
    for line in Path(a.per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            per[row['name']].append(row)

    changed_edges = changed_mols = 0
    stats = collections.Counter()
    out = []
    for name, rows in per.items():
        pred = {(r['i'], r['j']): r['pred'] for r in rows}
        z = {}
        for r in rows:
            z[r['i']] = r['z_i']; z[r['j']] = r['z_j']
        adj = collections.defaultdict(set)
        for (i, j), c in pred.items():
            if c:
                adj[i].add(j); adj[j].add(i)
        touched = False
        for centre in sorted(adj):
            zc = z.get(centre)
            if zc not in (6, 7, 15, 16):
                continue
            terminal = [n for n in adj[centre] if z.get(n) == 8 and len(adj[n]) == 1]
            if len(terminal) < 2:
                continue
            key = (zc, len(terminal))
            if key not in table:
                continue
            edges = []
            for o in terminal:
                e = tuple(sorted((centre, o)))
                edges.append(e if e in pred else (e[1], e[0]))
            labels = tuple(sorted(pred[e] for e in edges))
            if labels == table[key]:
                continue
            if any(l not in (1, 2) for l in labels):
                continue
            stats[f'Z{zc}_n{len(terminal)}_pred{"".join(map(str, labels))}'] += 1
            for e in edges:
                pred[e] = 5
                changed_edges += 1
            touched = True
        changed_mols += touched
        for r in rows:
            r = dict(r)
            r['pred'] = pred[(r['i'], r['j'])]
            r['correct'] = int(r['pred'] == r['true'])
            out.append(json.dumps(r))

    Path(a.output).write_text('\n'.join(out)+'\n', encoding='utf-8')
    report = {'molecules': len(per), 'molecules_changed': changed_mols,
              'edges_set_to_delocalized': changed_edges,
              'training_canonical_multisets': support,
              'trigger_counts': dict(stats),
              'rule': 'a terminal-oxygen group whose predicted multiset differs from '
                      'the training-canonical multiset for its (centre element, group '
                      'size) is re-expressed as delocalized; eligibility from the '
                      'predicted adjacency only'}
    Path(a.report).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'training_canonical_multisets'},
                     indent=1))
    print('training canonical multisets:', json.dumps(support, indent=1))


if __name__ == '__main__':
    main()
