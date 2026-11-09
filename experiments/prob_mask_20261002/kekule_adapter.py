"""Convert all-aromatic bond encodings to a Kekule form, without reading truth.

Why this exists. The external-state comparison hands two bond graphs to one
builder. One of them encodes delocalized systems as class-4 (aromatic) and the
other as alternating 1/2. Our builder sanitizes the second far more often, so the
comparison confounds bond-graph quality with representation compatibility. An
adapter that normalises the encoding before building removes that confound.

The rule reads only the predicted graph: never a reference label, never a
reference SMILES, never the atom states. It is therefore applicable to any arm,
and it is the identity on a graph that contains no class-4 edge -- which is the
point, not an asymmetry: both arms are passed through the same function, and a
graph already in Kekule form needs no conversion.

Rule. Take the subgraph of class-4 edges. In each connected component, find a
maximum matching: every matched edge becomes a double bond (2) and every other
edge in the component becomes single (1). An atom left unmatched carries no
double bond, which is the correct outcome for the heteroatoms that contribute a
lone pair (pyridine N, furan O) rather than a pi electron. If a component admits
no matching at all, its edges are left as class-4 and the builder sees the
original encoding.

This is a representation change, not a correction: it never adds, removes or
retypes a bond outside the class-4 subgraph, and the total number of edges is
preserved.

    python kekule_adapter.py --selftest
"""
import argparse
import collections


def _search(edges, idx, used, chosen, best):
    """Exact maximum matching by backtracking. Components are ring systems and
    stay small; the caller falls back to greedy above a size bound."""
    remaining = len(edges) - idx
    if len(chosen) + remaining <= len(best[0]):
        return
    if idx == len(edges):
        if len(chosen) > len(best[0]):
            best[0] = list(chosen)
        return
    i, j = edges[idx]
    if i not in used and j not in used:
        used.add(i); used.add(j); chosen.append((i, j))
        _search(edges, idx + 1, used, chosen, best)
        chosen.pop(); used.discard(i); used.discard(j)
    _search(edges, idx + 1, used, chosen, best)


def _greedy(edges):
    used, chosen = set(), []
    for i, j in edges:
        if i not in used and j not in used:
            used.add(i); used.add(j); chosen.append((i, j))
    return chosen


MAX_EXACT_EDGES = 22


def maximum_matching(nodes, edges):
    """Return a set of vertex-disjoint edges.

    Correctness (every atom in at most one double bond) is enforced by an
    assertion; maximality is exact for small components and greedy for large
    ones. A greedy fallback can leave an atom unmatched, which is a valid
    Kekule outcome, so it never produces an illegal graph.
    """
    edges = [(min(i, j), max(i, j)) for i, j in edges]
    if len(edges) <= MAX_EXACT_EDGES:
        best = [[]]
        _search(edges, 0, set(), [], best)
        pairs = best[0]
    else:
        pairs = _greedy(edges)
    seen = set()
    for i, j in pairs:
        assert i not in seen and j not in seen, (
            f'illegal matching: atom reused in {pairs}')
        seen.add(i); seen.add(j)
    return set(pairs)


def kekulize_graph(bonds):
    """bonds: {(i,j): label}. Returns a new dict with class-4 resolved to 1/2.

    Components that admit no matching keep their class-4 labels.
    """
    arom = [e for e, c in bonds.items() if c == 4]
    if not arom:
        return dict(bonds), 0, 0
    adj = collections.defaultdict(set)
    for i, j in arom:
        adj[i].add(j)
        adj[j].add(i)
    seen, components = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            seen.add(u)
            stack.extend(adj[u] - comp)
        components.append(comp)

    out = dict(bonds)
    converted = failed = 0
    for comp in components:
        edges = [e for e in arom if e[0] in comp and e[1] in comp]
        pairs = maximum_matching(comp, edges)
        if not pairs:
            failed += 1
            continue
        for e in edges:
            out[e] = 2 if (min(e), max(e)) in pairs else 1
        converted += 1
    return out, converted, failed


def _selftest():
    # benzene: six-ring, every atom matched, three double bonds
    ring = {(0, 1): 4, (1, 2): 4, (2, 3): 4, (3, 4): 4, (4, 5): 4, (0, 5): 4}
    out, conv, fail = kekulize_graph(ring)
    doubles = sum(1 for c in out.values() if c == 2)
    assert conv == 1 and fail == 0, (conv, fail)
    assert doubles == 3, out
    assert all(c in (1, 2) for c in out.values()), out
    deg = collections.Counter()
    for (i, j), c in out.items():
        if c == 2:
            deg[i] += 1; deg[j] += 1
    assert all(v == 1 for v in deg.values()), deg

    # pyrrole-like five-ring: one atom must stay unmatched
    five = {(0, 1): 4, (1, 2): 4, (2, 3): 4, (3, 4): 4, (0, 4): 4}
    out, conv, fail = kekulize_graph(five)
    assert conv == 1 and fail == 0
    assert sum(1 for c in out.values() if c == 2) == 2, out
    deg = collections.Counter()
    for (i, j), c in out.items():
        if c == 2:
            deg[i] += 1; deg[j] += 1
    assert all(v == 1 for v in deg.values()), deg
    assert len([a for a in range(5) if deg[a] == 0]) == 1

    # no aromatic edges: identity, and edge count preserved
    kek = {(0, 1): 1, (1, 2): 2, (2, 3): 1}
    out, conv, fail = kekulize_graph(kek)
    assert out == kek and conv == 0 and fail == 0

    # mixed: only the class-4 part is touched
    mixed = {(0, 1): 1, (1, 2): 4, (2, 3): 4, (1, 3): 4, (3, 4): 3}
    out, conv, fail = kekulize_graph(mixed)
    assert out[(0, 1)] == 1 and out[(3, 4)] == 3, out
    assert len(out) == len(mixed)
    assert all(out[e] in (1, 2) for e in [(1, 2), (2, 3), (1, 3)]), out

    # two disjoint rings are handled independently
    two = dict(ring)
    two.update({(10, 11): 4, (11, 12): 4, (12, 13): 4,
                (13, 14): 4, (14, 15): 4, (10, 15): 4})
    out, conv, fail = kekulize_graph(two)
    assert conv == 2 and fail == 0
    assert sum(1 for c in out.values() if c == 2) == 6

    # fused bicyclic with an odd ring (not bipartite): the old bipartite
    # augmenting-path code produced an illegal matching here
    fused = {}
    for e in [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5),(3,6),(6,7),(7,4)]:
        fused[e] = 4
    out, conv, fail = kekulize_graph(fused)
    deg = collections.Counter()
    for (i, j), c in out.items():
        if c == 2:
            deg[i] += 1; deg[j] += 1
    assert all(v <= 1 for v in deg.values()), ('illegal', out, deg)
    assert len(out) == len(fused)

    # exhaustive legality check on random graphs
    import random
    rnd = random.Random(0)
    for _ in range(300):
        n = rnd.randint(3, 9)
        g = {}
        for i in range(n):
            for j in range(i+1, n):
                if rnd.random() < 0.45:
                    g[(i, j)] = 4
        if not g:
            continue
        out, _, _ = kekulize_graph(g)
        assert len(out) == len(g)
        d = collections.Counter()
        for (i, j), c in out.items():
            assert c in (1, 2, 4), out
            if c == 2:
                d[i] += 1; d[j] += 1
        assert all(v <= 1 for v in d.values()), ('illegal', g, out)

    print('kekule_adapter._selftest: 7 cases OK (benzene, five-ring, identity, '
          'mixed, two components, fused odd ring, 300 random graphs)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    if ap.parse_args().selftest:
        _selftest()
    else:
        ap.error('this module is a library; run --selftest or import kekulize_graph')
