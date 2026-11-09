"""K-best product-state enumeration, conditional on Q and optional charge mass.

No target q/h, bond truth, molecular identity, or stratum is an input here.
This enumerates a frozen model's factorized atom-state scores, not a calibrated
molecular posterior and not a learned autoregressive/diffusion model.
"""
import itertools
import numpy as np


def kbest(logits, Q, k, charge_mass=False):
    values = np.asarray(logits, dtype=np.float64)
    n = len(values)
    assert values.shape == (n, 25) and n > 0 and k > 0 and np.isfinite(values).all()
    if abs(Q) > 2*n: return {}
    if charge_mass:
        p0, n0 = max(Q, 0), max(-Q, 0)
        pmax, nmax = p0+2, n0+2
        pp, nn = np.indices((pmax+1, nmax+1))
        pp, nn = pp.ravel(), nn.ravel(); width = len(pp)
        start = 0
        targets = {j: (p0+j)*(nmax+1)+n0+j for j in range(3)}
        transitions = {}
        for q in range(-2, 3):
            dp, dn = max(q, 0), max(-q, 0)
            src = np.flatnonzero((pp+dp <= pmax) & (nn+dn <= nmax))
            dst = src + dp*(nmax+1)+dn
            transitions[q] = (src, dst, dp*(nmax+1)+dn)
    else:
        width = 4*n+1; start = 2*n
        targets = {'Q': start+Q}; transitions = {}
        for q in range(-2, 3):
            src = np.arange(max(0, -q), min(width, width-q))
            transitions[q] = (src, src+q, q)
    prev = np.full((width, k), -np.inf); prev[start, 0] = 0.
    back = []
    for node in range(n):
        options = np.full((width, 25*k), -np.inf)
        for state in range(25):
            src, dst, _ = transitions[state//5-2]
            options[dst, state*k:(state+1)*k] = prev[src] + values[node, state]
        chosen = np.argpartition(-options, k-1, axis=1)[:, :k]
        scores = np.take_along_axis(options, chosen, axis=1)
        order = np.argsort(-scores, axis=1, kind='stable')
        chosen = np.take_along_axis(chosen, order, axis=1)
        prev = np.take_along_axis(scores, order, axis=1)
        back.append(chosen.astype(np.int32))
    result = {}
    for level, target in targets.items():
        items = []
        for rank in range(k):
            score = prev[target, rank]
            if not np.isfinite(score): continue
            resource = target; r = rank; states = []
            for node in range(n-1, -1, -1):
                pick = int(back[node][resource, r]); state, r = divmod(pick, k)
                states.append(state); resource -= transitions[state//5-2][2]
            states.reverse()
            assert resource == start and sum(s//5-2 for s in states) == Q
            if charge_mass:
                assert sum(max(s//5-2, 0) for s in states) == max(Q, 0)+level
            assert abs(sum(values[i, s] for i, s in enumerate(states))-score) < 1e-7
            items.append({'states': states, 'score': float(score), 'positive_mass_level': level if charge_mass else None})
        assert len({tuple(x['states']) for x in items}) == len(items)
        result[level] = items
    return result


def proposals(logits, Q, baseline, strategy):
    """At most 32 raw proposals including unchanged baseline, no target access."""
    base = {'states': list(baseline), 'score': float(sum(logits[i, s] for i, s in enumerate(baseline))),
            'positive_mass_level': sum(max(s//5-2, 0) for s in baseline)-max(Q, 0), 'baseline': True}
    if strategy == 'plain_kbest':
        extra = kbest(logits, Q, 31)['Q']
    elif strategy == 'mass_stratified':
        levels = kbest(logits, Q, 11, charge_mass=True)
        extra = []
        # Fixed round-robin quota: 11 at base mass, 10 at +1 and +2.
        for rank in range(11):
            for level, quota in ((0, 11), (1, 10), (2, 10)):
                if rank < quota and rank < len(levels[level]): extra.append(levels[level][rank])
    else:
        raise ValueError(strategy)
    assert len(extra) <= 31
    return [base]+[dict(x, baseline=False) for x in extra]


def selftest():
    rng = np.random.default_rng(20261102)
    for n in (1, 2, 3):
        values = rng.normal(size=(n, 25))
        for Q in (-1, 0, 1):
            brute = []
            for states in itertools.product(range(25), repeat=n):
                if sum(s//5-2 for s in states) == Q:
                    brute.append((sum(values[i, s] for i, s in enumerate(states)), states))
            brute.sort(reverse=True)
            plain = kbest(values, Q, 7)['Q']
            assert np.allclose([x['score'] for x in plain], [x[0] for x in brute[:7]])
            diverse = kbest(values, Q, 7, True)
            for level, items in diverse.items():
                subset = [x for x in brute if sum(max(s//5-2, 0) for s in x[1]) == max(Q, 0)+level]
                assert np.allclose([x['score'] for x in items], [x[0] for x in subset[:7]])
    # A charge-separated pair can be a low-probability alternative to neutral Q0.
    values = np.full((2, 25), -30.)
    values[:, 10] = 0.; values[0, 15] = -4.; values[1, 5] = -4.
    neutral = [10, 10]
    diverse = proposals(values, 0, neutral, 'mass_stratified')
    assert any(x['states'] == [15, 5] for x in diverse)
    assert diverse[0]['states'] == neutral and len(diverse) <= 32
    assert not kbest(np.zeros((1, 25)), 3, 2)
    print('k-best selftest passed: brute-force n=1..3, Q=-1/0/+1, positive mass levels, charge separation, infeasible Q', flush=True)


if __name__ == '__main__': selftest()
