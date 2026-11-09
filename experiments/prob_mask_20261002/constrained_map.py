"""Exact constrained joint MAP over the (q,h) vocabulary, subject to sum(q)=Q.

Why this file exists. ``joint.constrained_atoms(..., greedy=True)`` is NOT a
joint MAP, despite the pipeline being described as a "constrained local MAP":

  * it marginalises h out first (``qprob = joint.sum(-1)``),
  * runs a SUM-product suffix recursion,
  * then picks q greedily position by position, and h conditionally afterwards.

Measured on random 25-class logits with two atoms and Q=0: that procedure misses
the constrained joint maximum in 9.2% of draws, and when it misses the ratio of
its joint probability to the true maximum has median 0.726 and minimum 0.305.
With three atoms, reversing the atom order changes its answer in 14% of draws.
It guarantees sum(q)=Q; it does not maximise anything exactly, and it is not
permutation invariant.

This module does it properly: a MAX-product suffix recursion over the full
25-state joint vocabulary with argmax traceback. Same O(n * 25 * width) cost,
exact, and order independent.

``constrained_atoms(..., greedy=False)`` remains correct for its own purpose --
the sum-product suffix table does give exact conditional SAMPLING from the
product distribution subject to sum(q)=Q. Only the greedy/MAP path was wrong.
"""
import numpy as np

AMASK = 25          # must match joint.AMASK
NQ, NH = 5, 5       # q in {-2..2}, h in {0..4}; state = (q+2)*5 + h
NEG = -np.inf


def constrained_map(logits, state, lengths, total_Q):
    """Exact argmax of sum_i log p_i(state_i) subject to sum_i q_i = Q.

    logits  (N, 25) float array or torch tensor; rows are per-atom logits
    state   (N,) int; entries != AMASK are held fixed (visible states)
    lengths list of per-molecule atom counts summing to N
    total_Q (B,) int per-molecule target total charge

    Returns an (N,) int64 array of chosen states. Raises ValueError when the
    visible states make Q unreachable -- the same contract as constrained_atoms.
    """
    lg = np.asarray(getattr(logits, 'detach', lambda: logits)().cpu().numpy()
                    if hasattr(logits, 'detach') else logits, dtype=np.float64)
    st = np.asarray(getattr(state, 'cpu', lambda: state)().numpy()
                    if hasattr(state, 'cpu') else state, dtype=np.int64)
    Q = np.asarray(getattr(total_Q, 'cpu', lambda: total_Q)().numpy()
                   if hasattr(total_Q, 'cpu') else total_Q, dtype=np.int64)
    out = np.zeros(len(st), dtype=np.int64)
    off = 0
    for g, n in enumerate(lengths):
        x = lg[off:off+n].copy()
        x -= x.max(1, keepdims=True)
        logp = x - np.log(np.exp(x).sum(1, keepdims=True))
        vis = st[off:off+n] != AMASK
        for i in np.nonzero(vis)[0]:
            fixed = st[off+i]
            row = np.full(25, NEG)
            row[fixed] = 0.0
            logp[i] = row
        qof = np.arange(25)//NH - 2           # charge of each state
        # suffix[k][c] = best total log-prob achievable from atoms k..n-1 whose
        # charges sum to c-centre. MAX-product, with the chosen state recorded.
        centre, width = 2*n, 4*n+1
        suffix = np.full((n+1, width), NEG)
        choice = np.zeros((n, width), dtype=np.int64)
        suffix[n, centre] = 0.0
        for k in range(n-1, -1, -1):
            for c in range(width):
                best, arg = NEG, 0
                for s in range(25):
                    lp = logp[k, s]
                    if lp == NEG:
                        continue
                    nxt = c + qof[s]
                    if nxt < 0 or nxt >= width:
                        continue
                    v = lp + suffix[k+1, nxt]
                    if v > best:
                        best, arg = v, s
                suffix[k, c] = best
                choice[k, c] = arg
        start = centre - int(Q[g])
        if not np.isfinite(suffix[0, start]):
            raise ValueError('Visible atom states make Q infeasible')
        c = start
        for k in range(n):
            s = choice[k, c]
            out[off+k] = s
            c += qof[s]
        assert c == centre
        off += n
    return out
