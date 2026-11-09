"""Noise-level (t) conditioning for the masked joint (B,h,q) completion network.

The baseline ``joint.JointNet`` has no notion of how corrupted its input is. It is
trained at mask rates sampled per molecule but never told which rate it is looking
at, and at decode time it is called at whatever rate the schedule happens to
produce. That is the one ingredient an absorbing-state discrete diffusion model has
and this network does not, and it is the plausible reason the multi-round
confidence schedule *hurt* (``pose_iterate_summary.json``: species 0.7254 -> 0.7131
over five rounds): a reverse kernel should commit aggressively at high noise and
conservatively at low noise, and without t it can only apply one averaged policy.

``JointNetT`` is ``JointNet`` plus a per-molecule embedding of the two realized
masked fractions (edges and atoms), injected additively into both streams. The
fractions are computed *inside* forward from the state tensors, so every existing
call site -- ``joint.sample``, ``pose_eval.py``, ``pose_localmap.py``,
``pose_confidence.py``, ``pose_headroom.py``, ``localsearch.propose`` -- works
unchanged; only the class to instantiate differs, which ``load_joint`` resolves
from the checkpoint.

Note honestly what this does and does not add: the masked fraction is in principle
recoverable from the input, since the node stream already aggregates per-atom counts
of each bond state including MASK. What is new is (i) a global, explicit and
smoothly parameterised signal rather than a local count, and (ii) in the
``diffusion`` training variant, an explicit schedule over that signal instead of
the baseline's ad-hoc rate mixture.
"""
import math

import torch
from torch import nn

import joint as J

N_FREQ = 4


def masked_fractions(b, bs, ats):
    """Per-molecule (edge, atom) masked fractions, plus the edge->molecule map."""
    device = bs.device
    el = torch.tensor(b['lengths'], device=device)
    al = torch.tensor(b['atom_lengths'], device=device)
    eg = torch.repeat_interleave(torch.arange(len(el), device=device), el)
    ag = b['atom_graph']
    tB = torch.zeros(len(el), device=device, dtype=torch.float32)
    tB.index_add_(0, eg, (bs == J.BMASK).float())
    tB = tB/el.clamp_min(1).float()
    tA = torch.zeros(len(al), device=device, dtype=torch.float32)
    tA.index_add_(0, ag, (ats == J.AMASK).float())
    tA = tA/al.clamp_min(1).float()
    return tB, tA, eg, ag


def t_features(tB, tA, n_freq=N_FREQ):
    """[tB, tA] plus sin/cos at n_freq octaves of each; shape (n_mol, 2+4*n_freq)."""
    feats = [tB[:, None], tA[:, None]]
    for k in range(n_freq):
        w = math.pi*(2**k)
        feats += [torch.sin(w*tB)[:, None], torch.cos(w*tB)[:, None],
                  torch.sin(w*tA)[:, None], torch.cos(w*tA)[:, None]]
    return torch.cat(feats, -1)


class JointNetT(J.JointNet):
    """JointNet + additive per-molecule noise-level embedding."""

    def __init__(self, width=96, layers=2, n_freq=N_FREQ):
        super().__init__(width=width, layers=layers)
        self.n_freq = n_freq
        self.temb = nn.Sequential(nn.Linear(2+4*n_freq, width), nn.SiLU(),
                                  nn.Linear(width, width))
        # start as a no-op so the warm-started edge subnet is not perturbed at init
        nn.init.zeros_(self.temb[-1].weight)
        nn.init.zeros_(self.temb[-1].bias)

    def forward(self, b, bs, ats):
        # This is joint.JointNet.forward with the two t-injection lines added.
        i, j = b['pairs'].unbind(1)
        tB, tA, eg, ag = masked_fractions(b, bs, ats)
        t = self.temb(t_features(tB, tA, self.n_freq))
        h = self.eg(b['feat'])+self.bemb(bs)+t[eg]
        Q = b['Q'][b['atom_graph']].float()
        count = b['atom_counts'][b['atom_graph']].float()
        qfeature = torch.stack([Q, Q.abs(), Q/count], -1)
        n = (self.ng(b['node_feat'])+self.zemb(b['z'])+self.aemb(ats)
             + self.qembed(qfeature)+t[ag])
        deg = torch.zeros(len(n), 1, device=n.device)
        one = torch.ones(len(h), 1, device=n.device)
        deg.index_add_(0, i, one)
        deg.index_add_(0, j, one)
        code = nn.functional.one_hot(bs, 6).float()
        counts = torch.zeros(len(n), 6, device=n.device)
        counts.index_add_(0, i, code)
        counts.index_add_(0, j, code)
        for eb, nb, en, nnorm in zip(self.eblocks, self.nblocks, self.enorm, self.nnorm):
            h = en(h+eb(torch.cat([h, n[i]+n[j], (n[i]-n[j]).abs()], -1)))
            agg = torch.zeros_like(n)
            agg.index_add_(0, i, h)
            agg.index_add_(0, j, h)
            n = nnorm(n+nb(torch.cat([n, agg/deg.clamp_min(1), counts], -1)))
        blogits = self.edge(b['feat'], b['base'], b['pairs'], bs, len(n))+self.bhead(h)
        return blogits, self.ahead(n)


def cosine_gamma(t):
    """Masked fraction at normalised time t in [0,1]; gamma(0)=1, gamma(1)=0."""
    return math.cos(math.pi/2*t)


def load_joint(path, device, map_location='cpu'):
    """Instantiate the right class for a checkpoint. Returns (net, checkpoint)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    args = ck.get('args', {})
    width = args.get('width', 96)
    layers = args.get('layers', 2)
    if args.get('arch') == 't':
        net = JointNetT(width, layers, args.get('n_freq', N_FREQ)).to(device)
    else:
        net = J.JointNet(width, layers).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ck


def tests():
    """t conditioning must change nothing structurally and must actually be used."""
    import numpy as np
    import probe as p
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(5)
    rows = torch.load(p.HERE/'joint_v1/sources.pt', map_location='cpu',
                      weights_only=False)['train'][:8]
    packed = []
    for r in rows:
        pairs = torch.tensor(sorted(r['truth']), dtype=torch.long)
        labels = torch.tensor([r['truth'][tuple(e.tolist())] for e in pairs],
                              dtype=torch.long)
        packed.append(dict(r, pairs=pairs, labels=labels, missed=0, sigma=0.0,
                           feat=torch.randn(len(pairs), 360).half(),
                           node_feat=torch.randn(len(r['z']), 360).half(),
                           base=torch.randn(len(pairs), 5).log_softmax(-1)))
    b = J.pack(packed, 'cpu')
    net = JointNetT()
    net.eval()
    allB = torch.full_like(b['labels'], J.BMASK)
    allA = torch.full_like(b['atarget'], J.AMASK)

    # 1. shapes and the all-masked fractions
    tB, tA, eg, ag = masked_fractions(b, allB, allA)
    assert torch.allclose(tB, torch.ones_like(tB)) and torch.allclose(tA, torch.ones_like(tA))
    half = allB.clone()
    half[::2] = b['labels'][::2]
    tB2, _, _, _ = masked_fractions(b, half, allA)
    assert bool(((tB2 > 0) & (tB2 < 1)).all()), 'partially revealed must give 0<t<1'

    with torch.no_grad():
        bl, al = net(b, allB, allA)
    assert bl.shape == (len(b['labels']), 5) and al.shape == (len(b['atarget']), 25)

    # 2. at initialisation the t head is zeroed, so outputs match the base class
    base = J.JointNet()
    base.load_state_dict({k: v for k, v in net.state_dict().items()
                          if not k.startswith('temb.')}, strict=True)
    base.eval()
    with torch.no_grad():
        bl0, al0 = base(b, allB, allA)
    assert torch.allclose(bl, bl0, atol=1e-6) and torch.allclose(al, al0, atol=1e-6), \
        'zero-initialised t embedding must leave the base network unchanged'

    # 3. isolate t: perturb the heads so signals reach the outputs, then compare
    #    the same input against a copy whose t embedding is zeroed. The bond head
    #    and the edge subnet head are zero-initialised by design (blogits == base
    #    at init), so they have to be perturbed for this to test anything.
    import copy
    with torch.no_grad():
        for mod in (net.temb[-1], net.bhead, net.edge.head):
            mod.weight.normal_(0, .3)
            mod.bias.normal_(0, .3)
    off = copy.deepcopy(net)
    with torch.no_grad():
        off.temb[-1].weight.zero_()
        off.temb[-1].bias.zero_()
    half2 = allB.clone()
    half2[::2] = b['labels'][::2]
    with torch.no_grad():
        bl_on, al_on = net(b, half2, allA)
        bl_off, al_off = off(b, half2, allA)
    assert not torch.allclose(bl_on, bl_off, atol=1e-5), 'bond head ignores t'
    assert not torch.allclose(al_on, al_off, atol=1e-5), 'atom head ignores t'

    # 4. schedule sanity
    assert abs(cosine_gamma(0)-1) < 1e-9 and abs(cosine_gamma(1)) < 1e-9
    fr = [cosine_gamma(k/8) for k in range(9)]
    assert all(x >= y-1e-12 for x, y in zip(fr, fr[1:])), 'schedule must be monotone'
    print(f'ok: JointNetT parameters {sum(x.numel() for x in net.parameters()):,} '
          f'(base {sum(x.numel() for x in base.parameters()):,}); '
          f't is consumed, zero-init is a no-op, schedule monotone')


if __name__ == '__main__':
    tests()
