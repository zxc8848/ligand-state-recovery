"""Check fixed model settings and v3 behavioral/driver contracts.

The frozen-configuration document is only worth something if it is verifiable, so
this script re-reads the checkpoints, the source files and the generated tables
and asserts the documented values. It changes nothing. On success it records the
document's own SHA256 in results/frozen_config_verification.json, which is what
the locked-set evaluation checks before it will run.

Historical ledgers retain their definitions. V3 fixes scorer/runtime bugs without
changing model weights; exact byte freezing is checked by frozen_manifest.py.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT/'results/FROZEN_CONFIG_V3_1_2026-10-05.md'
DOC_V1 = ROOT/'results/FROZEN_CONFIG_2026-10-04.md'

OK, BAD = [], []


def check(name, got, want):
    if isinstance(want, float) and isinstance(got, (int, float)):
        good = abs(got-want) < 1e-12
    else:
        good = got == want
    (OK if good else BAD).append((name, got, want))


def source_has(name, path, pattern):
    # re.M so that '^def foo' can assert a MODULE-LEVEL definition rather than a
    # nested one -- several section-4.3 checks depend on that distinction.
    hit = re.search(pattern, Path(path).read_text(encoding='utf-8'), re.M)
    (OK if hit else BAD).append(
        (name, 'found' if hit else 'NOT FOUND', pattern))


def main():
    import torch

    # --- section 2: the frozen geometric encoder -------------------------------
    bb = torch.load(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt',
                    map_location='cpu', weights_only=False)
    a = bb['args']
    check('backbone family', a['backbone'], 'painn')
    check('num_interactions', a['num_interactions'], 4)
    check('hidden_size', a['hidden_size'], 360)
    check('cutoff', a['cutoff'], 2.5)
    check('edge_embedding_size', a['edge_embedding_size'], 20)
    check('vector_norm_limit', a['vector_norm_limit'], 3.0)
    check('backbone noise_min', a['noise_min'], 0.0)
    check('backbone noise_max', a['noise_max'], 0.15)
    check('backbone noise_clean_prob', a['noise_clean_prob'], 0.0)
    check('backbone seed', a['seed'], 42)
    check('backbone lr', a['lr'], 1e-4)
    check('backbone epochs', a['epochs'], 50)
    check('backbone batch_size', a['batch_size'], 128)
    check('backbone weight_decay', a['weight_decay'], 1e-5)
    check('backbone grad_clip', a['grad_clip'], 1.0)
    check('backbone explicit_h', a['explicit_h'], False)
    check('backbone split_key', a['split_key'], 'geom_mol_idx')
    check('backbone val_split', a['val_split'], 0.1)
    check('backbone selected epoch', bb['epoch'], 45)
    check('backbone parameters',
          sum(v.numel() for v in bb['model_state_dict'].values()), 6443825)

    # --- section 3: the completion net ----------------------------------------
    jv = torch.load(HERE/'joint_v1/best.pt', map_location='cpu', weights_only=False)
    ja = jv['args']
    check('JointNet parameters', sum(v.numel() for v in jv['state'].values()), 388499)
    check('joint_v1 selected epoch', jv['epoch'], 13)
    check('joint width', ja['width'], 96)
    check('joint layers', ja['layers'], 2)
    check('joint batch', ja['batch'], 32)
    check('joint epochs', ja['epochs'], 18)
    check('joint lr', ja['lr'], 2e-4)
    check('joint seed', ja['seed'], 20261002)
    check('feature noise levels', list(ja['noises']), [0.0, 0.1, 0.2])
    check('charge_train', ja['charge_train'], 1000)
    check('internal_train', ja['internal_train'], 500)
    check('charge_val', ja['charge_val'], 150)
    check('internal_val', ja['internal_val'], 100)

    for s in (20261002, 20261003, 20261004):
        f = HERE/f'joint_v5/seed{s}/best.pt'
        check(f'seed {s} checkpoint present', f.exists(), True)
        if f.exists():
            ck = torch.load(f, map_location='cpu', weights_only=False)
            check(f'seed {s} parameters',
                  sum(v.numel() for v in ck['state'].values()), 388499)
            check(f'seed {s} recipe seed', ck['args']['seed'], s)

    source_has('MaskedBondDecoder(width=96, layers=3, context=True)', HERE/'joint.py',
               r"MaskedBondDecoder\(width=96,\s*layers=3,\s*context=True\)")
    source_has('bond head 5 classes, atom head 25', HERE/'joint.py',
               r"self\.bhead = nn\.Linear\(width, 5\); self\.ahead = nn\.Linear\(width, 25\)")
    source_has('AdamW wd 1e-4', HERE/'joint_seeds_train.py',
               r"AdamW\(net\.parameters\(\), lr=a\.lr, weight_decay=1e-4\)")
    source_has('grad clip 2.0', HERE/'joint_seeds_train.py',
               r"clip_grad_norm_\(net\.parameters\(\), 2\.\)")
    source_has('warm start from pilot_v2/decoder_best.pt', HERE/'joint_seeds_train.py',
               r"pilot_v2/decoder_best\.pt")

    # --- section 3.1: masking / corruption rates ------------------------------
    source_has('bond mask rate U(0.2,1.0)', HERE/'probe.py',
               r"rate = \.2 \+ \.8\*torch\.rand")
    source_has('bond mask full-mask prob 0.2', HERE/'probe.py',
               r"if torch\.rand\(\(\), device=mask\.device\) < \.2:")
    source_has('entropy clamp (0.05, 1.0)', HERE/'probe.py',
               r"clamp\(rate\*\(\.5\+entropy\[start:start\+n\]\), \.05, 1\.\)")
    source_has('atom-incident bond mask 0.3 / 0.15', HERE/'probe.py',
               r"< \.3:\s*\n\s*atommask = torch\.rand\(b\['nnode'\], device=mask\.device\) < \.15")
    source_has('bond corruption 0.05', HERE/'probe.py',
               r"corrupt = \(~mask\) & \(torch\.rand_like\(entropy\) < \.05\)")
    source_has('atom mask rate U(0.2,1.0), full-mask 0.3', HERE/'joint.py',
               r"rate = \.2\+\.8\*torch\.rand\(\(\), device=ats\.device\)")
    source_has('atom-star coupling 0.4', HERE/'joint.py',
               r"if torch\.rand\(\(\), device=ats\.device\) < \.4:")
    source_has('atom corruption 0.03', HERE/'joint.py',
               r"torch\.rand\(len\(ats\), device=ats\.device\) < \.03")

    # --- section 4: the decoder ------------------------------------------------
    sys.argv = ['verify']
    import pose_localmap as PL
    d = PL.parse_args()
    check('localmap fraction', d.fraction, 0.10)
    check('localmap max_flagged', d.max_flagged, 8)
    check('localmap beam', d.beam, 48)
    check('localmap topk_per_edge', d.topk_per_edge, 2)
    check('localmap batch', d.batch, 16)
    check('localmap inference sigma', d.sigma, 0.0)
    source_has('sampling seed 11', HERE/'pose_localmap.py', r"p\.seed_all\(11\)")
    source_has('rng seed 11', HERE/'pose_localmap.py',
               r"np\.random\.default_rng\(11\)")
    check('localmap atom_solver default', d.atom_solver, 'greedy')
    source_has('atom_solver choices greedy|map', HERE/'pose_localmap.py',
               r"--atom_solver', choices=\['greedy', 'map'\], default='greedy'")
    source_has('constrained_atoms greedy=True', HERE/'pose_localmap.py',
               r"constrained_atoms\(al, aR, bR\['atom_lengths'\], bR\['Q'\],\s*\n?\s*"
               r"draws, greedy=True\)")
    source_has('names_from_baseline is an explicit switch', HERE/'pose_localmap.py',
               r"--names_from_baseline', action='store_true'")
    source_has('reference_from_sdf is an explicit switch', HERE/'pose_localmap.py',
               r"--reference_from_sdf', action='store_true'")
    source_has('buildability outranks score', HERE/'pose_localmap.py',
               r"\(entry\[0\], entry\[1\]\) > \(best\[0\], best\[1\]\)")
    source_has('catalogue pseudocount 1e-4', HERE/'joint.py', r"probs = counts\+1e-4")
    source_has('charge DP float64', HERE/'joint.py',
               r"logits\.detach\(\)\.double\(\)\.cpu\(\)\.numpy\(\)")

    # --- section 4.2: the exact joint MAP exists and is NOT the default -------
    check('constrained_map.py present', (HERE/'constrained_map.py').exists(), True)
    source_has('constrained_map is max-product with argmax traceback',
               HERE/'constrained_map.py', r"def constrained_map\(")
    import constrained_map as CM
    check('constrained_map exposes constrained_map',
          callable(getattr(CM, 'constrained_map', None)), True)
    # the document claims the exact solver is reachable from the decoder
    source_has('decoder can dispatch to the exact solver', HERE/'pose_localmap.py',
               r"CM\.constrained_map\(")

    # --- section 4.3: the export interface ------------------------------------
    source_has('write_dumps is module level', HERE/'pose_localmap.py',
               r"^def write_dumps\(")
    source_has('build_reference_rows is module level', HERE/'pose_localmap.py',
               r"^def build_reference_rows\(")
    source_has('export self-test present', HERE/'pose_localmap.py', r"^def _selftest\(")
    source_has('refedges dump written', HERE/'pose_localmap.py', r"_refedges\.jsonl")
    source_has('fullgraph dump written', HERE/'pose_localmap.py', r"_fullgraph\.jsonl")
    source_has('off-reference predictions recorded, not dropped',
               HERE/'pose_localmap.py', r"'off_reference': True")
    for mod in ('noise_aggregate.py', 'noise_seeds_aggregate.py',
                'convention_split.py', 'convention_aggregate.py'):
        source_has(f'{mod} defaults to the fullgraph endpoint', HERE/mod,
                   r"--kind', default='fullgraph'")

    # --- section 7.1: the two endpoints ---------------------------------------
    check('score_two_endpoints.py present',
          (HERE/'score_two_endpoints.py').exists(), True)
    source_has('full endpoint has no paired baseline, and the script says so',
               HERE/'score_two_endpoints.py', r"IMPORTANT ASYMMETRY")

    # --- section 7.2: ONE equivalence definition for all three systems --------
    check('unified_resonance.py present', (HERE/'unified_resonance.py').exists(), True)
    source_has('unified definition: terminal-oxy align on C,N,P,S',
               HERE/'evaluation_common.py',
               r"resonance_align_terminal_oxy\(pred, truth, mol, elements=\(6, 7, 15, 16\)\)")
    source_has('unified definition: validated aromatic components',
               HERE/'evaluation_common.py', r"aligned = aromatic_component_align\(aligned, truth, mol\)")
    import unittest
    import test_eval_v3
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_eval_v3)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    check('chemical/denominator/statistics regression tests', result.wasSuccessful(), True)
    smoke_file = ROOT/'work/evaluation_fix_v3/end_to_end/PASSED.json'
    check('actual formal-entry development smoke present', smoke_file.exists(), True)
    if smoke_file.exists():
        smoke = json.loads(smoke_file.read_text(encoding='utf-8'))
        check('development-only smoke passed', smoke.get('passed') and smoke.get('development_only'), True)
        for name, h in smoke.get('source_sha256', {}).items():
            check(f'smoke tested current {name}', hashlib.sha256((HERE/name).read_bytes()).hexdigest(), h)
        check('smoke covers all pipeline modules', len(smoke.get('source_sha256', {})), 15)
    source_has('formal record carries z', HERE/'pose_ablate_search.py', r"'z': \[int\(x\) for x in r\['z'\]\.tolist\(\)\]")
    check('knodle_to_per_edge.py present',
          (HERE/'knodle_to_per_edge.py').exists(), True)
    source_has("Knodle label map {'ar':4,1:1,2:2,3:3}", HERE/'knodle_to_per_edge.py',
               r"ORDER = \{'ar': 4, 1: 1, 2: 2, 3: 3")
    check('classify_fixed.py present', (HERE/'classify_fixed.py').exists(), True)
    source_has('Kekule POSITION check, not just the double-bond count',
               HERE/'classify_fixed.py', r"def kekule_alternates\(")

    # --- section 7.3: statistics ----------------------------------------------
    check('cluster_bootstrap.py present', (HERE/'cluster_bootstrap.py').exists(), True)
    source_has('clusters are resampled whole', HERE/'cluster_bootstrap.py',
               r"def boot_clusters\(")
    source_has('molecule-level interval kept alongside', HERE/'cluster_bootstrap.py',
               r"def boot_molecules\(")
    sys.argv = ['verify']
    import cluster_bootstrap as CB
    source_has('bootstrap reps default 2000', HERE/'cluster_bootstrap.py',
               r"--reps', type=int, default=2000")
    source_has('bootstrap seed default 11', HERE/'cluster_bootstrap.py',
               r"--seed', type=int, default=11")

    # section 1: the identity structure of the development set, recomputed
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        spec = json.loads((HERE/'joint_v6/joint_v1/pdbbind_sigma0p00'
                           '/species_localmap_deloc.json').read_text(encoding='utf-8'))
        names = {r['name'] for r in spec['per_molecule']}
        sdf = ROOT/'work/paper2_noise_ccd_probe/pdbbind_sigma0p00/reference_for_scoring.sdf'
        keys = []
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
            if mol is None or not mol.HasProp('_Name'):
                continue
            if mol.GetProp('_Name') in names:
                keys.append(CB.identity_key(mol))
        import collections as _c
        groups = _c.Counter(keys)
        dup = {k: v for k, v in groups.items() if v > 1}
        check('development poses', len(keys), 488)
        check('distinct identities', len(groups), 471)
        check('duplicate identity groups', len(dup), 12)
        check('largest duplicate group', max(dup.values()) if dup else 1, 5)
        check('molecules in a duplicate group', sum(dup.values()), 29)
    except ImportError:
        BAD.append(('identity structure', 'rdkit unavailable', 'rdkit'))

    # --- section 9: the pre-registered component ladder -----------------------
    source_has('component ladder rung B exists', HERE/'ablate_aggregate.py',
               r"def pick_bond_only\(")
    source_has('rung B is in the selector list', HERE/'ablate_aggregate.py',
               r"\('first buildable by P\(B\)', pick_bond_only\)")
    source_has('rung B ranks by (good, bscore)', HERE/'ablate_aggregate.py',
               r"max\(cands, key=lambda x: \(x\['good'\], x\['bscore'\]\)\)")
    source_has('the search records the two score components separately',
               HERE/'pose_ablate_search.py', r"'bscore': blog_total\[t\], 'ascore': atom_lp")
    source_has('rung B gets its own label vector', HERE/'pose_ablate_search.py',
               r"'bondonly_labels': \[int\(x\) for x in bsplit\[bt\]\.tolist\(\)\]")
    source_has('the ladder exports through the SHARED write_dumps',
               HERE/'pose_ablate_search.py', r"PLM\.write_dumps\(records, base, tag")
    source_has('write_dumps takes the selector list as a parameter',
               HERE/'pose_localmap.py', r"def write_dumps\(records, base, tag, outdir, sigma, log=None, selectors=None\)")
    # the locked-set driver must point at THIS document and must no longer carry
    # the interface-fix block that refused to run
    drv = (ROOT/'scripts/run_locked_eval.ps1').read_text(encoding='utf-8')
    check('driver points at the v3.1 frozen config',
          'FROZEN_CONFIG_V3_1_2026-10-05.md' in drv, True)
    check('driver no longer gates on BONDNET_LOCKED_EVAL_INTERFACE_FIXED',
          'BONDNET_LOCKED_EVAL_INTERFACE_FIXED' in drv, False)
    check('driver decodes with the beam-recording entry point',
          'pose_ablate_search.py' in drv, True)
    check('driver scores both export kinds',
          "@('fullgraph','refedges')" in drv, True)
    check('driver scores all four ladder rungs',
          "@('greedy','bondonly','localmap','localmap_deloc')" in drv, True)
    check('driver scores Knodle under the unified definition',
          'knodle_to_per_edge.py' in drv and 'unified_resonance.py' in drv, True)
    check('driver uses shared paired verdicts, not legacy scorers',
          'paired_verdicts.py' in drv and 'score_knodle_vs_bondnet.py' not in drv
          and 'compare_old_new.py' not in drv, True)
    check('driver has persistent started/failed state',
          "status='STARTED'" in drv and "$State.status='FAILED'" in drv, True)

    # --- the superseded document must still say so ----------------------------
    check('v1 frozen config still present as history', DOC_V1.exists(), True)
    if DOC_V1.exists():
        head = DOC_V1.read_text(encoding='utf-8')[:400]
        check('v1 is marked superseded', '已作废' in head, True)

    # --- section 5: the delocalized table -------------------------------------
    rep = json.loads((HERE/'joint_v6/joint_v1/pdbbind_sigma0p00/pose_delocalize.json')
                     .read_text(encoding='utf-8'))
    want = {'16:2': [2, 2], '6:2': [1, 2], '15:3': [1, 1, 2], '15:2': [1, 2],
            '16:3': [1, 2, 2], '7:2': [1, 2]}
    got = {k: v['canonical'] for k, v in rep['training_canonical_multisets'].items()}
    check('delocalized canonical multisets', got, want)
    check('S,2 canonical is (2,2)', got.get('16:2'), [2, 2])
    sys.argv = ['verify']
    import pose_delocalize as PD
    check('delocalized min_support', PD.parse_args().min_support, 5)

    # --- section 1: the locked test set ---------------------------------------
    fz = json.loads((ROOT/'data/pdbbind_pose_locked500_freeze.json')
                    .read_text(encoding='utf-8'))
    sdf = ROOT/'data/pdbbind_pose_locked500.sdf'
    h = hashlib.sha256(sdf.read_bytes()).hexdigest()
    check('locked set sha256 matches freeze record', h, fz['sha256'])
    check('locked set sha256 matches this document',
          h, 'fe032b5100e4b066faa4dbbd6a1b50ae6fbf7a8be73dbaa1476cc276abe160d2')
    check('locked set size', fz['written'], 500)
    check('locked set shuffle seed', fz['shuffle_seed'], 20261004)
    from population_manifest import validate_population
    population_file = ROOT/'data/pdbbind_pose_locked500_population_v3_1.json'
    refs, excluded = validate_population(sdf, population_file)
    check('raw locked population', len(refs)+len(excluded), 500)
    check('eligible locked population under original element rule', len(refs), 491)
    check('fixed element exclusions', sorted(excluded),
          sorted(['6jn6','1nyy','3ixg','5ee8','1p01','3sjt','3e9b','4hww','3p3j']))
    # Historical metadata is not proof the set is untouched. Detect existing
    # outputs without inspecting their values; the driver blocks unsafe reuse.
    partial = list((HERE/'locked_eval').rglob('*.json')) + list((HERE/'locked_eval').rglob('*.jsonl'))
    if partial:
        print('WARNING: pre-existing locked artifacts; untouched status NOT certified')

    # --- report ---------------------------------------------------------------
    for name, got_, want_ in OK:
        print(f'  ok    {name}')
    for name, got_, want_ in BAD:
        print(f'  FAIL  {name}: got {got_!r}, document says {want_!r}')
    print(f'\n{len(OK)} checks passed, {len(BAD)} failed')
    if BAD:
        print('\nFrozen configuration or implementation contract failed; do not run locked evaluation.')
        raise SystemExit(1)

    doc_hash = hashlib.sha256(DOC.read_bytes()).hexdigest()
    out = ROOT/'results/frozen_config_verification.json'
    out.write_text(json.dumps({
        'document': str(DOC), 'document_sha256': doc_hash,
        'checks_passed': len(OK), 'checks_failed': 0,
        'locked_set_sha256': h,
        'locked_untouched_certified': not bool(partial),
        'note': ('Model settings, regression tests and driver contracts checked. '
                 'Exact byte manifest is a separate mandatory driver check. '
                 'Existing locked artifacts are not evidence of an untouched test set.')}, indent=1), encoding='utf-8')
    print(f'document sha256 {doc_hash}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
