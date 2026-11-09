"""Designed chemical positive/negative controls of actual D and graph scorer.

No frozen benchmark is rescored. Class 5 has no unique q/H-complete molecular
interpretation; its completions are used only as an existential diagnostic.
"""
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys

import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/charged_benchmark_20261102'))
from study import save, sha, SOURCES
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
import pose_delocalize as D

SCORER = ROOT/'archive/legacy_two_stage_20261003/scripts/benchmark_templatefree_motifs.py'
spec = importlib.util.spec_from_file_location('control_terminal_scorer', SCORER)
scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)


def data(smiles):
    mol = Chem.MolFromSmiles(smiles); assert mol is not None
    return mol, {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                 4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}, [
        (a.GetFormalCharge(), a.GetTotalNumHs()) for a in mol.GetAtoms()]


def fixture(name, ref_smi, pred_smi=None, mutate=None, expected_equivalent=False, purpose=''):
    ref, truth, states = data(ref_smi)
    if pred_smi:
        pred_mol, pred, states = data(pred_smi)
        assert [a.GetAtomicNum() for a in ref.GetAtoms()] == [a.GetAtomicNum() for a in pred_mol.GetAtoms()]
        assert set(pred) == set(truth), 'Control atom mapping/connectivity must be explicit'
    else: pred = dict(truth)
    if mutate:
        pred, states = mutate(ref, pred, list(states))
    return {'name':name, 'ref':ref, 'truth':truth, 'pred':pred, 'states':states,
            'expected_equivalent':expected_equivalent, 'purpose':purpose}


def terminal_groups(mol, pred):
    adj = {i:set() for i in range(mol.GetNumAtoms())}
    for i,j in pred:
        adj[i].add(j); adj[j].add(i)
    for center, ns in adj.items():
        if mol.GetAtomWithIdx(center).GetAtomicNum() not in (6,7,15,16): continue
        term = sorted(i for i in ns if mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and len(adj[i]) == 1)
        if len(term) >= 2: yield center, term


def swap_resonance(mol, pred, states):
    center, os = next(terminal_groups(mol,pred)); assert len(os) == 2
    a,b = [tuple(sorted((center,o))) for o in os]
    pred[a],pred[b] = pred[b],pred[a]
    states[os[0]],states[os[1]] = states[os[1]],states[os[0]]
    return pred,states


def swap_bonds_only(mol, pred, states):
    center, os = next(terminal_groups(mol,pred)); assert len(os) == 2
    a,b = [tuple(sorted((center,o))) for o in os]
    pred[a],pred[b] = pred[b],pred[a]
    return pred,states


def all_single(mol, pred, states):
    for center, os in terminal_groups(mol,pred):
        for o in os: pred[tuple(sorted((center,o)))] = 1
    return pred,states


def rebuild(ref, bonds, states):
    rw = Chem.RWMol()
    for atom, (q,h) in zip(ref.GetAtoms(),states):
        a = Chem.Atom(atom.GetAtomicNum()); a.SetFormalCharge(q)
        a.SetNumExplicitHs(h); a.SetNoImplicit(True); rw.AddAtom(a)
    kinds = {1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
    for (i,j),c in bonds.items():
        if c not in kinds: return None
        rw.AddBond(i,j,kinds[c])
        if c == 4:
            rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        mol = rw.GetMol(); Chem.SanitizeMol(mol)
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()): return None
        if len(Chem.GetMolFrags(mol)) != 1: return None
        return mol
    except Exception: return None


def signature(mol):
    return (tuple(a.GetFormalCharge() for a in mol.GetAtoms()),
            tuple(sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),
                          4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())) for b in mol.GetBonds())))


def chemical_equivalent(ref, bonds, states):
    # Resonance moves electrons, not protons. Atom mapping is fixed in controls.
    if [h for _,h in states] != [a.GetTotalNumHs() for a in ref.GetAtoms()]: return False
    if sum(q for q,_ in states) != Chem.GetFormalCharge(ref): return False
    pred = rebuild(ref,bonds,states)
    if pred is None: return False
    forms = {signature(m) for m in Chem.ResonanceMolSupplier(ref)}
    forms.add(signature(ref))
    return signature(pred) in forms


def completions(ref, pred, states):
    edges = sorted(e for e,c in pred.items() if c == 5)
    matching = []; valid = 0
    for values in itertools.product((1,2), repeat=len(edges)):
        bonds = dict(pred); bonds.update(zip(edges,values))
        mol = rebuild(ref,bonds,states)
        valid += mol is not None
        if chemical_equivalent(ref,bonds,states): matching.append(list(values))
    return {'n_binary_completions':2**len(edges), 'n_constructible_completions':valid,
            'n_reference_resonance_equivalent':len(matching), 'matching_assignments':matching}


def main():
    RDLogger.DisableLog('rdApp.*'); torch.set_num_threads(2)
    paths = [Path(__file__), HERE/'README.md', Path(D.__file__), SCORER, SOURCES]
    plan = {'scope':'designed chemical unit controls; post-outcome; no cohort prevalence estimate',
            'actual_D':str(Path(D.__file__).relative_to(ROOT)), 'actual_scorer':str(SCORER.relative_to(ROOT)),
            'table':'actual training source modal multisets, min_support5',
            'state_equivalence':'fixed atom H, fixed total Q, RDKit resonance-form membership with q/B retained',
            'class5':'existential single/double completions holding predicted q/H fixed, not validated state recovery',
            'old_results_modified':False, 'sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    p = HERE/'D_CONTROL_PLAN.json'
    if p.exists(): assert json.loads(p.read_text())['spec'] == plan, 'Plan drift'
    else: save(p, {'recorded_utc':datetime.now(timezone.utc).isoformat(),'spec':plan})
    cases = [
        fixture('acetate_identity','CC(=O)[O-]',expected_equivalent=True,purpose='unchanged valid state'),
        fixture('acetate_resonance_swap','CC(=O)[O-]',mutate=swap_resonance,expected_equivalent=True,purpose='electron redistribution; no H movement'),
        fixture('nitro_resonance_swap','C[N+](=O)[O-]',mutate=swap_resonance,expected_equivalent=True,purpose='nitro electron redistribution positive'),
        fixture('acid_missing_double_reference_states','CC(=O)O',mutate=all_single,purpose='invalid B but correct q/H; D may contain a valid completion'),
        fixture('gem_diol_against_acid','CC(=O)O','CC(O)O',purpose='valid different species, same Q0 and same heavy adjacency; H2 difference'),
        fixture('glycine_gem_diol_against_acid','NCC(=O)O','NCC(O)O',purpose='second valid same-Q negative with a D trigger'),
        fixture('glycine_proton_transfer','NCC(=O)O','[NH3+]CC(=O)[O-]',purpose='same Q0, same heavy B, different q/H; not resonance'),
        fixture('acid_vs_acetate','CC(=O)O','CC(=O)[O-]',purpose='different net Q and H; graph-only endpoints intentionally cannot distinguish'),
        fixture('acid_bond_swap_without_H','CC(=O)O',mutate=swap_bonds_only,purpose='bond-only swap leaves OH on the wrong oxygen; invalid state'),
        fixture('acid_proton_site_swap','CC(=O)O',mutate=swap_resonance,purpose='same molecular SMILES by O symmetry, but not fixed-atom proton-preserving resonance'),
        fixture('phosphate_wrong_multiset','COP(=O)(O)O',mutate=all_single,purpose='P3 all class5 is rejected by frozen scorer'),
        fixture('sulfate_wrong_multiset','COS(=O)(=O)O',mutate=all_single,purpose='S3 all class5 is rejected by frozen scorer'),
        fixture('sulfone_wrong_multiset','CS(=O)(=O)C',mutate=all_single,purpose='S2 reference {2,2}, class5 rejected'),
        fixture('benzene_identity','c1ccccc1',expected_equivalent=True,purpose='D must not touch aromatic ring'),
    ]
    incoming = []; before = {}
    for case in cases:
        name = case['name']; ref = case['ref']
        assert chemical_equivalent(ref,case['pred'],case['states']) == case['expected_equivalent'], name
        before[name] = case
        for (i,j),c in case['pred'].items():
            incoming.append({'name':name,'i':i,'j':j,'z_i':ref.GetAtomWithIdx(i).GetAtomicNum(),
                             'z_j':ref.GetAtomWithIdx(j).GetAtomicNum(),'true':case['truth'][(i,j)],'pred':c})
    input_path = HERE/'D_fixture_input.jsonl'; output_path = HERE/'D_fixture_output.jsonl'
    input_path.write_text('\n'.join(json.dumps(r) for r in incoming)+'\n',encoding='utf-8')
    argv = sys.argv[:]
    try:
        sys.argv = [str(D.__file__),'--per_edge',str(input_path),'--sources',str(SOURCES),
                    '--output',str(output_path),'--report',str(HERE/'D_ACTUAL_IMPLEMENTATION_REPORT.json')]
        with contextlib.redirect_stdout(io.StringIO()): D.main()
    finally: sys.argv = argv
    outputs = {c['name']:{} for c in cases}
    for line in output_path.read_text().splitlines():
        r = json.loads(line); outputs[r['name']][(r['i'],r['j'])] = r['pred']
    results = []
    for name,case in before.items():
        ref,truth,pred,states = (case[k] for k in ('ref','truth','pred','states'))
        after = outputs[name]
        graph_c = scorer.resonance_align_terminal_oxy(pred,truth,ref) == truth
        graph_d = scorer.resonance_align_terminal_oxy(after,truth,ref) == truth
        built = rebuild(ref,pred,states)
        completion = completions(ref,after,states)
        results.append({'name':name,'purpose':case['purpose'],'Q_reference':Chem.GetFormalCharge(ref),
                        'Q_predicted':sum(q for q,_ in states),'reference_smiles':Chem.MolToSmiles(ref),
                        'predicted_smiles_C':Chem.MolToSmiles(built) if built else None,
                        'C_constructible':built is not None,'C_native':bool(built and Chem.MolToSmiles(built)==Chem.MolToSmiles(ref)),
                        'C_state_aware_resonance_equivalent':case['expected_equivalent'],
                        'D_changed':after != pred,'C_graph_accepted':graph_c,'D_graph_accepted':graph_d,
                        'D_state_compatible_completion_exists':completion['n_reference_resonance_equivalent'] > 0,
                        'completion_details':completion,'predicted_q_h':states,
                        'B_C':[[*e,c] for e,c in pred.items()],'B_D':[[*e,c] for e,c in after.items()]})
    named = {r['name']:r for r in results}
    for name in ('gem_diol_against_acid','glycine_gem_diol_against_acid'):
        r = named[name]
        assert r['C_constructible'] and not r['C_graph_accepted'] and r['D_changed'] and r['D_graph_accepted']
        assert not r['D_state_compatible_completion_exists'] and not r['C_native']
    for name in ('acetate_identity','acetate_resonance_swap','nitro_resonance_swap'):
        assert named[name]['D_graph_accepted'] and named[name]['D_state_compatible_completion_exists']
    for name in ('phosphate_wrong_multiset','sulfate_wrong_multiset','sulfone_wrong_multiset'):
        assert named[name]['D_changed'] and not named[name]['D_graph_accepted']
    for rel,want in plan['sha256'].items(): assert sha(ROOT/rel) == want,rel
    summary = {'scope':plan['scope'],'plan_sha256':sha(p),'n_controls':len(results),'all_assertions_passed':True,
               'actual_implementation_used':True,'results':results,
               'conclusion':'D can convert valid different-species same-Q errors into graph-accepted outputs. Graph acceptance is not state-aware chemical equivalence; no frozen benchmark prevalence or rescoring is inferred.'}
    save(HERE/'D_CONTROL_SUMMARY.json',summary)
    lines = ['# State-aware chemical controls of D', '',
             'Designed examples using the actual D rewrite and actual frozen terminal-O scorer. No locked benchmark rescoring.', '',
             '| Control | C buildable | C graph accepted | D changed | D graph accepted | State-compatible completion exists |',
             '|---|---|---|---|---|---|']
    for r in results:
        lines.append('| '+r['name']+' | '+' | '.join(str(r[k]) for k in ('C_constructible','C_graph_accepted','D_changed','D_graph_accepted','D_state_compatible_completion_exists'))+' |')
    lines += ['', summary['conclusion'],
              'A compatible completion is only existential: class5 does not specify which B/q/H molecule is intended.',
              'The glycine neutral/zwitterion case also shows that even an unchanged correct graph cannot validate atom states.',
              'Fixed-atom resonance equivalence preserves proton locations; native molecular identity additionally quotients atom symmetry. The acid proton-site-swap illustrates that difference.']
    (HERE/'D_CONTROL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__': main()
