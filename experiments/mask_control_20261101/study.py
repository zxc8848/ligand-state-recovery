"""Supplementary matched training control; never writes original paper artifacts."""
import argparse
import collections
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'experiments/prob_mask_20261002'
sys.path.insert(0, str(OLD))
import joint as J
import probe as P
import evaluation_common as E
from cluster_bootstrap import identity_key, mcnemar
from pose_localmap import write_dumps

FEATURES = OLD/'joint_v1/features.pt'
SOURCES = OLD/'joint_v1/sources.pt'
ENCODER = ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'
CONDITIONS = {
    'pose_clean': ('pdbbind_sigma0p00', 488),
    'pose_noise010': ('pdbbind_sigma0p10', 488),
    'ccd_ideal': ('ccd_sigma0p00', 1000),
}
SEEDS = [20261101, 20261102, 20261103]
MODES = ['ordinary_marginal', 'masked_marginal', 'masked_conditional']


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def refpath(condition):
    return ROOT/'work/paper2_noise_ccd_probe'/CONDITIONS[condition][0]/'reference_for_scoring.sdf'


def log(message):
    line = time.strftime('%H:%M:%S')+' '+str(message)
    print(line, flush=True)
    with (HERE/'run.log').open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def status(**kw):
    save(HERE/'STATUS.json', {'updated_local': time.strftime('%Y-%m-%d %H:%M:%S'), **kw})


def freeze():
    inputs = [FEATURES, SOURCES, ENCODER, HERE/'study.py', HERE/'README.md']
    inputs += [OLD/x for x in ('joint.py','probe.py','evaluation_common.py','pose_localmap.py')]
    inputs += [refpath(c) for c in CONDITIONS]
    inputs += [OLD/'joint_pose/geom_overlap_names.json', ROOT/'data/ccd_control_identity_hits.json']
    fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol = {
        'schema': 'matched-mask-control.v1', 'seeds': SEEDS, 'epochs': 18,
        'batch': 32, 'lr': 2e-4, 'weight_decay': 1e-4, 'clip': 2.,
        'initialization': 'paired random from scratch; no mask-pretrained warm start',
        'selection': 'minimum all-placeholder validation bond CE + joint atom CE',
        'conditions': {k:list(v) for k,v in CONDITIONS.items()},
        'training_sources': {'geom_core': 3000, 'diverse_charge': 1500},
        'training_geometries': 13500, 'validation_geometries': 1650,
        'noise_levels_in_cache': [0., .1, .2], 'modes': MODES,
        'primary_contrast': 'ordinary_marginal -> masked_marginal selected candidate',
        'secondary_contrast': 'masked_marginal -> masked_conditional selected candidate',
        'candidate_pool': 'model-specific, identical algorithm/budget; not shared graphs',
        'decoder': {'fraction': .10, 'cap': 8, 'topk': 2, 'beam': 48,
                    'atom_solver': 'released sum-product greedy, not exact MAP', 'prior_weight': 1.},
        'graph_delocalized_output_postprocessing': False,
        'statistics': 'new descriptive 72-test Holm family; no new confirmatory/untouched claim',
        'identity_sensitivity': 'exclude direct head-training matches and archived corpus-level overlap names; descriptive only',
        'timing': {'condition':'pose_clean','first_n':50,'repeats':3,'warmup':5,
                   'includes':'candidate graph + encoder + completion + catalogue/Q solve + build/rank',
                   'excludes':'model loading, catalogue construction, reference scoring'},
        'source_sha256': fingerprint,
    }
    path = HERE/'PROTOCOL.json'
    if path.exists():
        if read(path) != protocol:
            raise ValueError('Protocol or frozen input bytes changed; do not overwrite this study')
    else:
        save(path, protocol)
    return sha(path)


def classifier_logits(net, batch):
    # Values of labels/atarget are not read, only their shapes.
    return net(batch, torch.full_like(batch['labels'], J.BMASK),
               torch.full_like(batch['atarget'], J.AMASK))


def train_one(data, seed, arm, out, device, epochs=18, batch=32, protocol_hash='SMOKE'):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        done = read(out/'COMPLETE.json')
        if done['protocol_sha256'] != protocol_hash:
            raise ValueError('Completed training belongs to another protocol')
        if done.get('best_checkpoint_sha256',sha(out/'best.pt')) != sha(out/'best.pt'):
            raise ValueError('Completed checkpoint bytes changed')
        log(f'skip complete training {arm} seed={seed}')
        return
    if (out/'last.pt').exists():
        raise ValueError(f'Partial training at {out}; explicit recovery needed, no silent restart')
    P.seed_all(seed)
    net = J.JointNet(96, 2).to(device)
    initial = state_sha(net.state_dict())
    init_file = out.parent/'INITIAL_STATE.json'
    if init_file.exists():
        if read(init_file)['state_sha256'] != initial:
            raise AssertionError('Paired initial states differ')
    else:
        save(init_file, {'seed': seed, 'state_sha256': initial, 'warm_start': None})
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed)
    order_hash = hashlib.sha256()
    history = []; best = float('inf'); best_epoch = None
    supervision = {'bond_entries':0,'atom_entries':0,'optimizer_steps':0}
    start = time.perf_counter()
    log(f'TRAIN START {arm} seed={seed}: {len(data["train"])} train geometries, '
        f'{sum(p.numel() for p in net.parameters())} parameters, {epochs} epochs, no warm start')
    for epoch in range(1, epochs+1):
        net.train(); order = order_rng.permutation(len(data['train']))
        order_hash.update(order.tobytes())
        losses = []
        for offset in range(0, len(order), batch):
            b = J.pack([data['train'][int(j)] for j in order[offset:offset+batch]], device)
            if arm == 'masked':
                bs, ats, bsuper, asuper = J.corrupt_states(b)
            else:
                bs = torch.full_like(b['labels'], J.BMASK)
                ats = torch.full_like(b['atarget'], J.AMASK)
                bsuper = torch.ones_like(bs, dtype=torch.bool)
                asuper = torch.ones_like(ats, dtype=torch.bool)
            bl, al = net(b, bs, ats)
            supervision['bond_entries'] += int(bsuper.sum())
            supervision['atom_entries'] += int(asuper.sum())
            supervision['optimizer_steps'] += 1
            loss = (nn.functional.cross_entropy(bl[bsuper], b['labels'][bsuper])
                    + nn.functional.cross_entropy(al[asuper], b['atarget'][asuper]))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            losses.append(loss.item())
        net.eval(); vb, va = J.val_ce(net, data['val'], device, batch)
        P.sync()
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_bond_CE': vb,
               'val_atom_CE': va, 'seconds': time.perf_counter()-start}
        history.append(row)
        payload = {'state': net.state_dict(), 'epoch': epoch, 'arm': arm, 'seed': seed,
                   'protocol_sha256': protocol_hash, 'initial_state_sha256': initial,
                   'data_order_sha256': order_hash.hexdigest()}
        torch.save(payload, out/'last.pt')
        if vb+va < best:
            best = vb+va; best_epoch = epoch; torch.save(payload, out/'best.pt')
        save(out/'training.json', {'arm': arm, 'seed': seed, 'history': history,
             'parameters': sum(p.numel() for p in net.parameters()), 'selected_epoch': best_epoch,
             'selection': 'all-placeholder validation bond CE + atom CE',
             'initial_state_sha256': initial, 'data_order_sha256': order_hash.hexdigest(),
             'protocol_sha256': protocol_hash, 'warm_start': None,
             'cumulative_supervision':supervision})
        log(f'{arm} seed={seed} epoch={epoch}/{epochs} val={vb+va:.5f} elapsed={row["seconds"]:.1f}s')
        if protocol_hash != 'SMOKE':
            status(stage='training', seed=seed, arm=arm, epoch=epoch, epochs=epochs)
    save(out/'COMPLETE.json', {'protocol_sha256': protocol_hash, 'selected_epoch': best_epoch,
         'initial_state_sha256': initial, 'data_order_sha256': order_hash.hexdigest(),
         'best_checkpoint_sha256':sha(out/'best.pt')})
    del net, optimizer
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def prepare_eval(out, device, limit=0):
    out.mkdir(parents=True, exist_ok=True)
    model, args = P.load_backbone(ENCODER, device)
    for condition, (_, expected) in CONDITIONS.items():
        path = out/(condition+'.pt')
        if path.exists():
            blob=torch.load(path,map_location='cpu',weights_only=False)
            if blob['input_reference_sha256']!=sha(refpath(condition)) or blob['encoder_sha256']!=sha(ENCODER):
                raise ValueError('Evaluation feature-cache source changed')
            continue
        refs, excluded = E.reference_population(refpath(condition))
        if len(refs) != expected: raise ValueError('Development population changed')
        refs_list = list(refs.items())[:limit or None]
        rows = [J.make_row(mol, name, -1, 'supplementary_development') for name, mol in refs_list]
        geos = [P.geometry(r, 0., np.random.default_rng(11), args['cutoff']) for r in rows]
        feats = []
        for i in range(0, len(geos), 16):
            feats.extend(J.extract(model, geos[i:i+16], device))
        scoring_sdf=refpath(condition)
        if limit:
            scoring_sdf=out/(condition+'_smoke_reference.sdf')
            writer=Chem.SDWriter(str(scoring_sdf))
            for name,mol in refs_list: writer.write(mol)
            writer.close()
        torch.save({'rows': feats, 'reference_sdf': str(scoring_sdf),
                    'input_reference_sha256':sha(refpath(condition)),
                    'reference_sha256': sha(scoring_sdf), 'n': len(feats),
                    'excluded_by_element': excluded, 'encoder_sha256': sha(ENCODER)}, path)
        log(f'prepared {condition}: {len(feats)} fixed-coordinate graphs')
    del model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def candidate_vectors(blog):
    base = blog.argmax(-1); m = len(base)
    if not m: raise ValueError('Empty candidate-pair graph')
    confidence = blog.gather(1, base[:, None]).squeeze(1)
    k = min(8, max(1, round(.10*m)))
    flagged = torch.topk(-confidence, k).indices.tolist()
    options = [torch.topk(blog[e], 2).indices.tolist() for e in flagged]
    choices = list(itertools.product(*options))
    choices.sort(key=lambda c: -sum(float(blog[e, v]) for e, v in zip(flagged, c)))
    candidates = []
    for choice in choices[:48]:
        c = base.clone()
        for e, v in zip(flagged, choice): c[e] = v
        candidates.append(c)
    if not torch.equal(candidates[0], base): raise AssertionError('Argmax missing from beam')
    return candidates, k


@torch.no_grad()
def infer(net, row, catalogue, device, conditional=False, search=True, fixed_pool=None):
    b = J.pack([row], device)
    bl, marginal_al = classifier_logits(net, b)
    blog = bl.log_softmax(-1)
    if not search:
        candidates=[blog.argmax(-1)]; k=0
    elif fixed_pool is None:
        candidates,k=candidate_vectors(blog)
    else:
        if row['pairs'].tolist()!=fixed_pool['pairs']:
            raise AssertionError('Secondary readout candidate pairs changed')
        blog=torch.tensor(fixed_pool['blog'],dtype=blog.dtype,device=device)
        candidates=[torch.tensor(c,dtype=torch.long,device=device) for c in fixed_pool['candidates']]
        k=fixed_pool['flagged']
    n = len(candidates)
    br = J.pack([row]*n, device)
    bs = torch.cat(candidates); amask = torch.full_like(br['atarget'], J.AMASK)
    if conditional:
        _, al = net(br, bs, amask)
    else:
        # Never feed the ordinary classifier an observed-state/OOD input.
        al = marginal_al.repeat(n, 1)
    al = al + J.catalogue_logits(br, bs, catalogue)
    alog = al.log_softmax(-1)
    atoms = J.constrained_atoms(al, amask, br['atom_lengths'], br['Q'],
                                np.random.default_rng(11), greedy=True)
    atom_parts = atoms.cpu().split(br['atom_lengths'])
    alog_parts = alog.cpu().split(br['atom_lengths'])
    details = []
    for c, at, a_log in zip(candidates, atom_parts, alog_parts):
        c = c.cpu()
        smi, good = J.build_species(row, c, at)
        bscore = float(blog.gather(1, c.to(device)[:, None]).sum())
        ascore = float(a_log.gather(1, at[:, None]).sum())
        if sum((int(x)//5-2) for x in at) != row['Q']:
            raise AssertionError('Charge conservation violated')
        details.append({'bond_labels': c.tolist(), 'atom_states': at.tolist(),
                        'smiles': smi, 'good': bool(good), 'bscore': bscore,
                        'ascore': ascore, 'score': bscore+ascore})
    picked = max(range(n), key=lambda i: (details[i]['good'], details[i]['score']))
    return {'pairs': row['pairs'].tolist(), 'flagged': k, 'candidates': n,
            'bond_log_probs':blog.cpu().tolist(),
            'selected_index': picked, 'candidates_detail': details}


def graph_errors(row, pred):
    truth = row['truth']; inferred = {tuple(e): c for e, c in zip(row['pairs'].tolist(), pred) if c}
    wrong_ref = {e for e, c in truth.items() if inferred.get(e, 0) != c}
    fp = set(inferred)-set(truth)
    return {'wrong_reference_edges': len(wrong_ref),
            'missing_bonds': sum(e not in inferred for e in truth),
            'wrong_nonzero_type': sum(e in inferred for e in wrong_ref),
            'false_positive_edges': len(fp), 'full_graph_errors': len(wrong_ref)+len(fp),
            'strict': inferred == truth}


def add_targets(prediction, row):
    # This is deliberately downstream of infer()/selection.
    for c in prediction['candidates_detail']:
        c['native_correct'] = bool(c['good'] and c['smiles'] == row['smiles'])
        c['errors'] = graph_errors(row, c['bond_labels'])
        states=np.asarray(c['atom_states']); target=row['atom_target'].numpy()
        c['errors'].update({'atom_count':len(states),'q_hits':int((states//5==target//5).sum()),
                            'h_hits':int((states%5==target%5).sum()),'joint_atom_hits':int((states==target).sum())})
    chosen = prediction['candidates_detail'][prediction['selected_index']]
    base = prediction['candidates_detail'][0]
    return dict(prediction, name=row['id'], z=row['z'].tolist(), Q=row['Q'],
                ref_edges=[[i,j,int(row['z'][i]),int(row['z'][j]),c] for (i,j),c in sorted(row['truth'].items())],
                selected_labels=chosen['bond_labels'], baseline_labels=base['bond_labels'])


def evaluate_one(model_path, mode, cache, out, catalogue, device):
    result_path = out/'evaluation.json'
    if result_path.exists():
        previous = read(result_path)
        if (previous['model_sha256'] != sha(model_path) or previous['mode'] != mode
                or previous['feature_cache_sha256']!=sha(cache)):
            raise ValueError('Evaluation artifact belongs to another model')
        return
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    net = J.JointNet(96, 2).to(device); net.load_state_dict(ck['state'], strict=True); net.eval()
    blob=torch.load(cache, map_location='cpu', weights_only=False); rows=blob['rows']
    if sha(blob['reference_sdf'])!=blob['reference_sha256']: raise ValueError('Scoring reference changed')
    pools={}
    if mode=='masked_conditional':
        sibling=out.parent.parent/'masked_marginal'/cache.stem/'predictions.json'
        # Smoke has no condition subdirectory; formal runs always do.
        if not sibling.exists(): sibling=out.parent/'masked_marginal'/'predictions.json'
        for r in read(sibling)['records']:
            pools[r['name']]={'pairs':r['pairs'],'blog':r['bond_log_probs'],
                              'candidates':[c['bond_labels'] for c in r['candidates_detail']],
                              'flagged':r['flagged']}
    records = []
    for i, row in enumerate(rows):
        fixed_pool=pools[row['id']] if pools else None
        records.append(add_targets(infer(net,row,catalogue,device,mode.endswith('_conditional'),
                                         fixed_pool=fixed_pool),row))
        if (i+1) % 100 == 0: log(f'{mode} {cache.stem}: {i+1}/{len(rows)}')
    base = {r['name']: [{'name': r['name'],'i': i,'j':j,'z_i':zi,'z_j':zj,'true':c}
                        for i,j,zi,zj,c in r['ref_edges']] for r in records}
    write_dumps(records, base, 'matched', out, 0., log=log,
                selectors=[('selected_labels','selected'),('baseline_labels','argmax')])
    verdicts = {}
    for selector in ('selected','argmax'):
        old_path = list(sys.path)
        try:
            v = E.score_dump(out/f'pose_{selector}_matched_fullgraph.jsonl',
                             blob['reference_sdf'], expected_n=blob['n'])
        finally:
            sys.path[:] = old_path
        native = {r['name']: r['candidates_detail'][r['selected_index'] if selector=='selected' else 0]
                  for r in records}
        for r in v['per_molecule']:
            r['native'] = native[r['name']]['native_correct']
            r['native_constructible'] = native[r['name']]['good']
            r.update(native[r['name']]['errors'])
        v['native'] = statistics.mean(r['native'] for r in v['per_molecule'])
        v['native_constructible'] = statistics.mean(r['native_constructible'] for r in v['per_molecule'])
        v['typed_true_edge_recall']=sum(r['true_edge_hits'] for r in v['per_molecule'])/sum(r['true_edge_count'] for r in v['per_molecule'])
        verdicts[selector] = v
        save(out/f'verdicts_{selector}.json', v)
    save(out/'predictions.json', {'mode':mode,'records':records})
    save(result_path, {'mode':mode, 'model_sha256':sha(model_path),
         'protocol_sha256':ck['protocol_sha256'], 'condition':cache.stem,'n':len(rows),
         'feature_cache_sha256':sha(cache),'reference_sha256':blob['reference_sha256'],
         'selected_epoch':ck['epoch'],
         'summary':{s:{k:v[k] for k in ('native','native_constructible','strict','unified_resonance','smiles_species','typed_true_edge_recall')}
                    for s,v in verdicts.items()}})
    log(f'{mode} {cache.stem}: {read(result_path)["summary"]}')
    del net
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def paired_interval(diffs, clusters, seed=11):
    rng = np.random.default_rng(seed); diffs = np.asarray(diffs, dtype=float)
    ids = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    molecule = np.quantile(diffs[ids].mean(1), [.025,.975]).tolist()
    sums = np.array([diffs[g].sum() for g in clusters]); sizes = np.array([len(g) for g in clusters])
    ids = rng.integers(0, len(clusters), size=(2000,len(clusters)))
    identity = np.quantile(sums[ids].sum(1)/sizes[ids].sum(1), [.025,.975]).tolist()
    return molecule, identity


def summarize():
    endpoints = ['native','strict','unified_resonance','smiles_species']
    report = {'scope':'supplementary development; not a new locked confirmation', 'cells':{}, 'contrasts':[]}
    tests = []; identity_audit=read(HERE/'IDENTITY_AUDIT.json')
    for condition in CONDITIONS:
        refs,_ = E.reference_population(refpath(condition)); names = sorted(refs)
        groups = collections.defaultdict(list)
        for i,name in enumerate(names): groups[identity_key(refs[name])].append(i)
        clusters = list(groups.values())
        flags = {}
        for mode in MODES:
            values = collections.defaultdict(list)
            for seed in SEEDS:
                v = read(HERE/'evaluations'/f'seed{seed}'/mode/condition/'verdicts_selected.json')
                indexed = {r['name']:r for r in v['per_molecule']}
                flags[(seed,mode)] = indexed
                for endpoint in endpoints+['native_constructible','typed_true_edge_recall']:
                    values[endpoint].append(v[endpoint])
            report['cells'][f'{condition}/{mode}'] = {
                e:{'per_seed':vs,'mean':statistics.mean(vs),'sd':statistics.stdev(vs)} for e,vs in values.items()}
            excluded=set(identity_audit['conditions'][condition]['sensitivity_exclusions'])
            subset=[n for n in names if n not in excluded]
            sensitivity={}
            for endpoint in endpoints:
                vs=[statistics.mean(flags[(seed,mode)][n][endpoint] for n in subset) for seed in SEEDS]
                sensitivity[endpoint]={'per_seed':vs,'mean':statistics.mean(vs),'sd':statistics.stdev(vs)}
            report['cells'][f'{condition}/{mode}']['no_known_corpus_hit_sensitivity']={'n':len(subset),'metrics':sensitivity}
        for first, second in zip(MODES,MODES[1:]):
            for endpoint in endpoints:
                seed_diffs = []
                for seed in SEEDS:
                    pairs = [(int(flags[(seed,first)][n][endpoint]), int(flags[(seed,second)][n][endpoint])) for n in names]
                    ds = np.array([y-x for x,y in pairs]); seed_diffs.append(ds)
                    mc,cl = paired_interval(ds,clusters)
                    t = {'condition':condition,'old':first,'new':second,'endpoint':endpoint,'seed':seed,
                         'delta':float(ds.mean()),'gained':sum(y and not x for x,y in pairs),
                         'lost':sum(x and not y for x,y in pairs),'raw_p':mcnemar(pairs),
                         'molecule_ci95':mc,'identity_cluster_ci95':cl}
                    tests.append(t)
                mean_diffs = np.mean(seed_diffs,axis=0)
                mc,cl = paired_interval(mean_diffs,clusters)
                report['contrasts'].append({'condition':condition,'old':first,'new':second,'endpoint':endpoint,
                    'mean_delta':float(mean_diffs.mean()),'seed_sd':float(np.std([d.mean() for d in seed_diffs],ddof=1)),
                    'seed_averaged_molecule_ci95':mc,'seed_averaged_identity_cluster_ci95':cl})
    if len(tests) != 72: raise AssertionError('Unexpected descriptive test family')
    running = 0.
    for rank,t in enumerate(sorted(tests,key=lambda x:x['raw_p'])):
        running = max(running,min(1.,(len(tests)-rank)*t['raw_p']))
        t['holm_p_72'] = running
    report['tests'] = tests; report['holm_family_size'] = 72
    save(HERE/'SUMMARY.json',report)
    lines = ['# Matched-control supplementary results','',
             'From-scratch paired initialization; no original locked results modified.','',
             '| Condition | Mode | Native B/h/q | Strict | Resonance | Graph SMILES |',
             '|---|---|---:|---:|---:|---:|']
    for key,cell in report['cells'].items():
        condition,mode=key.split('/')
        vals=[f'{100*cell[e]["mean"]:.2f} +/- {100*cell[e]["sd"]:.2f}' for e in endpoints]
        lines.append('| '+' | '.join([condition,mode]+vals)+' |')
    lines += ['', 'All values are percent, mean +/- SD over three training seeds.',
              'Full per-seed paired intervals and the separate 72-test Holm family: SUMMARY.json.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('Wrote SUMMARY.json and RESULTS.md')


def audit_saved():
    """Read saved historical predictions only; no locked SDF or model inference."""
    result = {'scope':'post-hoc paired audit of existing locked predictions; no new inference', 'seeds':[]}
    for seed in (20261002,20261003,20261004):
        path = OLD/'locked_eval'/f'seed{seed}'/'pose_localmap_locked.json'
        records = read(path)['records']; counts = collections.Counter(); cases=[]
        for r in records:
            truth = {(i,j):c for i,j,zi,zj,c in r['ref_edges']}
            predicted = []
            for key in ('baseline_labels','localmap_labels'):
                pred = {tuple(e):int(c) for e,c in zip(r['pairs'],r[key]) if c}
                errors = {e for e in set(truth)|set(pred) if truth.get(e,0)!=pred.get(e,0)}
                predicted.append((pred,errors))
            (p0,e0),(p1,e1)=predicted
            fixed=e0-e1; added=e1-e0
            group = ('rescued' if e0 and not e1 else 'damaged' if not e0 and e1 else
                     'both_correct' if not e0 and not e1 else 'both_wrong')
            counts[group]+=1; counts['old_full_errors']+=len(e0); counts['new_full_errors']+=len(e1)
            counts['fixed_edges']+=len(fixed); counts['introduced_edges']+=len(added)
            counts[group+'_fixed_edges']+=len(fixed); counts[group+'_introduced_edges']+=len(added)
            counts['old_false_positive_edges']+=len(set(p0)-set(truth))
            counts['new_false_positive_edges']+=len(set(p1)-set(truth))
            if p0!=p1: cases.append({'name':r['name'],'transition':group,'old_errors':len(e0),
                                    'new_errors':len(e1),'fixed_edges':sorted(fixed),'introduced_edges':sorted(added)})
        if counts['new_full_errors']-counts['old_full_errors'] != counts['introduced_edges']-counts['fixed_edges']:
            raise AssertionError('Error-transition accounting mismatch')
        result['seeds'].append({'seed':seed,'input_sha256':sha(path),'n':len(records),
                                'counts':dict(counts),'changed_cases':cases})
    save(HERE/'HISTORICAL_ERROR_TRANSITIONS.json',result)
    log('Saved read-only historical paired error-transition audit')


def audit_identities():
    sources=torch.load(SOURCES,map_location='cpu',weights_only=False)
    keys={Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']),isomericSmiles=False) for r in sources['train']}
    geom=read(OLD/'joint_pose/geom_overlap_names.json')
    ccd=read(ROOT/'data/ccd_control_identity_hits.json')
    result={'scope':'completion training checked directly; archived encoder/source-corpus hits are conservative, not exact split labels',
            'source_domains':dict(collections.Counter(r['domain'] for r in sources['train'])),'conditions':{}}
    for condition in CONDITIONS:
        refs,_=E.reference_population(refpath(condition))
        direct=[n for n,m in refs.items() if Chem.MolToSmiles(m,isomericSmiles=False) in keys]
        if condition.startswith('pose'):
            known=set(geom['exact_smiles'])|set(geom['skeleton'])
        else:
            known=set(ccd['plain'])|set(ccd['stereo'])
        archived=sorted(set(refs)&known)
        result['conditions'][condition]={'n':len(refs),'direct_completion_train_hits':sorted(direct),
            'archived_source_corpus_hits':archived,'sensitivity_exclusions':sorted(set(direct)|set(archived))}
    save(HERE/'IDENTITY_AUDIT.json',result)
    log('Identity audit: '+str({c:{'head_hits':len(v['direct_completion_train_hits']),
        'conservative_exclusions':len(v['sensitivity_exclusions'])} for c,v in result['conditions'].items()}))


def benchmark(device, catalogue):
    if (HERE/'TIMING.json').exists(): return
    encoder,args=P.load_backbone(ENCODER,device)
    rows=torch.load(HERE/'eval_features/pose_clean.pt',map_location='cpu',weights_only=False)['rows'][:50]
    configs=[('ordinary_argmax','ordinary',False,False),
             ('ordinary_search','ordinary',False,True),
             ('masked_marginal_search','masked',False,True),
             ('masked_conditional_search','masked',True,True)]
    timing={'device':str(device),'torch_version':torch.__version__,
            'gpu':torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None,
            'threads':torch.get_num_threads(),'n':len(rows),'repeats':3,'warmup':5,
            'includes':'candidate graph, encoder, completion, catalogue/Q solve, RDKit build and ranking',
            'excludes':'loading weights, loading inputs, constructing training catalogue, reference scoring',
            'records':{},'summary':{}}
    for name,arm,conditional,search in configs:
        net=J.JointNet().to(device)
        path=HERE/'training'/f'seed{SEEDS[0]}'/arm/'best.pt'
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);net.eval()
        observations=[]
        for rep in (-1,0,1,2):
            for original in rows[:5] if rep<0 else rows:
                row={k:v for k,v in original.items() if k not in
                     ('feat','node_feat','base','pairs','directed','labels','sigma','missed')}
                P.sync();t0=time.perf_counter()
                geom=P.geometry(row,0.,np.random.default_rng(11),args['cutoff'])
                P.sync();t1=time.perf_counter()
                feat=J.extract(encoder,[geom],device)[0]
                P.sync();t2=time.perf_counter()
                pred=infer(net,feat,catalogue,device,conditional,search)
                P.sync();t3=time.perf_counter()
                if rep>=0:
                    observations.append({'name':row['id'],'repeat':rep,'heavy_atoms':len(row['z']),
                        'graph_ms':1000*(t1-t0),'encoder_ms':1000*(t2-t1),
                        'decoder_ms':1000*(t3-t2),'total_ms':1000*(t3-t0),
                        'constructible':pred['candidates_detail'][pred['selected_index']]['good']})
        vals=np.array([r['total_ms'] for r in observations])
        timing['records'][name]=observations
        timing['summary'][name]={'mean_ms':float(vals.mean()),'median_ms':float(np.median(vals)),
            'p95_ms':float(np.quantile(vals,.95)), 'failed_predictions_including_repeats':
            sum(not r['constructible'] for r in observations),'timed_predictions':len(observations),
            'model_sha256':sha(path)}
        log(f'timing {name}: {timing["summary"][name]}')
        del net
    save(HERE/'TIMING.json',timing)
    del encoder
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def selftest():
    P.seed_all(7)
    row = {'z':torch.tensor([6,8]),'pairs':torch.tensor([[0,1]]), 'feat':torch.randn(1,360),
           'node_feat':torch.randn(2,360),'base':torch.log(torch.tensor([[.1,.3,.4,.1,.1]])),
           'labels':torch.tensor([2]),'atom_target':torch.tensor([12,10]),'Q':0}
    b=J.pack([row],'cpu'); net=J.JointNet(96,2).eval()
    with torch.no_grad():
        first=classifier_logits(net,b)
        fake=dict(b,labels=torch.tensor([4]),atarget=torch.tensor([24,0]))
        second=classifier_logits(net,fake)
    assert all(torch.equal(a,c) for a,c in zip(first,second)), 'Target leakage'
    assert sum(x.numel() for x in net.parameters())==388499
    other=J.JointNet(96,2); other.load_state_dict(net.state_dict())
    assert state_sha(other.state_dict())==state_sha(net.state_dict())
    candidates,k=candidate_vectors(first[0].log_softmax(-1))
    assert k==1 and len(candidates)==2
    # Catalogue plus charge-constrained solver and constructibility path.
    source=dict(row,truth={(0,1):2})
    catalogue=J.make_catalogue([source])
    original=infer(net,row,catalogue,'cpu')
    assert J.build_species(row,row['labels'],row['atom_target'])[1]
    assert original['candidates_detail'][original['selected_index']]['good']
    blinded=dict(row,labels=torch.tensor([4]),atom_target=torch.tensor([24,0]))
    changed=infer(net,blinded,catalogue,'cpu')
    assert original==changed, 'Decoded predictions depend on held-out targets'
    pool={'pairs':original['pairs'],'blog':original['bond_log_probs'],
          'candidates':[c['bond_labels'] for c in original['candidates_detail']],
          'flagged':original['flagged']}
    conditional=infer(net,row,catalogue,'cpu',conditional=True,fixed_pool=pool)
    assert [c['bond_labels'] for c in original['candidates_detail']]==[c['bond_labels'] for c in conditional['candidates_detail']]
    assert mcnemar([(0,1)]*5)==.0625
    log('SELFTEST PASS: equal capacity/init, target blinding, candidate pool, charge/build path')


def run(device, smoke=False):
    selftest()
    if smoke:
        out=HERE/'smoke_final'
        data=torch.load(FEATURES,map_location='cpu',weights_only=False)['rows']
        data={'train':data['train'][:64],'val':data['val'][:32]}
        for arm in ('ordinary','masked'):
            train_one(data,SEEDS[0],arm,out/f'seed{SEEDS[0]}'/arm,device,epochs=1)
        catalogue=J.make_catalogue(torch.load(SOURCES,map_location='cpu',weights_only=False)['train'])
        row=data['val'][0]
        for arm in ('ordinary','masked'):
            net=J.JointNet().to(device)
            net.load_state_dict(torch.load(out/f'seed{SEEDS[0]}'/arm/'best.pt',map_location=device,weights_only=False)['state'])
            net.eval(); pred=infer(net,row,catalogue,device)
            assert pred['candidates_detail']
        prepare_eval(out/'eval_features',device,limit=4)
        for mode in MODES:
            arm=mode.split('_')[0]
            evaluate_one(out/f'seed{SEEDS[0]}'/arm/'best.pt',mode,
                         out/'eval_features/pose_clean.pt',out/'evaluation'/mode,catalogue,device)
        save(out/'PASSED.json',{'scope':'debug only, not a study result','passed':True})
        log('SMOKE PASS')
        return
    protocol_hash=freeze()
    status(stage='preparing',protocol_sha256=protocol_hash)
    audit_saved()
    audit_identities()
    prepare_eval(HERE/'eval_features',device)
    data=torch.load(FEATURES,map_location='cpu',weights_only=False)['rows']
    if (len(data['train']),len(data['val']))!=(13500,1650): raise ValueError('Training inventory changed')
    for seed in SEEDS:
        for arm in ('ordinary','masked'):
            train_one(data,seed,arm,HERE/'training'/f'seed{seed}'/arm,device,protocol_hash=protocol_hash)
        left=read(HERE/'training'/f'seed{seed}'/'ordinary'/'COMPLETE.json')
        right=read(HERE/'training'/f'seed{seed}'/'masked'/'COMPLETE.json')
        if (left['initial_state_sha256'],left['data_order_sha256']) != (right['initial_state_sha256'],right['data_order_sha256']):
            raise AssertionError('Paired training did not share initialization/order')
    del data
    catalogue=J.make_catalogue(torch.load(SOURCES,map_location='cpu',weights_only=False)['train'])
    for seed in SEEDS:
        for mode in MODES:
            arm=mode.split('_')[0]
            for condition in CONDITIONS:
                status(stage='evaluating',seed=seed,mode=mode,condition=condition)
                evaluate_one(HERE/'training'/f'seed{seed}'/arm/'best.pt',mode,
                             HERE/'eval_features'/(condition+'.pt'),
                             HERE/'evaluations'/f'seed{seed}'/mode/condition,catalogue,device)
    summarize()
    status(stage='timing')
    benchmark(device,catalogue)
    status(stage='COMPLETE',protocol_sha256=protocol_hash,summary=str(HERE/'SUMMARY.json'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['selftest','smoke','run','summarize','audit_saved'])
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args(); torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    if args.action=='selftest': selftest()
    elif args.action=='audit_saved': audit_saved()
    elif args.action=='summarize': summarize()
    else:
        try: run(args.device,args.action=='smoke')
        except Exception as exc:
            if args.action=='run': status(stage='FAILED',error=repr(exc))
            raise
