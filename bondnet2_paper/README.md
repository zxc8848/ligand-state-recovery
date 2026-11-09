# Reproducing the current BondNet heavy-atom manuscript

This document covers the released single-encoder + state-prediction + local
candidate re-ranking reference implementation in `bondnet2_paper/main.tex`. The previous reproduction
guide is preserved outside the paper directory at
`work/paper_revision_20261005/README_historical.md`. Do not use its checkpoint
recipes or figure sources as evidence for the current method.

Run commands from `E:\zxcproject\bondnet2.0`. The examples below use PowerShell.
Python with PyTorch and RDKit is required; current figures additionally require
matplotlib. Knodle execution requires its third-party binary and Linux/WSL.
No command in this guide installs those dependencies.

## 1. Frozen model and benchmark inventory

- Encoder: `checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt`.
- Completion models: `experiments/prob_mask_20261002/joint_v5/seed20261002/best.pt`,
  and the equivalent files for seeds 20261003 and 20261004.
- Catalogue source: `experiments/prob_mask_20261002/joint_v1/sources.pt`.
- Development poses: `data/pdbbind_ccd_pose500.sdf`, 488 eligible records.
- Locked inputs: `data/pdbbind_pose_locked500.sdf`, 500 raw records.
- Locked population: `data/pdbbind_pose_locked500_population_v3_1.json`,
  491 eligible records; eight boron-containing and one ruthenium-containing record
  are excluded by the original allowed-element rule.

The encoder checkpoint is fixed. Reported three-seed SD measures completion-network
variation, not encoder training variation. The released head is mask-trained; the
new matched study below does not establish that masking is needed. Neither benchmark supplies reference bonds,
atom-wise charges or hydrogen counts to inference; molecular total charge Q is
a required input to the primary decoder and evaluation. A separate development
pilot withholds Q; it does not change the released model or the locked results.

The 488 development poses are not encoder-training identity-disjoint: the retained
original cache audit finds 66 pose records in its training fold (75 distinct pose
records across all cached folds). The CCD control has 49 training-fold hits.
These counts are conditional on the retained cache representing training-time bytes:
the original checkpoint did not record that cache's hash. No direct matches to the
completion head's own training-source molecules were found. Do not add fold-wise
hit counts because the same chemical identity can occur in more than one fold.

The locked 491 reference identities have zero hits against the saved complete-source
exclusions under both canonical keys. This later check was identity-only; it did
not make new locked predictions or reset the locked state.

The locked run was interrupted and resumed with an explicit raw500/eligible491
population-contract repair. The state has `is_untouched_test=false`; do not describe
it as a certified untouched or single-read experiment. States, content hashes and
migration receipts remain under `results/`.

## 2. Endpoint definitions

Graph-level strict and resonance-aware recovery compare bond labels. Aromatic
equivalence is validated by RDKit at connected-component level with reference atom
states fixed for this scoring-only check; it is not a ring double-bond-count rule.
Graph canonical-SMILES recovery uses a separate builder that zeroes formal charges
and re-infers hydrogens on both sides. None of these tests predicted h or q.

Native molecular identity builds the molecule from saved predicted bonds, formal
charges and hydrogen counts, then compares non-isomeric canonical SMILES with the
annotated reference. It retains predicted q/h, but is not atom-indexwise exactness
of the entire (B,h,q) triple and does not test stereochemistry or exhaustively
normalize tautomers/formal-charge resonance drawings.
It is a separate endpoint, with no paired external baseline on the locked cohort.
There is now a paired **same-project completed archived-system** baseline (§6.5),
not an independent external native model or a fully non-neural end-to-end method.
Internal development controls compare ordinary/masked training and neural atom
logits versus catalogue-only atom inference. Neither is an external locked native
baseline. A subsequent same-B, same-budget CCD control is reported in §8.2; it is
post hoc on an already evaluated cohort, not an independent end-to-end comparison.
The delocalized output convention is a graph-level postprocessing step and has no
native-identity counterpart. Designed chemical controls confirm that it can gain
graph credit without a reference-equivalent complete state; see §8.2.

Graph-level endpoints are not interchangeable. Typed true-edge recall is a local
diagnostic and does not penalize false-positive edges.

## 3. Existing result files and headline values

Current development rescoring is under:

- `work/evaluation_fix_v3/development/unified/`;
- `work/evaluation_fix_v3/development/unified_resonance_summary.json`;
- `work/evaluation_fix_v3/development/holm_development.json`.

The correct aggregation root is `work/evaluation_fix_v3/development`, not
`experiments/prob_mask_20261002/joint_v6`. The aggregator appends `unified/`
itself.

Locked results are under `experiments/prob_mask_20261002/locked_eval/`.
Use the `*_fullgraph.json` verdicts for the manuscript, not reference-edge-projected
exports. The final multiplicity artifact is `holm_confirmatory.json` (78 tests).
The filename is historical and is not an untouched-status certificate.

| Endpoint | Development mean +/- SD | Locked mean +/- SD |
|---|---:|---:|
| Strict graph, selector D | 0.6626 +/- 0.0113 | 0.6904 +/- 0.0106 |
| Resonance-aware graph, selector D | 0.8299 +/- 0.0094 | 0.8344 +/- 0.0072 |
| Canonical-SMILES graph, selector D | 0.7842 +/- 0.0012 | 0.7766 +/- 0.0047 |
| Native molecular identity, selector C | 0.8320 +/- 0.0102 | 0.8289 +/- 0.0143 |

A keeps the bond-argmax candidate with its neural-plus-catalogue, charge-constrained
atom assignment. B selects the first constructible candidate by bond probability;
C adds the normalized atom-score ranking term; D applies the delocalized graph
convention. B already uses neural atom predictions for construction. The C-B ladder
is not an ablation of the whole atom predictor. Only B/C share the unrestricted
active pool/filter; diagnostic rows can disable filtering or restrict moves.

The development native value is now reported by the atom-logit ablation counterpart.
Its original-model final bonds/buildability/native outcomes reproduce the archived
outputs for every model-molecule instance. It is not a new model attached to old
locked scores, and native identity is evaluated before D.

Both the 108-test development and 78-test locked Holm families remain unchanged.
Drivers retain additional archived comparisons for provenance and the original
correction. The manuscript now displays the archived raw graph beside current
outputs; its completed native output is identified separately.
Do not recalculate a smaller correction family after removing table columns.

## 4. Source/model verification and development smoke

```powershell
D:\minconda\python.exe experiments\prob_mask_20261002\verify_frozen_config.py
D:\minconda\python.exe experiments\prob_mask_20261002\frozen_manifest.py
D:\minconda\python.exe experiments\prob_mask_20261002\test_eval_v3.py
```

The configuration verifier may write a diagnostic JSON under results/; it does not
rerun model inference. The exact manifest check is read-only. In the current
diagnostic checkout it intentionally reports the two registered changed source
files and six added diagnostic files; it must not be made green by overwriting
the October frozen manifest. Use the separate diagnostic inventory below to
check these declared changes, and the historical source snapshot when reproducing
the original byte-frozen environment. A development-only
end-to-end smoke creates artifacts under work/, not locked outputs:

```powershell
D:\minconda\python.exe experiments\prob_mask_20261002\smoke_pipeline_v3.py
```

Do not regenerate or overwrite locked results just to read the paper. The recorded
state is COMPLETE; historical driver commands remain available for audited
reproduction in a separately prepared environment. This guide does not reset that
state or create a new untouched test by renaming its outputs.

## 5. Development rescoring and plots

Rescoring existing development predictions (no training):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_unified_resonance.ps1
```

That driver writes the current root above and retains the full 108-test family.
To aggregate already-scored flags without rescoring:

```powershell
D:\minconda\python.exe experiments\prob_mask_20261002\unified_aggregate.py --root work\evaluation_fix_v3\development --output work\evaluation_fix_v3\development\unified_resonance_summary.json
```

Generate the two current-method figures from recorded summary values:

```powershell
D:\minconda\python.exe bondnet2_paper\plot_current_figures.py
```

This writes `figures/fig1_single_stage.pdf` and
`figures/fig6_current_regimes.pdf`. The unchanged terminal-oxygen figure is
`figures/fig3_terminal_oxy.pdf`, based on the geometric separability audit.
The manuscript no longer includes the historical coverage or error-analysis figures.

## 6. Supplementary matched cold-start study (2026-11-01)

This study is separate from the released warm-start models and original locked
evaluation. Do not copy its development scores into the main locked table.

- Root: `experiments/mask_control_20261101/`.
- Protocol and completion state: `PROTOCOL.json`, `STATUS.json`.
- Full results: `SUMMARY.json`, `RESULTS.md`, `RESEARCH_REPORT.md`.
- Integrity: `FINAL_VERIFICATION.json`.
- Provenance: `IDENTITY_AUDIT.json`, `ENCODER_CACHE_METADATA.json`,
  `ENCODER_SPLIT_IDENTITY_AUDIT.json`, `LOCKED_IDENTITY_ONLY_AUDIT.json`.
- Paired original-output audit: `HISTORICAL_ERROR_TRANSITIONS.json`.
- Timing: `TIMING.json`.

Ordinary and masked arms have the same architecture (388,499 parameters), within-seed
random initial state, data order and 18-epoch budget. Both start cold, with no
mask-pretrained pilot. Both use the same frozen encoder and training feature cache.
Seeds are 20261101, 20261102 and 20261103. The primary comparison uses one all-MASK
forward pass and reuses atom logits for each candidate. Models have their own
candidate contents but identical algorithms/budgets. The secondary masked conditional
readout shares the exact masked marginal beam and bond scores. Catalogue, Q solver
and constructibility filter are identical; no arm applies delocalized rewriting.

Native identity, percentages as mean +/- training SD:

| Input | Ordinary marginal | Masked marginal | Masked conditional |
|---|---:|---:|---:|
| Pose clean, n=488 | 82.72 +/- 0.31 | 83.20 +/- 0.41 | 82.86 +/- 0.72 |
| Pose sigma=0.10, n=488 | 76.71 +/- 0.78 | 74.93 +/- 3.00 | 75.07 +/- 2.67 |
| CCD ideal, n=1000 | 82.13 +/- 0.83 | 82.23 +/- 0.67 | 82.30 +/- 1.30 |

No positive comparison passes the separate 72-test Holm family. One noisy-pose seed
has significant adverse native and graph-SMILES contrasts; this is not uniform harm
across seeds. Identity-cluster intervals are conditional on the fixed trained models,
not confidence intervals over all possible training seeds. Conservative known-source-
excluded subsets (407 poses, 926 CCD) are descriptive and show the same broad pattern.
The original 78/108 families remain unchanged.

Resume/verify the separately recorded study, without touching locked predictions:

```powershell
D:\minconda\python.exe -B -u experiments\mask_control_20261101\study.py run --device cuda
D:\minconda\python.exe -B experiments\mask_control_20261101\report.py
```

The completed-state guards skip completed cells and verify fingerprints. Do not delete
completion guards, use forced retraining, or reset the locked run to make this command
run again. The optional original-cache membership audit requires the retained training
cache; the saved audit records its provenance limitation. That old cache is not needed
for inference or for the matched head-training run from the provided local features.

Timing is for these new models only: 50 fixed poses x 3 repeats, CUDA synchronized,
including failed builds. Ordinary argmax/search medians are 10.60/50.97 ms; masked
marginal/conditional search medians are 52.68/55.83 ms. These are not latency claims for
the original main checkpoints or cross-tool throughput comparisons.

### 6.1 Inference-time neural atom-logit ablation

Root: `experiments/catalogue_only_20261101/`. Current protocol is
`PROTOCOL_v3.json`, current cells are under `evaluations_v3/`. Earlier protocol
versions and exact source snapshots are preserved, not pooled as additional seeds.

The original three released heads share one exact candidate pool and bond scores
within each pair. Replacing neural atom logits with zero leaves catalogue, Q solving,
normalized scoring, filtering and ranking unchanged. Bond predictions remain neural.
This changes inference-time atom assignment/scoring, not the atom supervision used
when training the checkpoint.

| Input | Neural + catalogue native (%) | Catalogue-only (%) | Difference pp |
|---|---:|---:|---:|
| Pose clean | 83.20 +/- 1.02 | 77.66 +/- 0.71 | +5.53 |
| Pose sigma=0.10 | 76.30 +/- 1.57 | 72.06 +/- 0.78 | +4.23 |
| CCD ideal | 82.53 +/- 2.20 | 72.43 +/- 0.85 | +10.10 |

Native comparisons pass the separate 36-test Holm correction in 3/3, 2/3 and 3/3
head seeds respectively. Source-excluded subsets have the same direction.
Graph-SMILES gains are not uniform. Do not call native identity atomic-charge
accuracy or claim every charged subgroup improves.

Primary outputs: `SUMMARY.json`, `RESEARCH_REPORT.md`, `DIAGNOSTICS.json`,
`OUTPUT_VERIFICATION.json`, `FINAL_VERIFICATION.json`.

```powershell
D:\minconda\python.exe -B experiments\catalogue_only_20261101\study.py run --device cuda
D:\minconda\python.exe -B experiments\catalogue_only_20261101\verify_outputs.py
D:\minconda\python.exe -B experiments\catalogue_only_20261101\report.py
```

Completed cells are guarded by protocol/input hashes. No locked evaluation is run.
The stored pools permit offline charge/score and native-build verification.

### 6.2 Post hoc diagnostics added on 2026-11-02

Saved diagnostic outputs are in `work/reviewer_checks_20261102/`:

- Graph-SMILES reference self-score: 444/488 development and 437/491 locked.
  The locked ceiling 89.00% is intrinsic to the charge-zeroing proxy. Its common
  failures do not change McNemar discordant pairs; conditional accuracy differences
  and bootstrap intervals have different denominators and are not automatically
  unchanged. Original scores and intervals are retained.
- Atom renumbering: 120 development molecules, three nontrivial permutations each,
  one head checkpoint, clean coordinates. No selected bond/state/identity/build
  changes were observed. This does not prove permutation invariance, posterior
  sharpness or equality with exact MAP.
- External completion: reference/our/Knodle graphs completed by the same catalogue,
  charge solver and native builder. Our graph is the already selected stage-C graph
  from seed20261002. The unadapted native rates are 442/491, 378/491 and 226/491
  with reference Q. The first is a reference-bond control, not an upper bound;
  85.5% and 51.1% are ratios to that control, not fractions of a theoretical ceiling.
  The forced-Q0 external sensitivity is 220/491.
- The global graph-matching adapter is a failed control, not an encoding correction:
  reference native falls 0.9002 -> 0.5764; our native 0.7699 -> 0.4521; external
  native 0.4603 -> 0.4155 despite constructibility increasing 0.5397 -> 0.6436.
  Its source and outputs are retained unchanged for provenance. The chemical
  caveat about its historical comments is recorded in the location-audit README.
- The existing noise-seed diagnostic uses a 120-molecule development subset and
  must not replace the full-cohort three-head means in the original main tables.

The saved-output location audit is now separately reproducible:

```powershell
D:\minconda\python.exe -B experiments\encoding_location_audit_20261102\analyze.py
```

It writes `experiments/encoding_location_audit_20261102/SUMMARY.json`, with source
digests, exact populations and unchanged-input checks. Reference/our/Knodle class-4
edge counts are 6045/6069/6600, but non-ring counts are 0/0/636 and terminal-oxygen
counts 0/0/595. Thus a shared class number does NOT establish common chemical usage
or equal builder compatibility. This audit measures a representational difference,
not its causal share of the external composite's deficit. No new inference or
historical scoring is performed.

The source/data provenance record `results/PDBBIND_SOURCE_2026-10-04.md` documents
the package as a v2020/v2024 intersection with v2024-reprocessed structures.
The geometric terminal-oxygen audit uses 9517 ideal CCD molecules, not the model's
fixed 1000-molecule CCD control. Its scripts are retained under
`archive/legacy_two_stage_20261003/scripts/measure_terminal_oxy_separability.py`
and `archive/legacy_two_stage_20261003/scripts/plot_terminal_oxy.py`.

The separate diagnostic inventory does not alter the original frozen manifest:

```powershell
D:\minconda\python.exe -B experiments\prob_mask_20261002\diagnostic_manifest.py --frozen results\FROZEN_MANIFEST_V3_1_2026-10-05.json --output results\DIAGNOSTIC_MANIFEST_2026-11-02_v2.json --check
```

The revised guard checks added files against the actual original source-inventory
scope, not just against declared filenames; undeclared additions fail. Passing this
registration check is NOT proof of default-output equivalence. Keep the original
manifest and the previous diagnostic receipt; do not reset or rerun locked outputs.

### 6.3 Original graph comparison after conservative source exclusions

Root: `experiments/source_excluded_sensitivity_20261101/`. This filters original
saved D-selector and Knodle unified flags; it does not infer models or rescore graphs.

Pose exclusions use saved GEOM non-isomeric identities OR connectivity-only
skeletons (ignoring bond type, H, charge and stereo), plus direct head-training hits.
CCD exclusions instead use stereo-retaining OR non-isomeric identities from the
saved GEOM/Diverse/development-pose source union, plus direct head-training hits.
These different conservative rules retain 407 poses and 926 CCD molecules.
They are not the locked cohort's two-key filter and do not create an untouched test.

```powershell
D:\minconda\python.exe -B experiments\source_excluded_sensitivity_20261101\analyze.py
```

`ANALYSIS_MANIFEST.json` fingerprints inputs and exclusions; `SUMMARY.json` retains
all three graph endpoints; `RESULTS.md` gives full/subset comparisons.
Clean-pose resonance difference is +2.21pp (unadjusted cluster CI -1.06 to +5.50);
pose sigma=0.10/0.15 differences are +63.06/+68.39pp; ideal CCD remains -6.19pp.
This is post hoc descriptive sensitivity: no new p-values, no copied significance
marks from the original full cohort, and no changes to any correction family.

### 6.4 Within-group oxygen bond-length ranks

Root: `experiments/within_group_geometry_20261102/`. This new post hoc coordinate-only
diagnostic uses the same9517 ideal CCD and488 pose molecules as the pooled audit;
old pooled bond counts and means reproduce exactly. It does not rescore models.
Select C/P centres with>=2 terminal heavy-degree1 oxygens, exactly one reference
double bond and all other bonds single. Choose the shorter bond for two oxygens,
or shortest for larger groups. Ties within1e-6A receive1/k credit. Random choice is
1/nO. Groups are pooled with5,000 molecule-cluster bootstrap draws; identity-cluster
sensitivity and every group record are retained.

| Stratum | Pose groups / molecules | Ideal credit (%) | Pose credit (%) | Pose molecule-cluster95% CI | Chance (%) |
|---|---:|---:|---:|---:|---:|
| C, two O |134 /107|99.51|46.27|38.23–54.55|50.00|
| P, two O |51 /31|97.94|58.82|44.64–71.88|50.00|
| P, three O |53 /39|98.68|33.96|22.58–45.65|33.33|

These are **within-group** localization diagnostics, distinct from pooled AUC.
None proves chemical equivalence or non-identifiability from the full Z/R/Q input.
The one ideal three-O carbon group is kept in detailed output, not generalized.

```powershell
D:\minconda\python.exe experiments\within_group_geometry_20261102\analyze.py
D:\minconda\python.exe experiments\within_group_geometry_20261102\verify.py
```

### 6.5 Archived raw graph table completion and completed native comparator

The complete-output native comparison is documented in
`experiments/milp_native_completion_20261102/FINDINGS.md`. It replays2467 frozen
raw graphs (491+488+488+1000), then explicitly normalizes predicted delocalized
groups and infers H with the archived RDKit output policy. Final B/q/H is saved
and independently rebuilt with implicit inference disabled. It must not be
confused with the raw graph row. Native family12 is post hoc and separate from
all older families. Overall locked native is82.89±1.43% versus74.34%; most of
that net difference is output availability. On ideal CCD, the archived system is
more accurate in the output-conditioned both-buildable subsets. It is not a
controlled architecture or atom-state-learning ablation.

The raw graph table completion uses only saved edge dumps, unified verdicts and
unchanged source-exclusion sets:
`experiments/archived_graph_tables_20261102/` (`analyze.py`, `PLAN.json`,
`SUMMARY.json`, `VERIFICATION.json`, `REPORT.md`). No inference, new scoring
definition or extra hypothesis tests are involved.

| Condition | Strict | Resonance | Graph-SMILES | Typed correct / reference | Typed recall |
|---|---:|---:|---:|---:|---:|
| Locked clean491 |0.6171|0.8147|0.7576|14003/14518|0.9645|
| Development clean488 |0.5840|0.8012|0.7623|13727/14269|0.9620|
| Development sigma0.05 |0.5205|0.7643|0.7254|13672/14269|0.9582|
| Development sigma0.10 |0.3934|0.5717|0.5512|13474/14269|0.9443|
| Development sigma0.15 |0.1311|0.2090|0.2316|12690/14269|0.8893|
| CCD ideal1000 |0.7390|0.9550|0.7640|27202/27842|0.9770|
| CCD sigma0.10 |0.4860|0.6780|0.5530|26556/27842|0.9538|

On the unchanged407-pose subsets, archived resonance is80.59/77.15/57.25/19.41%
at noise0/.05/.10/.15; on the926-CCD subsets it is95.14/67.28% at0/.10.
The fixed exclusions are not an independent audit of the archived decoder's
separate training corpus. All current/Knodle subset cells and old families remain
unchanged. Main graph tables and Figure6 display all three systems; Figure6 uses
the new saved-output summary for the archived curve, not normalized native graphs.

```powershell
D:\minconda\python.exe experiments\archived_graph_tables_20261102\analyze.py
D:\minconda\python.exe -c "import sys;sys.path.insert(0,'bondnet2_paper');import plot_current_figures as p;p.regimes()"
```

## 7. Training and implementation facts

- Encoder: GEOM heavy-only, cached candidate radius 3.0 A; radial filter cutoff 2.5 A.
- Current completion/inference candidate graph: every heavy-atom pair within 2.5 A,
  with both directions and **no degree-15 truncation**.
- Head/catalogue source: 3,000 GEOM + 1,500 charged Diverse molecules, not GEOM alone.
  Three geometries per molecule give 13,500 training examples; validation is 550
  molecules / 1,650 geometries. The catalogue counts training source molecules only.
- Encoder reads Z and R; the head additionally receives Q. Q is required.
- Candidate atom probabilities are a normalized product of neural and catalogue
  terms, not an unnormalized score with a removable prior normalizer:

```text
atom_logprob = log_softmax(neural_atom_logits(B) + log_catalogue_prior(B))
candidate_score = sum(edge_logprob[chosen_B]) + sum(atom_logprob[chosen_qh])
```

The normalization depends on the candidate B and must remain when ranking graphs.
The default charge solver enforces sum(q)=Q but is not exact joint MAP.

The old “conserved edge budget” explanation is removed. The saved-output A-to-C
audit rescues 19/14/11 graphs and harms 5/4/7 across the three original seeds, including
false-positive pairs. Improved whole-graph recovery can coexist with unchanged or
increased edge errors; the algorithm does not transfer errors between molecules.

### 7.1 Pilot warm-start chain and recorded runtime

The frozen joint heads load **only edge-decoder weights** from
`experiments/prob_mask_20261002/pilot_v2/decoder_best.pt`; new atom modules start
randomly. The checkpoint is epoch10 of12, selected by minimum fully masked bond
validation CE. Pilot training uses neutral connected GEOM3000 train/300 validation
identities, historical-test exclusion and cross-split canonical deduplication;
noises0/0.1/0.2A, seed20261002, width96, three context-aware layers, batch32,
AdamW lr3e-4/wd1e-4, clipping2. Source subset: `source_subset.pt`; pilot feature cache:
`features.pt` at that experiment root, not `joint_v1/features.pt` used for the joint
heads. Preparation/selection are in `probe.py`, `prepare.json` and
`pilot_v2/training.json`; final-head initialization is in `joint_seeds_train.py`
and checkpoint arguments. Do not rerun preparation/training over archived files;
use an isolated copy for full reconstruction.

Pilot checkpoint SHA256:
`f014110a9fb2082840a8759276315777709925d05002e43eaa88c981842f2595`.
The full pilot/subset/cache/final-head fingerprints and args are in
`experiments/within_group_geometry_20261102/PROVENANCE.json`. Historical paths point
to the old workspace; files are retained in this tree and commands run here.
Current analysis/control runtime: RDKit2025.09.6, NumPy2.2.5, PyTorch2.5.1.
Historical checkpoints do not record their RDKit version; do not infer it from
the current installation.

## 8. Scope of claims

Clean CCD resonance-aware recovery is 0.8210 for the current decoder versus 0.8810
for Knodle. Two corrected seed-wise contrasts favour Knodle significantly; the third
has p=0.0535. Perturbed-coordinate results favour the current decoder, but the
comparison is input- and training-asymmetric. Do not infer uniform superiority or
performance on real generated molecules from synthetic noise alone.

Candidate search and constructibility filtering improve the native endpoint.
The extra atom-score ranking effect does not survive the original correction;
that does not mean the atom predictor is unnecessary, because B already uses it.
The separate atom-logit ablation supports its inference-time contribution.
The matched cold-start control further finds ordinary joint classification with
the same search budget competitive. This does not prove all masking ineffective,
nor fully ablate the original warm-start training recipe.

Current-method missing/wrong-charge stress tests, calibrated set-coverage guarantees
and comprehensive motif-stratified error counts are not reported. Empirical native
candidate coverage is now reported separately below; it is not a calibration claim.
Older experiments are not relabelled as current-model results. Unrecomputed
rule-based resonance cells are omitted.

### 8.1 Joint (q,h) candidate extension and frozen CCD holdout

This extension fixes the B already selected by the original stage-C decoder.
It does not change the original pose/CCD tables, training, masks or model weights.
With all atom states masked and true total Q given, it recomputes neural+catalogue
atom scores, enumerates the 31 globally highest-scoring joint assignments under
sum(q)=Q, prepends the unchanged single assignment, and tries at most 32 assignments.
At most 8 constructible, canonical-unique species are returned, baseline first.
Delocalized graph rewriting is not applied to this native endpoint.

Development inputs: `experiments/atom_state_candidates_20261102/development.pt`,
332 molecules (100 cation / 82 anion / 50 separated-neutral / 100 uncharged-neutral).
The CCD ideal-coordinate source snapshot is
`data/ccd_components_pub_20261102.sdf.gz`, from the official wwPDB endpoint recorded
in PREPARATION_PLAN.json. Eligibility, exact source hashes and split membership
are in DATASET.json. The old23 separated-neutral diagnostic cases are excluded.

Two equal-budget development methods were compared. Plain coverage@8 on the 50
separated-neutral molecules is 50.67%, versus 47.33% for charge-mass stratification;
single-answer native is 20.00%. Only plain k-best was selected for the holdout.

Frozen holdout: `experiments/atom_state_confirm_20261102/`:

- 280 unique identities,277 skeleton clusters; groups100/46/34/100.
- Primary endpoint: coverage@8 on34 separated-neutral molecules (32 skeletons).
- Protocol SHA256:
  `b3189b8794ae844d534c5be40b973df6906ad67d2225834ace4f3fc73f08bf2c`.
- Frozen UTC03:54:15 on2026-11-02; first held-back forward UTC03:54:19.
- Development-only implementation preflight:12 cases×3heads,36 exact checks.
- All original heads are reported, never best-of-three. No training or tuning.
- 5,000 paired skeleton-cluster bootstrap draws, molecule-weighted, fixed three
  heads; molecule bootstrap sensitivity; head SD separately. Secondary intervals
  are pointwise, not simultaneous. Formal hypothesis/ Holm family size0; old
  correction families remain unchanged.

| Stratum | n | Single native (%) | Coverage@8 (%) |
|---|---:|---:|---:|
| Cation |100|62.67±2.31|72.00±1.00|
| Anion |46|47.83±4.35|54.35±3.77|
| Separated-neutral |34|26.47±2.94|57.84±11.14|
| Uncharged-neutral |100|84.00±2.00|87.67±2.08|
| Pooled |280|63.45±2.03|72.98±2.70|

Primary95% skeleton-cluster interval:44.79–70.59%; molecule sensitivity46.08–69.61%.
The primary head counts are18/24/17 versus single9/10/8, each denominator34.
Mean primary returned count5.03; not every molecule returns8. Four-stratum macro
is55.24→67.96%, not the pooled63.45→72.98%.

Reproduce reports/verification without new model forward passes:

```powershell
D:\minconda\python.exe -B -u experiments\atom_state_confirm_20261102\run.py report
D:\minconda\python.exe -B -u experiments\atom_state_confirm_20261102\verify.py
```

All 26,880 raw assignments were independently rebuilt, 755,712 atom states checked,
baseline replay mismatches0, and the primary interval independently reproduced.
The holdout is now evaluated, not available for further tuning as untouched data.

Important provenance correction: the actual encoder is GEOM-cache trained, not
Diverse80k trained. The frozen preparation README/plan's "encoder Diverse80k"
label is inaccurate; its file-level exclusions still executed as recorded.
An additive correction, without rewriting frozen bytes, is at
`experiments/atom_state_candidates_20261102/SOURCE_SCOPE_CORRECTION.md`.
GEOM identities were excluded by the complete saved source union, while the original
skeleton exclusion covered Diverse80k, head sources and old evaluation inputs.
A new post-outcome read-only audit of all269,739 retained raw GEOM records finds
18/332 development and8/280 held-back skeleton hits; none among the34 primary cases.
These are source matches, not necessarily training-fold matches. No samples are
removed or scores replaced after this audit. Reproduce with:

```powershell
D:\minconda\python.exe -B -u experiments\atom_state_candidates_20261102\posthoc_geom_skeleton_audit.py
```

Native coverage is whether the reference occurs anywhere in a returned prefix,
not automatic-selection accuracy or a calibrated posterior. Baseline retention
guarantees nondecrease; even coverage@1 may spend32 attempts if the baseline is
unbuildable. All new inputs are ideal CCD coordinates with curated atom states,
not experimental protonation or real bound-pose validation.

### 8.2 Post-outcome same-budget scoring and chemical controls

Root: `experiments/reviewer_controls_20261102/`. Full interpretation and limits:
`FINDINGS.md`; executed plans: `MATCHED_PLAN.json` and `D_CONTROL_PLAN.json`.
No original frozen outputs, protocol definitions or correction families change.

All 280 previously evaluated CCD holdout identities and all three original heads
are included. Each arm receives the same saved full-model-selected B, Q, catalogue,
greedy conditional single solver, exact k-best enumeration and builder. Each uses
its own single assignment plus 31 k-best assignments: exactly 32 raw construction
attempts and at most8 unique returned species. Only atom-state scoring differs.
Thus this is conditional on a neural-selected B, not an external non-neural model.

| Group | n | Neural single (%) | Catalogue single (%) | Neural cov@8 (%) | Catalogue cov@8 (%) |
|---|---:|---:|---:|---:|---:|
| Cation |100|62.67±2.31|49.67±1.53|72.00±1.00|69.67±0.58|
| Anion |46|47.83±4.35|41.30±3.77|54.35±3.77|46.38±4.53|
| Charge-separated neutral |34|26.47±2.94|26.47±2.94|57.84±11.14|54.90±11.14|
| Uncharged neutral |100|84.00±2.00|65.33±0.58|87.67±2.08|86.67±2.08|
| Pooled |280|63.45±2.03|51.07±1.29|72.98±2.70|70.12±2.68|

Values are percentages, mean±sample SD over three fixed heads. Paired pointwise
skeleton-cluster bootstrap intervals (5,000 draws, heads averaged before resampling):
pooled original single-answer delta +12.38pp [8.33,16.49]; pooled first-return
coverage@1 delta +7.14pp [4.03,10.40]; pooled coverage@8 +2.86pp
[1.19,4.81]; charge-separated-neutral coverage@8 +2.94pp [-2.08,10.42]. These
are post hoc descriptive intervals, not multiplicity-adjusted significance claims.
The evidence supports useful single-answer atom scoring, but does not establish
an additional neural candidate-coverage advantage on the34-molecule group.

The pooled decomposition, using the **same already evaluated280-identity cohort**:

| Endpoint | Neural + catalogue (%) | Catalogue-only (%) | Difference pp |
|---|---:|---:|---:|
| Original single native |63.45±2.03|51.07±1.29|+12.38|
| First returned after search (coverage@1) |66.07±2.34|58.93±1.79|+7.14|
| Set coverage@8 |72.98±2.70|70.12±2.68|+2.86|

The first returned species is a reference-free selection rule. No additional
candidate re-ranker is evaluated; coverage@8 is not single-answer accuracy.
The initial candidate experiment is prospectively frozen; the later catalogue
attribution reuses that cohort post hoc, not a second independent test set.

Fourteen designed chemical controls execute the actual D rewrite with its real
training table and the actual graph scorer. Fixed-atom chemical equivalence retains
H locations and total Q and checks RDKit-enumerated resonance forms with q/B intact.
For class5, enumerated binary bond completions retain predicted q/H. Valid
different-species same-Q geminal-diol/acid examples become graph-accepted after D
without any state-compatible reference-resonance completion. This proves a possible
interpretation failure, not its prevalence on491 poses. D is a graph-level partial
representation, not chemical-state repair. Native state claims use C before D.

Verification rebuilt53,760 raw assignments. All840 full-arm single answers and
returned species sequences replay exactly; two head/molecule records have raw CPU/
GPU assignment differences in nine unconstructible proposal positions, with no
endpoint change. Independent resampling reproduces the paired intervals. Details
are in `MATCHED_SUMMARY.json`, `D_CONTROL_SUMMARY.json` and `AUDIT.json`.

```powershell
D:\minconda\python.exe experiments\reviewer_controls_20261102\matched_control.py verify
D:\minconda\python.exe experiments\reviewer_controls_20261102\delocalization_controls.py
D:\minconda\python.exe experiments\reviewer_controls_20261102\audit_controls.py
```

## 9. LaTeX build

The mainline submission revision retains four main tables: native comparison,
locked graph results, decoder component ladder, and the atom-logit ablation.
Additional file 1 preserves the full comparison matrices and implementation
details. Sections S12--S14 contain the detailed methods, benchmark design, and
extended results; Tables S12--S16 contain the moved CCD, development-pose,
coordinate-regime, candidate coverage, and matched-budget results. This is an
editorial reorganization, not a rerun or a change to the frozen comparisons.

Fig. 1 is an editable vector schematic generated independently of result data:

```powershell
D:\minconda\python.exe bondnet2_paper\draw_fig1.py
```

This writes `figures/fig1_single_stage.pdf`, `.svg` and `.png` under this
paper directory. The shared plotting entry point delegates its overview to the
same script. The figure separates the frozen geometric encoder, supplied Q,
bounded local decoding, stage-C native identity, stage-D partial graph output,
and the optional fixed-bond atom-state candidate extension.
The compact integrated version uses a 5.15 x 4.62 inch canvas at the final
single-column width, with the NPG-style navy/cyan/teal/slate/salmon palette.
The scalar/vector interaction schematic is followed by an invariant readout;
the vector channel is not labelled as an invariant pair feature.
The previous overwritten script and its artwork are preserved under
`work/fig1_integrated_20261103/before/`.

This is a multi-file Springer Nature LaTeX project, requiring
`sn-jnl.cls`, `sn-basic.bst`, `references.bib` and the figures in this directory.
On a machine with a working TeX installation:

```powershell
Set-Location E:\zxcproject\bondnet2.0\bondnet2_paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_diagnostics.tex
bibtex supplementary_diagnostics
pdflatex -interaction=nonstopmode -halt-on-error supplementary_diagnostics.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_diagnostics.tex
```

The current submission PDFs are `release/submission_candidate/main_submission.pdf`
and `release/submission_candidate/additional_file_1.pdf`, relative to the repository
root. Earlier paper-directory PDFs and build files were moved to
`work/paper_history_cleanup_20261103/`. The revision notes below describe historical
snapshots, not current submission paths.
The 2026-11-01 revision adds the matched controls, timing and paired error audit,
corrects training-source/identity/radius/score descriptions, and moves the locked
evaluation before the retrospective development tables. Build and visual QA status
are recorded in `REVISION_REMAINING_2026-11-01.md` (earlier build history remains in
`REVISION_2026-11-01.md`). That revision's preview was
`main_revised_preview.pdf`.

The current remaining-issues revision adds the inference-time atom control and
original-system source-excluded sensitivity, corrects the native/ladder definitions,
and keeps all original numerical result cells. Build/QA status is recorded in the
current revision note. Pre-edit sources and preview are preserved under
`work/paper_remaining_fixes_20261101/before/`.
The preceding 46-page verification applies to the 2026-11-01 snapshot, not to
later revisions. Current build/QA status and the PDF hash are recorded in
`REVISION_2026-11-02.md`; do not reuse a prior preview's page count or checksum.
The morning2026-11-02 corrected preview was54 pages; that count does not describe
the later candidate-set revision. Its build, numerical and visual QA are recorded
in `REVISION_CANDIDATES_2026-11-02.md`. The previous table bodies are retained;
candidate-set results use a new separate table. Normal underfull-box spacing
diagnostics are distinguished from unresolved references or overfull boxes.
The subsequent prose-polish revision reduces that60-page candidate snapshot to
54pages without changing any of the15 table numerical bodies or displayed
equations. Current PDF hash and QA are recorded in
`REVISION_POLISH_2026-11-02.md`; the earlier candidate revision remains historical.
The later reviewer/within-group revision separates C-native from D graph output,
adds the coverage@1 interpretation, and moves selected diagnostic history and the
full conditional matrix to `supplementary_diagnostics.tex` (Additional file1).
Current checks and output hashes are in `REVISION_WITHIN_GROUP_2026-11-02.md`.
The later native-comparison/mainline revision and completed archived raw-graph
tables supersede those page counts. Current graph-cell, curve, abstract and build
checks are recorded in `REVISION_ARCHIVED_GRAPH_TABLES_2026-11-02.md`.

If a newly installed MiKTeX is not yet on PATH, use the full executable paths under
`C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64`.



## 10. Missing-net-charge development sensitivity

The separate pilot compares the frozen head and a charge-blind adaptation on
the existing 488 development poses and the charge-enriched 332 CCD ideal
development molecules. It does not access the locked pose or held-back CCD
cohorts. See `experiments/unknown_charge_20261108/README.md` for commands and
`FINDINGS.md` in that directory for the interpretation. Per-head predictions,
training histories and the machine-readable summary are staged locally in
`work/unknown_charge_20261108/`. The supplementary section **Sensitivity to
missing net charge** reports the comparison. The separate ring-coherence
study is not an evaluation of this neural decoder.

## 11. Release synchronization

The [public code repository](https://github.com/zxc8848/ligand-state-recovery)
releases the source under MIT, subject to its third-party notices.
[Zenodo v2.2](https://doi.org/10.5281/zenodo.24071196) retains the v2.1 data ZIP
and adds the CC BY 4.0 inference ZIP with three Stage C completion heads and
the aggregate atom-state catalogue. Use the
[supplied-charge inference instructions](https://github.com/zxc8848/ligand-state-recovery/tree/main/tools/release_inference)
to obtain the separate public encoder and run Stage C on supplied coordinates.

Commands above that require PDBBind-derived poses, original serialized training
checkpoints, training caches, per-molecule catalogue sources or historical
comparator inputs still require those exact files. The public archive does not
reconstruct the locked-pose benchmark end to end.

