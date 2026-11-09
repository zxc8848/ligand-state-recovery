<#
Fixed v3 evaluation. DryRun never predicts/scores the locked set.
A/B/C share a beam; B/C share buildability filtering. D is graph-only rewriting.
Existing partial outputs without a matching STARTED state are blocked, not erased.
Resume is only allowed with identical frozen bytes; it is not permission to tune.
#>
param(
  [ValidateSet('all','main','knodle-export','knodle-score')][string]$Only='all',
  [string[]]$Seeds=@('seed20261002','seed20261003','seed20261004'),
  [string]$Python='D:\minconda\python.exe', [string]$Device='cuda',
  [switch]$DryRun, [switch]$Resume
)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
Set-Location $Root
$Exp=Join-Path $Root 'experiments\prob_mask_20261002'
$Legacy=Join-Path $Root 'archive\legacy_two_stage_20261003'
$Out=Join-Path $Exp 'locked_eval'
$Work=Join-Path $Root 'work\paper2_locked_eval'
$Sdf=Join-Path $Root 'data\pdbbind_pose_locked500.sdf'
$Freeze=Join-Path $Root 'data\pdbbind_pose_locked500_freeze.json'
$Doc=Join-Path $Root 'results\FROZEN_CONFIG_V3_1_2026-10-05.md'
$Manifest=Join-Path $Root 'results\FROZEN_MANIFEST_V3_1_2026-10-05.json'
$Population=Join-Path $Root 'data\pdbbind_pose_locked500_population_v3_1.json'
$ExpectedN='491'
$StatePath=Join-Path $Root 'results\LOCKED_EVAL_STATE_V3.json'
$Marker=Join-Path $Root 'results\LOCKED_EVAL_DONE.json'
$Log=Join-Path $Root ('results\locked_eval_v3_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
$Seeds=@($Seeds | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$FixedSeeds=@('seed20261002','seed20261003','seed20261004')
if (($Seeds -join ',') -ne ($FixedSeeds -join ',')) { throw 'Seeds must equal the frozen ordered three-seed list' }
$runMain=$Only -in @('all','main'); $runExport=$Only -in @('all','knodle-export')
$runScore=$Only -eq 'knodle-score'
$State=$null

function Save-State {
  $tmp=$StatePath+'.tmp'
  $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -LiteralPath $tmp -Destination $StatePath -Force
}
function Step {
  param([string]$Title,[string[]]$Cmd,[switch]$Always)
  Write-Host "`n=== $Title ===" -ForegroundColor Cyan
  Write-Host ($Cmd -join ' ')
  foreach ($arg in $Cmd) { if ([string]::IsNullOrWhiteSpace($arg)) { throw "empty argument: $Title" } }
  if ($DryRun -and -not $Always) { Write-Host '(dry run, not executed)'; return }
  $previous=$ErrorActionPreference; $ErrorActionPreference='Continue'
  try { & $Cmd[0] $Cmd[1..($Cmd.Length-1)] 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath $Log -Append }
  finally { $ErrorActionPreference=$previous }
  if ($LASTEXITCODE -ne 0) { throw "step failed ($LASTEXITCODE): $Title; log=$Log" }
}
function Score {
  param([string]$Edge,[string]$Verdict,[string]$Label)
  Step "unified verdicts $Label" @($Python,(Join-Path $Exp 'unified_resonance.py'),
    '--per_edge',$Edge,'--reference_sdf',$Sdf,'--expected_n',$ExpectedN,
    '--population_manifest',$Population,'--label',$Label,'--output',$Verdict)
}
function Pair {
  param([string]$Old,[string]$New,[string]$Dest)
  Step "paired statistics $Dest" @($Python,(Join-Path $Exp 'paired_verdicts.py'),
    '--old_verdicts',$Old,'--new_verdicts',$New,'--reference_sdf',$Sdf,
    '--expected_n',$ExpectedN,'--population_manifest',$Population,'--output',$Dest)
}

# Preflight ALL dependencies before the first real inference, not seed by seed.
$Required=@($Sdf,$Freeze,$Doc,$Manifest,$Population,$Python,
  (Join-Path $Root 'checkpoints\diverse80k_heavy_stage1\best.pt'),
  (Join-Path $Root 'checkpoints\diverse80k_heavy_struct6_vector_hardneg_to50\best.pt'),
  (Join-Path $Root 'checkpoints\revision_v3_one_stage_heavy_seed42\best_e2e.pt'),
  (Join-Path $Exp 'joint_v1\sources.pt'))
$Required+=@($Seeds | ForEach-Object { Join-Path $Exp "joint_v5\$_\best.pt" })
$Required+=@('verify_frozen_config.py','frozen_manifest.py','pose_ablate_search.py',
  'pose_delocalize.py','unified_resonance.py','evaluation_common.py','paired_verdicts.py',
  'cluster_bootstrap.py','holm_family.py','ablate_aggregate.py','native_ladder_stats.py',
  'knodle_to_per_edge.py' | ForEach-Object { Join-Path $Exp $_ })
$Required+=@('benchmark_templatefree_motifs.py','p0d_knodle_compare.py' | ForEach-Object { Join-Path $Legacy "scripts\$_" })
foreach ($need in $Required) { if (-not (Test-Path -LiteralPath $need)) { throw "missing $need" } }
Step 'verify frozen configuration and behavioral regressions' @($Python,'-u',(Join-Path $Exp 'verify_frozen_config.py')) -Always
Step 'verify exact frozen code/checkpoint/input hashes' @($Python,(Join-Path $Exp 'frozen_manifest.py')) -Always
Step 'verify raw500/eligible491 fixed population' @($Python,(Join-Path $Exp 'population_manifest.py'),
  '--sdf',$Sdf,'--manifest',$Population,'--raw_n','500','--eligible_n',$ExpectedN) -Always
$have=(Get-FileHash -Algorithm SHA256 -LiteralPath $Sdf).Hash.ToLower()
$want=(Get-Content -LiteralPath $Freeze -Raw | ConvertFrom-Json).sha256
if ($have -ne $want) { throw 'Locked SDF hash mismatch' }
$mh=(Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLower()
$partial=@(Get-ChildItem -LiteralPath $Out -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @('.json','.jsonl') })
if (Test-Path -LiteralPath $StatePath) {
  $State=Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
  if ($State.manifest_sha256 -ne $mh) { throw 'Existing evaluation state belongs to different frozen bytes' }
  if ($runMain -and -not $DryRun -and -not $Resume) { throw 'Existing STARTED/FAILED/COMPLETE state: explicit -Resume required' }
} elseif ($partial.Count -gt 0 -or (Test-Path -LiteralPath $Marker)) {
  $message='Pre-existing locked results have no v3 STARTED manifest. Do not call this untouched. No files will be overwritten; review provenance before any recovery/new confirmatory experiment.'
  if ($DryRun) { Write-Warning $message } else { throw $message }
}
if ($runScore -and -not $DryRun -and ($null -eq $State -or -not $State.main_complete)) { throw 'Knodle scoring requires a completed matching v3 main run' }
if (-not $DryRun) {
  if ($null -eq $State) {
    $State=[pscustomobject]@{status='STARTED';started_at=(Get-Date -Format 'o');manifest_sha256=$mh;
      locked_set_sha256=$have;main_complete=$false;knodle_complete=$false;seeds=$Seeds}
  } else { $State.status='STARTED' }
  Save-State
}
New-Item -ItemType Directory -Force -Path $Out,$Work | Out-Null

try {
  $oldVerdict=Join-Path $Out 'unified_old.json'
  if ($runMain) {
    $oldEdge=Join-Path $Out 'old_per_edge.jsonl'
    $oldCmd=@($Python,'-u',(Join-Path $Legacy 'scripts\benchmark_templatefree_motifs.py'),
      '--reference_sdf',$Sdf,'--output',(Join-Path $Out 'old_benchmark.json'),
      '--per_molecule_out',(Join-Path $Out 'old_per_molecule.json'),'--per_edge_out',$oldEdge,
      '--methods','bondnet','--input_mode','heavy_only','--noise','0.0',
      '--stage1_ckpt',(Join-Path $Root 'checkpoints\diverse80k_heavy_stage1\best.pt'),
      '--refiner_ckpt',(Join-Path $Root 'checkpoints\diverse80k_heavy_struct6_vector_hardneg_to50\best.pt'),
      '--allowed_atomic_nums','6','7','8','9','14','15','16','17','33','35','53','83')
    Step 'fixed archived comparator' $oldCmd
    Score $oldEdge $oldVerdict 'old'
    $family=@(); $beams=@()
    foreach ($s in $Seeds) {
      $dir=Join-Path $Out $s; New-Item -ItemType Directory -Force -Path $dir | Out-Null
      Step "decode $s" @($Python,'-u',(Join-Path $Exp 'pose_ablate_search.py'),
        '--joint_ckpt',(Join-Path $Exp "joint_v5\$s\best.pt"),'--device',$Device,
        '--sdf',$Sdf,'--reference_from_sdf','--population_manifest',$Population,
        '--sigma','0','--outdir',$dir,'--tag','locked')
      $beam=Join-Path $dir 'pose_localmap_locked.json'; $beams+=@($beam)
      foreach ($kind in @('fullgraph','refedges')) {
        $suf=if ($kind -eq 'fullgraph') {'_fullgraph'} else {''}
        Step "graph-only deloc $s/$kind" @($Python,(Join-Path $Exp 'pose_delocalize.py'),
          '--per_edge',(Join-Path $dir "pose_localmap_locked_$kind.jsonl"),
          '--sources',(Join-Path $Exp 'joint_v1\sources.pt'),
          '--output',(Join-Path $dir "pose_localmap_deloc_locked_$kind.jsonl"),
          '--report',(Join-Path $dir "pose_delocalize$suf.json"))
        foreach ($sel in @('greedy','bondonly','localmap','localmap_deloc')) {
          $v=Join-Path $dir "unified_$sel$suf.json"
          Score (Join-Path $dir "pose_${sel}_locked_$kind.jsonl") $v "$s/$sel/$kind"
          $paired=Join-Path $dir "comparison_vs_old_$sel$suf.json"
          Pair $oldVerdict $v $paired
          if ($kind -eq 'fullgraph') { $family+=@($paired) }
        }
      }
      $native=Join-Path $dir 'native_ladder.json'
      Step "native B/h/q component contrasts $s" @($Python,(Join-Path $Exp 'native_ladder_stats.py'),
        '--beam',$beam,'--reference_sdf',$Sdf,'--expected_n',$ExpectedN,
        '--population_manifest',$Population,'--output',$native)
      $family+=@($native)
    }
    Step 'native A/B/C ladder (D has no native endpoint)' (@($Python,(Join-Path $Exp 'ablate_aggregate.py'),'--runs')+$beams+@('--labels')+$Seeds+@('--output',(Join-Path $Out 'component_ladder.json')))
    Step 'provisional main family, 42 tests (final family needs Knodle)' (@($Python,(Join-Path $Exp 'holm_family.py'),'--inputs')+$family+@('--expected_tests','42','--output',(Join-Path $Out 'holm_main_provisional.json')))
    if (-not $DryRun) { $State.main_complete=$true; Save-State }
  }
  if ($runExport) {
    Step 'export CONECT-free inputs' @($Python,(Join-Path $Legacy 'scripts\p0d_knodle_compare.py'),'export','--sdf',$Sdf,'--out_dir',(Join-Path $Work 'pdb'))
    Write-Host 'Run sh scripts/run_knodle_locked.sh under Linux; then -Only knodle-score.'
  }
  if ($runScore) {
    $preds=Join-Path $Out 'preds_knodle.json'; $knEdge=Join-Path $Out 'knodle_per_edge.jsonl'
    Step 'adapt Knodle (missing outputs become failures)' @($Python,(Join-Path $Legacy 'scripts\p0d_knodle_compare.py'),'adapt',
      '--sdf',$Sdf,'--mol2_dir',(Join-Path $Work 'mol2'),'--manifest',(Join-Path $Work 'pdb\manifest.json'),'--output',$preds)
    Step 'Knodle shared fullgraph format' @($Python,(Join-Path $Exp 'knodle_to_per_edge.py'),
      '--knodle_pred',$preds,'--reference_sdf',$Sdf,'--output',$knEdge,'--report',(Join-Path $Out 'knodle_to_per_edge.json'))
    $kv=Join-Path $Out 'unified_knodle.json'; Score $knEdge $kv 'Knodle'
    $family=@()
    foreach ($s in $Seeds) {
      $family+=@(Join-Path $Out "$s\native_ladder.json")
      foreach ($kind in @('fullgraph','refedges')) {
        $suf=if ($kind -eq 'fullgraph') {'_fullgraph'} else {''}
        foreach ($sel in @('greedy','bondonly','localmap','localmap_deloc')) {
          $paired=Join-Path $Out "$s\knodle_vs_new_$sel$suf.json"
          Pair $kv (Join-Path $Out "$s\unified_$sel$suf.json") $paired
          if ($kind -eq 'fullgraph') {
            $family+=@($paired,(Join-Path $Out "$s\comparison_vs_old_$sel$suf.json"))
          }
        }
      }
    }
    Step 'FINAL confirmatory Holm family, 78 tests' (@($Python,(Join-Path $Exp 'holm_family.py'),'--inputs')+$family+@('--expected_tests','78','--output',(Join-Path $Out 'holm_confirmatory.json')))
    if (-not $DryRun) { $State.knodle_complete=$true; Save-State }
  }
  if (-not $DryRun) {
    $State.status=if ($State.knodle_complete) {'COMPLETE'} elseif ($State.main_complete) {'MAIN_COMPLETE_KNODLE_PENDING'} else {'EXPORT_COMPLETE'}
    Save-State
    if ($State.main_complete) { $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Marker -Encoding UTF8 }
  }
} catch {
  if (-not $DryRun -and $null -ne $State) { $State.status='FAILED'; Save-State }
  throw
}
if ($DryRun) { Write-Host 'DryRun complete: exact dependencies/arguments checked; no locked inference/scoring. Pre-existing partial results remain blocked for a real run.' }
