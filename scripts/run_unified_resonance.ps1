<#
Score all three systems with ONE equivalence definition.

The problem. Two "resonance-aware" definitions were in use and the ledger tables
mixed them:

  classify()                      old two-stage and the new pipeline
  resonance_align_terminal_oxy()  Knodle

Diagnosed per molecule at pose sigma=0: the gap is 32.8pp on Knodle, and every
disagreeing molecule looks like "S6-O7 2->4, S6-O8 2->4" -- Knodle writes a
delocalized terminal-oxy group as AROMATIC, the old pipeline writes it as
class-5, this pipeline writes Kekule {1,2}. classify() accepts class-5 but not
all-aromatic, so it forgave one convention and penalised another.

CAREFUL: "unify on classify()" would understate Knodle by ~30pp. The archive
already contains the fair function, and says so in its own comment: "refusing it
would relax the metric for one method's convention and not another's". The
unified definition composes the archive's two reference-blind alignment passes,
which is what benchmark_templatefree_motifs.py does for the motif benchmark.

Knodle's predictions are first converted into the shared per-edge format
(knodle_to_per_edge.py) so that literally the same scorer runs on all three.
That converter reproduces the archived Knodle strict and canonical-SMILES
figures exactly, which is how it is validated.

Usage (from the repository root):

    powershell -ExecutionPolicy Bypass -File scripts\run_unified_resonance.ps1
#>

param(
  [string[]]$Conditions = @('pdbbind_sigma0p00', 'pdbbind_sigma0p05',
                            'pdbbind_sigma0p10', 'pdbbind_sigma0p15',
                            'ccd_sigma0p00', 'ccd_sigma0p10'),
  [string[]]$Seeds = @('seed20261002', 'seed20261003', 'seed20261004'),
  [string]$Selector = 'localmap_deloc',
  [string]$Kind = 'fullgraph',
  [string]$Python = 'D:\minconda\python.exe',
  [string]$OutputRoot = 'E:\zxcproject\bondnet2.0\work\evaluation_fix_v3\development'
)

$Conditions = @($Conditions | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$ErrorActionPreference = 'Stop'
$Root = 'E:\zxcproject\bondnet2.0'
Set-Location $Root
$Exp = Join-Path $Root 'experiments\prob_mask_20261002'
$Probe = Join-Path $Root 'work\paper2_noise_ccd_probe'
$Out = Join-Path $OutputRoot 'unified'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Log = Join-Path $Root "results\unified_resonance_$Stamp.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log), $Out | Out-Null

function Step {
  param([string]$Title, [string[]]$Cmd)
  Write-Host ''
  Write-Host "=== $Title ===" -ForegroundColor Cyan
  Write-Host ($Cmd -join ' ')
  for ($i = 0; $i -lt $Cmd.Length; $i++) {
    if ($Cmd[$i] -eq '') { throw "empty-string argument at position $i of '$Title'" }
  }
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)] 2>&1 | ForEach-Object { [string]$_ } |
      Tee-Object -FilePath $Log -Append
  } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) {
    Write-Host "full output of the failing step is in $Log" -ForegroundColor Red
    throw "step failed: $Title (exit $LASTEXITCODE)"
  }
}

foreach ($c in $Conditions) {
  $sdf = Join-Path $Probe "$c\reference_for_scoring.sdf"
  $oldEdge = Join-Path $Probe "$c\bondnet_per_edge.jsonl"
  $kn = Join-Path $Probe "$c\preds_knodle.json"
  foreach ($need in @($sdf, $oldEdge, $kn)) {
    if (-not (Test-Path $need)) { throw "missing probe artifact $need" }
  }
  $dir = Join-Path $Out $c
  New-Item -ItemType Directory -Force -Path $dir | Out-Null

  $knEdge = Join-Path $dir 'knodle_per_edge.jsonl'
  Step "convert Knodle to the shared per-edge format, $c" `
    @($Python, '-u', (Join-Path $Exp 'knodle_to_per_edge.py'),
      '--knodle_pred', $kn, '--reference_sdf', $sdf,
      '--output', $knEdge, '--report', (Join-Path $dir 'knodle_convert.json'))

  Step "unified resonance, Knodle, $c" `
    @($Python, '-u', (Join-Path $Exp 'unified_resonance.py'),
      '--per_edge', $knEdge, '--reference_sdf', $sdf, '--label', 'Knodle',
      '--output', (Join-Path $dir 'knodle.json'))

  Step "unified resonance, archived two-stage, $c" `
    @($Python, '-u', (Join-Path $Exp 'unified_resonance.py'),
      '--per_edge', $oldEdge, '--reference_sdf', $sdf, '--label', 'old two-stage',
      '--output', (Join-Path $dir 'old.json'))

  foreach ($s in $Seeds) {
    $dump = Join-Path $Exp "joint_v6\$s\$c\pose_${Selector}_c_$Kind.jsonl"
    if (-not (Test-Path $dump)) { throw "missing $dump; run run_rescore_fullgraph.ps1 first" }
    Step "unified resonance, new pipeline $s, $c" `
      @($Python, '-u', (Join-Path $Exp 'unified_resonance.py'),
        '--per_edge', $dump, '--reference_sdf', $sdf, '--label', "new $s",
        '--output', (Join-Path $dir "new_$s.json"))
    foreach ($base in @('old','knodle')) {
      Step "paired UNIFIED statistics $c/$s vs $base" `
        @($Python,'-u',(Join-Path $Exp 'paired_verdicts.py'),
          '--old_verdicts',(Join-Path $dir "$base.json"),
          '--new_verdicts',(Join-Path $dir "new_$s.json"),
          '--reference_sdf',$sdf,'--output',(Join-Path $dir "paired_${base}_$s.json"))
    }
  }
}

Step 'aggregate: one definition, three systems, every condition' `
  @($Python, '-u', (Join-Path $Exp 'unified_aggregate.py'),
    '--seeds', ($Seeds -join ','), '--conditions', ($Conditions -join ','),
    '--root',$OutputRoot,'--output', (Join-Path $OutputRoot 'unified_resonance_summary.json'))

$family=@()
foreach ($c in $Conditions) { foreach ($s in $Seeds) { foreach ($base in @('old','knodle')) {
  $family+=@(Join-Path $Out "$c\paired_${base}_$s.json")
} } }
Step 'Holm over the complete development family' `
  (@($Python,(Join-Path $Exp 'holm_family.py'),'--inputs')+$family+
   @('--expected_tests',[string]($Conditions.Count*$Seeds.Count*2*3),
     '--output',(Join-Path $OutputRoot 'holm_development.json')))
