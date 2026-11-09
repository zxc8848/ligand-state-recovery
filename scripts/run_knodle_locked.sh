#!/bin/sh
# The Knodle leg of the locked-set evaluation.
#
# external/knodle/Knodle is a statically linked ELF x86-64 Linux executable
# (built 2019, stripped), so it cannot run on Windows. It needs no libraries and
# no install: any x86-64 Linux with the repository mounted will do -- WSL, a VM,
# or the Linux shell attached to a Claude session.
#
# Inputs : work/paper2_locked_eval/pdb/*.pdb   (written by run_locked_eval.ps1,
#                                               CONECT-free, so the reference bond
#                                               graph is NOT handed to the baseline)
# Outputs: work/paper2_locked_eval/mol2/*.mol2 (adapted and scored back on Windows
#                                               by run_locked_eval.ps1 -Only knodle-score)
#
# Run from the repository root:
#     sh scripts/run_knodle_locked.sh
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/external/knodle/Knodle"
PDB="$ROOT/work/paper2_locked_eval/pdb"
MOL2="$ROOT/work/paper2_locked_eval/mol2"

[ -x "$BIN" ] || { echo "missing or non-executable $BIN" >&2; exit 1; }
[ -d "$PDB" ] || { echo "missing $PDB; run run_locked_eval.ps1 first (it writes the PDB exports)" >&2; exit 1; }
[ -f "$PDB/manifest.json" ] || { echo "missing $PDB/manifest.json; the export did not finish" >&2; exit 1; }

mkdir -p "$MOL2"
total=0; made=0; failed=0
for f in "$PDB"/*.pdb; do
  [ -e "$f" ] || break
  total=$((total+1))
  b=$(basename "$f" .pdb)
  # A molecule Knodle fails on is left without a mol2 file on purpose: the adapt
  # step counts a missing prediction as wrong, which is the intended accounting.
  # Do not substitute anything for it.
  if "$BIN" --ipdb "$f" -o "$MOL2/$b.mol2" >/dev/null 2>&1 && [ -s "$MOL2/$b.mol2" ]; then
    made=$((made+1))
  else
    failed=$((failed+1))
    rm -f "$MOL2/$b.mol2"
    echo "  Knodle produced no output for $b"
  fi
done

echo "pdb inputs $total   mol2 written $made   no output $failed"
echo "next, back on Windows:"
echo "  powershell -ExecutionPolicy Bypass -File scripts\\run_locked_eval.ps1 -Only knodle-score"
