#!/bin/bash
# usage: run.sh <base|fix|roevt> <jobs.txt> <npar>
# Detached-safe: skips arms whose runs/<label>/<arm>.json exists; state in runs/<label>/<arm>.state.
# Builds (CPU Release, README cmake): base = a09cc256 installed in $VENV,
# fix = ba2b28ce and roevt = scheme1-roe-vt installed with pip --target into $LMX/site-<label>.
here=$(cd "$(dirname "$0")" && pwd)
LMX=${LMX:-/data00/ai_workspace/worker1/lmx}
VENV=${VENV:-$LMX/venv}
label=$1; jobs=$2; npar=${3:-5}
case $label in base) pp="" ;; *) pp="$LMX/site-$label" ;; esac
mkdir -p "$here/runs/$label"
one() {
  name=$1; shift
  out="$here/runs/$label/$name"
  [ -f "$out.json" ] && { echo "skip $name"; return; }
  echo "running start=$(date -u +%FT%TZ)" > "$out.state"
  PYTHONPATH="$pp" OMP_NUM_THREADS=1 "$VENV/bin/python" "$here/run_check.py" --out "$out" "$@" > "$out.log" 2>&1
  echo "exit rc=$? end=$(date -u +%FT%TZ)" >> "$out.state"
}
export -f one; export here label pp VENV
echo "launcher $label pid=$$ start=$(date -u +%FT%TZ)" >> "$here/runs/LAUNCHER.state"
xargs -P "$npar" -L 1 bash -c 'one "$@"' _ < "$jobs"
echo "launcher $label done $(date -u +%FT%TZ)" >> "$here/runs/LAUNCHER.state"
