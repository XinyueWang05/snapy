#!/bin/bash
# runcfg.sh <pyinst> <outroot> <MODE> <CLOSURE or -> : 8 cases x nb 1, 2 at 1 rank (W2 200 steps, 3-D 20 steps)
VB=/data00/ai_workspace/worker4/pr357/venv-head/bin; R=/data00/ai_workspace/worker1/mainsplit
T=/data00/ai_workspace/worker1/mr365/snapy/tests/test_axial_momentum_flux_form.py  # 0e240e15 helpers for both builds
export PYTHONPATH=$1 BACKEND=gloo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MODE=$3
[ "$4" = "-" ] && unset CLOSURE || export CLOSURE=$4
for c in f_slow f_deck g_slow g_deck h_slow h_deck w2_1 w2_4; do n=20; [[ $c == w2* ]] && n=200
  for nb in 1 2; do NB=$nb nice -n 19 $VB/python $R/drv/drvsplit.py $T $c $n $2/w1_nb$nb 2>&1 | grep -E "RESULT|Error"; done
done
