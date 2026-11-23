# Block-split comparison (nb2 vs nb1), posted 2026-11-14 22:56 PT: recovered driver

Origin: worker1 job (session a93bb76c, last write 2026-11-15T06:56:38Z = 11-14 22:56 PT). Originals were in
/data00/ai_workspace/worker1/mainsplit/{drv/drvsplit.py, drv/runcfg.sh, drv/cmpsplit.py, report_split.md} and
/data00/ai_workspace/worker1/mr365/{drv/drv365.py, build.sh, snapy@0e240e15, pyinst}. Both folders were later removed by
scratch cleanup (not present 2026-11-23). These files were rebuilt verbatim from the session's tool calls
(drv365.py heredoc + its three later edits + the drvsplit transforms); they were not run here.

Files (sha256):
- drvsplit.py  09bd83cb...  one case per call; MODE=full|transport, CLOSURE unset/0/1, NB=1|2 (nb2=nb3=NB, 6*NB^2 blocks at 1 rank)
- runcfg.sh    cdaf1920...  runcfg.sh <pyinst> <outroot> <MODE> <CLOSURE|->: 8 cases x NB 1,2 (3-D 20 steps, W2 200)
- cmpsplit.py  caf2d9c6...  cmpsplit.py <root> <cfg...>: per-var max|u2-u1|/max|u1|, dt sequence equality
- build.sh     63fa7fbd...  CPU Release: cmake -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DCUDA=OFF -DUCX=OFF -DBUILD_TESTS=OFF; pip --target
- report_split.md           the original results table
- upstream_ref/             tests/test_axial_momentum_flux_form.{py,yaml} at 0e240e15 and b84b8bdb (raw.githubusercontent)

Decks: no deck files of their own. Each case is built from tests/test_axial_momentum_flux_form.py + .yaml (helpers config/sw_config/
sw_state/initial_state/aam/interior, constants W_SB, U_PERT, W_UP, SW_OMEGA, SW_U0), taken from #365 head 0e240e15 for BOTH builds:
- f: yaml + conserve-axial-momentum MODE + coriolis {type: xyz, omega1: om}; explicit
- g: f + coriolis traditional: true, x1max = x1min + 8 m, implicit-scheme 9
- h: f + implicit-scheme 9, initial_state(..., w_up=W_UP)
- om: slow = W_SB, deck = 7e-5; all: interp_order 4
- W2: sw_config(MODE, m*SW_OMEGA), sw_state(blocks, m*SW_OMEGA, SW_U0/m), m = 1, 4
Run: runcfg.sh $PYINST $OUT/main_full full - ; runcfg.sh $PYINST $OUT/main_transport transport - ; cmpsplit.py $OUT main_full main_transport
(paths to venv/test file are hard-coded in runcfg.sh: VB, R, T must be edited).

At b84b8bdb (v3/final): "full" is renamed "exact" and conserve-axial-momentum-closure is refused. sw_config maps full->exact
itself, but drvsplit's 3-D path writes MODE directly, so use MODE=exact (or transport) with CLOSURE unset; main's old "full"
(no closure) is only reachable via the test hook MeshBlock.axial_closure(False). The test file at b84b8bdb keeps all helper names.
