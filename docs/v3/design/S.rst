==========================================================
Lane S: seams, coord, layout, exchange and boundaries (v3)
==========================================================

:Status: partial revision 1. Item 3 (the exchange schedule of one stage) goes
         first and on its own. Later sections follow the twelve-section order
         of the v3 design documents and are added in revision 1 proper.
:Pin: chengcli/snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``
:Scope: design and contract only. Nothing in this section was built, run or
        measured. Every claim cites ``file:line`` at the pin. A claim that a
        run would have to confirm is marked **hypothesis**.

All paths below are relative to ``src/`` at the pin.

1. Exchange schedule of one RK stage
====================================

1.1 What is counted
-------------------

One stage is ``MeshBlockImpl::forward`` (``mesh/meshblock.cpp:611-618``): the
stage-entry ghost sync, then ``advance_local`` (``mesh/meshblock.cpp:698-1007``),
which calls ``HydroImpl::forward`` (``hydro/hydro_forward.cpp:370``), the scalar
forward, the multi-stage averaging and the stage tail.

A row is one *exchange*: a point where a block reads data that another block
wrote in this stage (point-to-point message, same-process board, column gather
or global reduction). Physical boundary functions are block-local and are not
rows; they are named in the row they follow. The per-step ``dt`` reduction
(``mesh/meshblock.cpp:585-588``) is outside the stage and not a row.

Column meanings:

- **phase**: where in the stage the exchange sits (the step numbers are the
  code's own section comments).
- **interp/frame**: ``interp`` = cubed-sphere ghost interpolation
  (``layout/cubed_sphere_layout.cpp:818-820``); ``rot`` = the velocity is taken to
  the spherical frame on send and back on receive
  (``layout/cubed_sphere_layout.cpp:685-698``, ``:823-836``); ``raw`` = copied
  as is (``kScalar``, ``:699``, ``:837``, or a hand-written x1 relay).
- **corners**: ``skip_corner`` defaults to true (``layout/layout.hpp:152``) and
  ``for_each_direction`` then visits faces only (``layout/layout.cpp:189-202``).
  ``avg`` = corners synthesized as the mean of two edge strips
  (``layout/layout.cpp:681-721``) after an exchange that is neither half of a
  split sync (``layout/layout.cpp:736-741``, ``mesh/meshblock.cpp:657``).
  ``x1-full`` = x1 ghost rows sent over every x2/x3 column, ghosts included,
  so x1/x2 and x1/x3 corners come from the x1 neighbour
  (``hydro/hydro.cpp:344-377``).
- **rounds**: blocking rounds with the most options on (cubed sphere with a
  subdivided panel, ``pz > 1`` across processes). A two-phase cubed-sphere sync
  is two rounds (``mesh/meshblock.cpp:651-658``).
- **block/host**: where the wait is. ``finalize`` = ``work->wait()`` in
  ``LayoutImpl::finalize`` (``layout/layout.cpp:723-733``); ``recv`` = a
  blocking ``recv_from_block(...)->wait()``; ``host`` = a device-to-host read
  (``item``/``cpu``).

1.2 The table
-------------

.. list-table:: Within-stage exchanges in execution order (pin 9e5de2d4)
   :header-rows: 1
   :widths: 4 14 14 10 10 10 7 5 10 16

   * - row
     - name
     - variables
     - peers
     - phase
     - interp/frame
     - corners
     - rounds
     - block/host
     - file:line
   * - 0
     - implicit-radial pre-step ghost sync (stage 0 only)
     - ``hydro_u``, ``scalar_s``
     - all face neighbours
     - before stage 0 (only with ``implicit_radial`` diffusion)
     - interp, rot (``hydro_u``); interp, raw (``scalar_s``)
     - avg
     - 4
     - finalize
     - ``mesh/meshblock.cpp:714-718`` -> ``:1111-1128``
   * - 1
     - stage-entry conserved ghost sync
     - ``hydro_u``
     - all face neighbours (x2, x3, and x1 when the layout has 27 buffers,
       ``layout/layout.cpp:123-127``)
     - stage entry
     - interp, rot (``kConserved``)
     - avg; x1 corners then refreshed by the x1 bfuncs
       (``mesh/meshblock.cpp:1130-1196``)
     - 2
     - finalize
     - ``mesh/meshblock.cpp:616``, ``:1117-1121``
   * - 2
     - stage-entry scalar ghost sync
     - ``scalar_s``
     - as row 1
     - stage entry
     - interp, raw (``kScalar``)
     - avg
     - 2
     - finalize
     - ``mesh/meshblock.cpp:1123-1128``
   * - 3
     - conversion-order-4 primitive ghost sync
     - ``hydro_w`` (converted)
     - as row 1
     - step (1) primitives
     - interp, rot (``kPrimitive``)
     - avg
     - 2
     - finalize
     - ``hydro/hydro_forward.cpp:489-490``
   * - 4
     - conversion-order-4 x1 ghost rows
     - ``hydro_w``, ``ng`` rows, tag ``0x772a``
     - x1 below/above
     - step (1) primitives
     - raw
     - x1-full; then bfuncs on every ghost layer (``:492-503``)
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:491``; ``hydro/hydro.cpp:344-377``
   * - 5
     - centroid plain-mean x1 seam rows
     - ``wx1`` (plain means), ``il`` rows, tag ``0x7724``
     - x1 below/above
     - step (2) x1 sweep, before the reference
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:653-667``
   * - 6
     - x1 reference anchor, receive
     - running seam-face pressure (1, 2 or 3 channels), tag ``0x7715``
     - x1 above only
     - step (2) inside ``_hydro_ref_x1``
     - raw
     - none (one face)
     - 1, serial over the column
     - recv; same-process board waits up to 5 min (``layout/layout.cpp:850-876``)
     - ``hydro/hydro.cpp:622-641``
   * - 7
     - x1 reference anchor, send
     - bottom-face pressure (+ split parts, + gauge in fp32)
     - x1 below only
     - step (2) inside ``_hydro_ref_x1``
     - raw
     - none
     - 1
     - send ``wait()`` for a remote peer (``layout/layout.cpp:878-882``)
     - ``hydro/hydro.cpp:704-713``
   * - 8
     - reference ghost rows
     - ``pref``, ``dref`` (+ ``cell0``, ``cell_delta`` when paired), tags
       ``0x7717``/``0x7718``
     - x1 below/above
     - step (2) inside ``_hydro_ref_x1``
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro.cpp:727-776``
   * - 9
     - x1 seam flux averaging
     - ``_flux1`` slab, ``_face_pressure1``, ``bflux1``, ``pressure_base1``,
       tags ``0x7720``/``0x7721``
     - x1 below/above, process seams only
     - step (2) after the x1 Riemann solve and sedimentation
     - raw, then mean of both sides
     - none (one face)
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:882-946``
   * - 10
     - centroid face-pressure seam rows
     - ``_face_pressure1``, ``min(2, il)`` faces, tag ``0x7722``
     - x1 below/above
     - step (2) end of the x1 sweep
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:949-955``
   * - 11
     - x2 face-state seam sync
     - ``hydro_wl:+``, ``hydro_wr:-`` (directional keys,
       ``layout/cubed_sphere_layout.cpp:81-90``)
     - cross-panel x2 neighbours
     - step (3.A)/(3.C)
     - no interp, rot (``kPrimitive``); strip widened over x1 ghosts
       (``layout/cubed_sphere_layout.cpp:92-112``)
     - none (``cross_panel_only``, ``layout/layout.cpp:736-741``)
     - 1 (shared with row 12)
     - finalize
     - ``hydro/hydro_forward.cpp:1005-1019``, ``:1033-1048``
   * - 12
     - x3 face-state seam sync
     - ``hydro_wl:+``, ``hydro_wr:-``
     - cross-panel x3 neighbours
     - step (3.B)/(3.C)
     - as row 11
     - none
     - shared with row 11
     - finalize
     - ``hydro/hydro_forward.cpp:1021-1031``, ``:1033-1048``
   * - 13
     - x2 covariance face states, left
     - ``wlr2[ILT]`` 1 row, tag ``0x7730``
     - x1 below/above
     - step (4.A)
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:1059-1065``, ``:1096``
   * - 14
     - x2 covariance face states, right
     - ``wlr2[IRT]``, tag ``0x7732``
     - x1 below/above
     - step (4.A)
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:1097``
   * - 15
     - x3 covariance face states, left
     - ``wlr3[ILT]``, tag ``0x7734``
     - x1 below/above
     - step (4.B)
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:1131``
   * - 16
     - x3 covariance face states, right
     - ``wlr3[IRT]``, tag ``0x7736``
     - x1 below/above
     - step (4.B)
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:1132``
   * - 17
     - hydro species theta sync
     - ``hydro_theta``
     - all face neighbours
     - step (4.C) positivity limiter
     - no interp, raw (``kScalar``)
     - avg; then bfuncs
     - 1
     - finalize
     - ``hydro/hydro_forward.cpp:1227``; ``hydro/flux_positivity.cpp:177-193``
   * - 17p
     - **pending #370**: species enthalpy and donor velocity
     - ``hspec`` ghosts synced like theta; donor velocity synced as a face
       state, then taken to the face's covariant basis
     - as row 17
     - step (4.C), next to row 17
     - raw (as theta), per the PR text
     - as row 17
     - +1 or 0 if packed with row 17
     - finalize
     - not at the pin: ``hydro/hydro_forward.cpp:1224-1225`` says ``hspec`` is
       not exchanged (#238). Row from PR #370's description only
   * - 18
     - cubed-sphere diffusion flux seam
     - ``diffusion_flux`` (12 channels: area-weighted flux + trace), streamed
       as four sides
     - all face neighbours
     - step (6) forcings, inside ``f.forward`` (``hydro/hydro_forward.cpp:1378-1384``)
     - no interp, raw (``kScalar``, ``phyid 12``); Cartesian frame
       (``forcing/diffusion_cubed_sphere.cpp:190-195``)
     - avg
     - 4
     - finalize
     - ``forcing/diffusion_cubed_sphere.cpp:132-143``
   * - 19
     - potential-energy seam row
     - ``drho`` (x1 density change), 1 row, tag ``0x7726``
     - x1 below/above
     - step (6) face gravity work, radial-exact only
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:1483-1494``
   * - 20
     - scalar x2 face-state seam sync
     - ``scalar_wl:+``, ``scalar_wr:-``
     - cross-panel x2 neighbours
     - scalar forward (``mesh/meshblock.cpp:785``)
     - no interp, raw (``kScalar``)
     - none
     - 1 (shared with row 21)
     - finalize
     - ``scalar/scalar.cpp:109-122``, ``:136-147``
   * - 21
     - scalar x3 face-state seam sync
     - ``scalar_wl:+``, ``scalar_wr:-``
     - cross-panel x3 neighbours
     - scalar forward
     - as row 20
     - none
     - shared with row 20
     - finalize
     - ``scalar/scalar.cpp:124-132``, ``:136-147``
   * - 22
     - scalar x2 covariance rows, left
     - ``rl``, 1 row, tag ``0x7740``
     - x1 below/above
     - scalar forward, ``scalar_flux_covariance`` (``scalar/scalar.cpp:157``)
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:245-252``
   * - 23
     - scalar x2 covariance rows, right
     - ``rr``, tag ``0x7742``
     - x1 below/above
     - as row 22
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:253``
   * - 24
     - scalar x3 covariance rows, left
     - ``rl``, tag ``0x7744``
     - x1 below/above
     - as row 22
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:252``
   * - 25
     - scalar x3 covariance rows, right
     - ``rr``, tag ``0x7746``
     - x1 below/above
     - as row 22
     - raw
     - x1-full
     - 1
     - recv
     - ``hydro/hydro_forward.cpp:253``
   * - 26
     - scalar upper-bound theta sync
     - ``scalar_theta`` (upper bound)
     - all face neighbours
     - scalar forward, when ``upper_bound > 0``
     - no interp, raw
     - avg; then bfuncs
     - 1
     - finalize
     - ``scalar/scalar.cpp:178-181``, ``:205``
   * - 27
     - scalar theta sync
     - ``scalar_theta``
     - all face neighbours
     - scalar forward
     - no interp, raw
     - avg; then bfuncs
     - 1
     - finalize
     - ``scalar/scalar.cpp:178-181``, ``:239``
   * - 28
     - whole-column limiter gather
     - cloud (parentless) then vapor rows + dry total + volume, the whole x1
       column
     - every block of the x1 column (``pz`` pieces)
     - step (4) multi-stage averaging
     - raw
     - interior only
     - 1-2
     - all pieces waited (``layout/layout.cpp:792-802``); CUDA stream sync
       (``layout/layout.cpp:757-760``, ``:804-807``)
     - ``mesh/meshblock.cpp:908-909``; ``eos/equation_of_state.cpp:335-372``;
       ``layout/layout.cpp:748-818``
   * - 29
     - axial-momentum closure reduction
     - ``axial_closure_sums`` (2 numbers)
     - all ranks (allreduce SUM)
     - stage tail
     - n/a
     - n/a
     - 1
     - allreduce ``wait`` (``layout/process_group.cpp:184-195``); host read
       ``item`` (``mesh/meshblock.cpp:1017-1018``)
     - ``mesh/meshblock.cpp:983-990``
   * - 30
     - gravity-work fixer reduction (last stage only)
     - ``gravity_work_fixer_sums`` (incl. ``.cpu()``,
       ``mesh/meshblock.cpp:1058-1061``)
     - all ranks (allreduce SUM)
     - stage tail
     - n/a
     - n/a
     - 1
     - allreduce ``wait``; host copy
     - ``mesh/meshblock.cpp:997-1004``

Gates (when a row runs at all):

- row 0: stage 0 and ``implicit_radial`` diffusion (``mesh/meshblock.cpp:713-718``);
- rows 3-4: ``conversion_order == 4`` and ``nc1 > 1``
  (``hydro/hydro_forward.cpp:419``); row 4 also needs an x1 neighbour
  (``hydro/hydro.cpp:346-347``);
- rows 5, 10: ``SNAP_X1_CENTROID_EXACT`` on a spherical-polar grid
  (``hydro/hydro_forward.cpp:654-655``, ``:950-951``). Not on the cubed sphere;
- rows 6-8: gravity in x1 (``wb_x1``, ``hydro/hydro_forward.cpp:645-646``) and
  ``pz > 1`` non-periodic (``hydro/hydro.cpp:630-631``);
- row 9: ``pz > 1`` with a process group, process seams only
  (``hydro/hydro_forward.cpp:882-888``);
- rows 11-12, 18, 20-21: cubed-sphere layout
  (``hydro/hydro_forward.cpp:1013``, ``:1032``; ``scalar/scalar.cpp:111``);
- rows 13-16: ``flux_covariance()``; rows 22-25 also need scalars;
- row 17: ``limiter || cov2 || cov3`` and species (``hydro/hydro_forward.cpp:1165``);
- row 19: x1 gravity and ``radial_exact_work()`` (``hydro/hydro_forward.cpp:1471``);
- row 28: ``pz > 1`` (``eos/equation_of_state.cpp:340-341``), and only when
  the limiter runs (``mesh/meshblock.cpp:907-909``);
- row 29: ``conserve_axial_momentum_closure``; row 30: ``gravity_work_fixer``
  on the last stage.

Host reads in the stage that are not exchanges, but are blocking points for
lane X: the implicit solve's finiteness screen, ``item<double>()`` on every
stage with ``picorr`` (``implicit/implicit_hydro.cpp:476-487``), and the
same-process board waits behind rows 4-10, 13-16, 19 and 22-25
(``layout/layout.cpp:379-416``).

1.3 Lane C's four groups (Tianhao), checked at the pin
------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - lane C group, as cited
     - what is at the pin
     - rows
   * - ``hydro_forward.cpp:489-503`` (conversion-order-4 ghosts)
     - ``exchange`` of ``hydro_w`` at ``:490``, x1 rows ``0x772a`` at ``:491``,
       bfuncs at ``:492-503``. Matches.
     - 3, 4
   * - ``hydro.cpp:636, 712, 727-776`` (anchor relay, reference ghost rows)
     - ``take_x1_anchor`` at ``:636``, ``pass_x1_anchor`` at ``:712``,
       the ``pref``/``dref`` ghost-row exchange at ``:727-776``. Matches.
     - 6, 7, 8
   * - ``hydro.cpp:666, 953, 1059-1065`` (centroid and covariance seam rows)
     - ``hydro.cpp`` has 971 lines at the pin, so 1059 cannot be in it. The
       cited lines match ``hydro_forward.cpp``: plain-mean rows ``:666``,
       face-pressure rows ``:953``, ``x1_seam_rows`` ``:1059-1065`` (used
       at ``:1096-1097``, ``:1131-1132``). Read as ``hydro_forward.cpp``.
     - 5, 10, 13-16
   * - ``hydro_forward.cpp:1227``, ``flux_positivity.cpp:177-190``,
       ``scalar.cpp:179`` (theta sync)
     - hydro theta ``:1227``; the shared sync function
       ``flux_positivity_sync_theta_`` at ``flux_positivity.cpp:177-193``
       (exchange ``:184``, bfuncs ``:186-192``); scalar ``sync_theta`` at
       ``scalar.cpp:178-181``, called at ``:205`` and ``:239``. Matches.
     - 17, 26, 27

Rows not in either list: 0-2 (stage-entry syncs), 9 (seam flux averaging),
11-12 and 20-21 (face-state seam syncs), 18 (diffusion seam), 19
(potential-energy seam row), 22-25 (scalar covariance rows, a separate call
path at ``hydro_forward.cpp:245-253``), 28 (whole-column limiter gather),
29-30 (reductions).

1.4 Segment boundaries and merge proposals
------------------------------------------

Lane X's contract: one segment boundary per blocking round. With the most
options on that can hold together on the cubed sphere (rows 1-4, 6-9, 11-30
except 17p; ``pz > 1`` across processes; not stage 0; last stage), the count
is the sum of the ``rounds`` column:

2+2+2+1 (rows 1-4) + 1+1+1+1 (6-9) + 1 (11-12) + 4 (13-16) + 1 (17) + 4 (18)
+ 1 (19) + 1 (20-21) + 4 (22-25) + 1+1 (26-27) + 2 (28) + 1+1 (29-30) = **33**.

Rows 5 and 10 are spherical-polar only and do not add to this count. Whether
the diffusion seam (row 18), radial-exact work (row 19) and the covariance
rows really run together in one cubed-sphere deck is a **hypothesis**. I did
not check the option validation for every pair.

Merge proposals. All are **hypotheses**: none was built or run. "Bitwise"
means the receiving block holds the same bits as before. For a pure copy this
is a property of the code. A changed summation order is not bitwise.

.. list-table::
   :header-rows: 1
   :widths: 6 34 12 12 36

   * - id
     - proposal
     - rows
     - rounds saved
     - bitwise?
   * - M1
     - one stage-entry sync carrying ``hydro_u`` and ``scalar_s`` with a
       per-variable type (rot vs raw)
     - 1+2
     - 2
     - yes, if the interpolation and corner averaging see the same inputs per
       variable (pure per-channel ops, ``layout/layout.cpp:681-721``)
   * - M2
     - fold the conversion-order-4 x1 rows into the primitive sync as the
       ``dz`` direction with x1-full extent
     - 3+4
     - 1
     - yes for the copied rows. The layout's ``dz`` strips skip corners
       (``layout/layout.cpp:197-199``), so x1-full extent needs a new strip
       shape
   * - M3
     - the anchor relay becomes one column all-gather of each block's local
       hydrostatic drop plus a prefix sum
     - 6+7
     - 1, and latency drops from ``pz - 1`` serial hops to 1 collective
     - **no**: the prefix sum changes the summation order of the scan
       (``hydro/hydro.cpp:614-622``)
   * - M4
     - pack the four covariance face-state rows into one x1 message
     - 13-16
     - 3
     - yes (pure copies)
   * - M5
     - pack the four scalar covariance rows into one x1 message
     - 22-25
     - 3
     - yes (pure copies)
   * - M6
     - send the four diffusion sides in one 48-channel round, as before the
       streaming change (``forcing/diffusion_cubed_sphere.cpp:132-136``)
     - 18
     - 3
     - yes per the code comment ("four completed rounds reproduce the old
       48-channel round"). Costs 4x the buffer
   * - M7
     - one theta sync for hydro and scalar species, by running the scalar
       lower-bound theta before row 17
     - 17+27
     - 1
     - yes for the copies. The reorder is safe only if the scalar theta reads
       no hydro flux that row 17's scaling changes: the scalar's mass flux is
       ``flux*()[IDN]`` (``scalar/scalar.cpp:166-170``), which
       ``flux_positivity_scale_`` does not scale. That this holds for the
       carry (``:1233-1236``) is a **hypothesis**
   * - M8
     - gather cloud and vapor in one column gather and repair both
     - 28
     - 1
     - yes if the cloud repair does not feed the vapor repair's inputs. The
       clamp at ``eos/equation_of_state.cpp:378`` lies between them, so this
       is a **hypothesis**
   * - M9
     - one allreduce for rows 29 and 30 on the last stage
     - 29+30
     - 1
     - yes if the backend reduces element by element. That depends on the
       backend and is a **hypothesis**
   * - M10
     - pack row 17p into row 17 when #370 lands
     - 17+17p
     - 1 (vs. #370 as written)
     - yes (pure copies)

With M1-M9: 33 - (2+1+1+3+3+3+1+1+1) = **17** blocking rounds (**hypothesis**).
Of these merges, only M3 is not bitwise.
Also proposed, with no change to the count: start rows 11-12 (and 20-21)
before the x1 sweep so they overlap it. The x2/x3 face states are
reconstructed from ``w`` (``hydro/hydro_forward.cpp:1011``, ``:1023``). The x1
sweep subtracts the reference from ``w`` in place when ``wx1`` aliases it
(``:703``, ``:705``) and restores the saved copies (``:696``, ``:765-767``).
That an earlier x2/x3 reconstruction therefore reads the same bits is a
**hypothesis**.

1.5 The four failing identities (cited, not rerun)
--------------------------------------------------

These are cited as the PM reported them. The figures come from the 919936da
inventory, as carried into Linfeng's design table (handed-in
``snapy-919936da-whole-code-target-design-linfeng.md``, section (d)). They
are not rerun here.

1. **Seam areas not bitwise** (Jiheng H6): 48/384 seam faces differ by up to
   3.8e-14, and x2f antisymmetry is off by 8.3e-17. Each panel computes its
   seam arcs with ``acos`` of a near-1 argument. That this is the whole cause
   is a **hypothesis**. Lane S owns the fix: the owner panel computes the seam
   metric once per physical face.
2. **1-ULP exchange** (Jiheng H12): with uniform rho = 1.3, 96/23,040 ghost
   entries are off by 2.2e-16 on nb = 1 and 2 and on CUDA. Rows 1-3 interpolate
   (``layout/cubed_sphere_layout.cpp:818-820``). Cause, a **hypothesis**:
   the interpolation weights sum to 1 only to round-off.
3. **nb = 4 not bitwise** (Jiheng H16): 8 + 28 of 6,144 entries differ after
   20 steps, by at most 2.5e-16. Per Xinyue's cross-check, the cause is
   ``atan2`` under AVX2, and the difference is gone with
   ``ATEN_CPU_CAPABILITY=default``. The mechanism is a **hypothesis**.
4. **Periodic diffusion edge ghosts** (Zoey (b)): 504/864 edge and corner
   ghosts are not periodic after ``initialize()``, and x2/x3 momentum loses
   1.6e-7/2.1e-7 per rk3 step. The diffusion reads corner ghosts that the
   periodic fill does not set. That is the ``avg`` corner policy of rows 1-2
   versus a periodic layout. The link to the momentum loss is Zoey's reading.
