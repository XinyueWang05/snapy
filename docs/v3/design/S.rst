V3 lane S: seams, coord, layout, exchange and boundaries
=========================================================

:Lane: S (cubed-sphere seams, coord, layout, exchange, bc)
:Owner: Xinyue Wang (backup owner, on the PM's assignment)
:Revision: 1.1 (from ``fe1612b1``): Tianhao's four points, re-cited to lane
           R rev 1.2. Item 3 went up first as ``48b2328d`` and is section 3.
:Pin: chengcli/snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``
:Lane R header: happysky19/snapy ``docs/v3/skeleton/R/flux_register.hpp`` at
                ``3490d7e1aee481684da998e3b9ba7ea12d697f84`` (branch
                ``v3-design/R``, R.rst rev 1.2, 118 lines; confirmed with
                ``git ls-remote``). Its names and signatures are used here
                unchanged. Rev 1 used ``543ff84f``.

Every ``file:line`` is at the pin and relative to ``src/`` unless it names
another tree. Lines of R's header are written ``flux_register.hpp:N``. A
claim I have not tested is marked **hypothesis**. A change in results beyond
round-off is marked **FOR CHENG**. Nothing in this document was built or run.

.. contents::
   :local:
   :depth: 1


1. Scope and pins
-----------------

**Scope.** Lane S owns ``src/coord/*``, ``src/layout/*`` and ``src/bc/*``,
the transport of every within-stage exchange (section 3), the panel-seam
geometry and the one rotation (S-1). S reads but does not own the exchange
call sites in ``hydro/hydro_forward.cpp``, ``hydro/hydro.cpp:344-377, 606-776``,
``scalar/scalar.cpp:100-181``, ``forcing/diffusion_cubed_sphere.cpp``,
``eos/equation_of_state.cpp:335-372`` and ``mesh/meshblock.cpp:611-1196``.

**Pins.**

- snapy ``9e5de2d4`` (checked with ``git rev-parse`` in a fresh clone).
- Lane R header and R.rst at ``3490d7e1`` (fetched read-only). R.rst lines
  are written ``R.rst:N``.
- #370 (open; species enthalpy and donor velocity across seams) is used only
  through its PR text. It appears as the pending row 17p.

**The brief's numbers, checked at the pin.**

.. list-table::
   :header-rows: 1

   * - number
     - brief
     - at the pin
     - command
   * - lane baseline
     - 8,472
     - 8,472, confirmed: coord 4,008 + layout 3,316 + bc 1,155 = 8,479,
       less ``coord/cylindrical.cpp_`` (7 lines, never compiled)
     - ``wc -l src/coord/* src/layout/* src/bc/*``
   * - ceiling
     - 6,777 (at least -20%)
     - 8,472 x 0.8 = 6,777.6
     - arithmetic

That the brief's baseline is exactly this file set is my reconstruction: the
numbers match, but I did not see the brief's own file list.

**Settled before this revision** (lane S thread):

- S-1: coord owns the single rotation and its data. Lane A owns the Omega
  and mass-flux consumers. The non-z axis is a declared parameter that refuses
  non-z. That FOR CHENG item sits in lane A.
- S-2: register posts are in global components after coord's rotation
  (``flux_register.hpp:8-9``). This replaces my earlier panel-frame position.
  Any rounding or cost problem goes to the R header review, not back here.
- S-3: diffusion gets a face-slab row and keeps its own exchange until
  phase 3.
- EF: solid cells are column breaks, never donor or receiver in
  ``fix_vapor``. They are excluded from theta, the hit census and the
  neighbour minimum. Solid faces are mask-zeroed for mass, species, energy
  and scalars, and normal momentum keeps the mirror-Riemann flux, posted as
  one named boundary increment.

**Not checked.** The PM's own wording of the four failing identities
(section 5 uses the 919936da inventory and its cross-checks). Commit
``9c568b92`` (the first item-3 table, never pushed). Lane X's and lane C's
drafts.


2. Concepts and current sites
-----------------------------

**Layouts.** ``LayoutImpl`` (``layout/layout.hpp``) has three subclasses:
slab (``layout/slab_layout.cpp``, ``pz == 1`` only, ``:12-13``), cubed
(``layout/cubed_layout.cpp``) and cubed sphere
(``layout/cubed_sphere_layout.cpp``). One exchange is serialize, launch and
finalize (``mesh/meshblock.cpp:664-685``). On a subdivided cubed-sphere panel
an interpolating exchange runs twice, intra-panel and then cross-panel
(``mesh/meshblock.cpp:651-658``).

**Two transports.** A same-process neighbour goes through boards guarded by
a mutex and a condition variable (``layout/layout.cpp:312-500, 895-940``). A
remote one goes through the process group (``layout/layout.cpp:556-653``,
``layout/process_group.cpp:183-196``). Hydro writes its own x1 relays on top
of both (``hydro/hydro.cpp:344-377, 727-776``;
``hydro/hydro_forward.cpp:882-946``).

**Four rotation paths** (S-1):

#. the exchange's spherical frame matrices: ``cs_velocity_transform_matrix``
   (``coord/cubed_sphere_utils.cpp:259-298``), applied at
   ``layout/cubed_sphere_layout.cpp:685-698, 823-836``, cached per layout
   (``:161-191``). Its angle comes from ``torch::atan2``
   (``coord/cubed_sphere_utils.cpp:189``);
#. Coriolis' own frame (``forcing/coriolis.cpp:128-142``, the same matrices);
#. the AAM rotation field (``coord/gnomonic_equiangle.cpp:22-35``), which goes
   through Cartesian;
#. diffusion's Cartesian route (``forcing/diffusion_cubed_sphere.cpp:193,
   371``, using ``coord/cubed_sphere_utils.cpp:194-243``).

The face rotations the solver uses, ``prim2local*_``/``flux2global*_``
(``coord/gnomonic_equiangle.cpp:365-484``), are a fifth, per-face use. They
stay as they are. "Global components" in R's sense (``flux_register.hpp:8-9``)
is their output: each panel's own covariant components
(``coord/gnomonic_equiangle.cpp:427-455``).

**Seam geometry.** At the pin each panel already evaluates seam arcs with one
function of the unordered endpoint pair
(``coord/gnomonic_equiangle.cpp:114-123``). Ghost interpolation is already
in delta form (``coord/gnomonic_equiangle.cpp:767-771``). Both were the
proposed fixes for H6 and H12 (section 5).

**Corners.** ``skip_corner`` defaults to true (``layout/layout.hpp:152``).
Corners are synthesized as 0.5·(edge + edge) on every layout
(``layout/layout.cpp:681-721``), which is wrong for periodic corners on slab
and cubed layouts (section 5, identity P).

**Wall predicates.** There are three questions under one word: physical and
not periodic (``hydro/hydro.cpp:296-301``), physical
(``mesh/meshblock_options.cpp:235``), and outflow or reflecting
(``bc/bc_func.cpp:231-250``).

**Periodic, twice.** The layout wraps on cubed layouts, and the bc functions
copy (``bc/bc_func.cpp:36-50``). Slab x1 periodic exists only in the bc
copy, because the slab layout has no x1 wrap (``layout/slab_layout.cpp:12-13``).

**Solids.** ``InternalBoundaryImpl`` marks solid cells
(``bc/internal_boundary.cpp:47-60``) and mirrors face states into a solid
neighbour, negating the normal velocity (``bc/internal_boundary.cpp:70-95``).
The Riemann solver then gives the mirror flux. No mask is applied to the
flux.


3. Target design
----------------

3.1 Data flow
~~~~~~~~~~~~~

::

   coord (geometry, PanelRotation, PanelGeometry, AxialRotationData)
     |  rotation, metric, seam arcs, solid-face mask
     v
   Exchange (S) ----- Channel: local board | process group -----> peers
     |  ghosts()      x1_rows()      seam_slab()     gather_x1()
     |     |              |               |               |
     |  rows 0-3,17,   rows 4-5,8,     rows 9, 16r     rows 6-7 (scan),
     |  26-27          10,13-16,19,    (reconcile)     28
     |                 22-25              |
     v                                    v
   solver sweeps -- post --> FluxRegister (R) -- replace(d,faces,peer,rule)
                                  |  enter(RegisterPhase::reconcile)
                                  |  set_prescribed(d,mask,value) (from S)
                                  |  apply_prescribed(d): boundary.prescribed
                                  v
                            total(d) -> divergence

S moves bytes and rotates. R decides what a seam face's value is and keeps
it. Each stage exchange is one row of the declared schedule
(``stage_schedule()``, ``exchange.hpp``), and lane X cuts segments at its
rows.

**One rotation.** ``PanelRotation`` (``coord_rotation.hpp``) is the only
code that changes a vector's basis between panels. It goes through the
global Cartesian frame (contravariant A → Cartesian → contravariant B),
which needs no ``atan2``. Under AVX2 the cross-check measured its matrices
as strip-independent (0 of 8,640) and found it agrees with the spherical
route to 6.2e-16. Both numbers are at 919936da. That nb = 4 then becomes
bitwise end to end is a **hypothesis**. The spherical-frame code is deleted
(section 9).

**FaceKind.** Each block face has one kind (``exchange.hpp``). The three
wall predicates become one lookup. ``periodic`` is a seam to self: the layout
wraps where it can, and slab x1 keeps the bc copy as the declared fallback.

**Corners.** ``CornerRule::diagonal`` on slab and cubed layouts, using the
diagonal neighbour that ``for_each_direction`` can already visit
(``layout/layout.cpp:189-202``). ``synthesized`` (today's average) only at
cubed-sphere panel corners, where no diagonal neighbour exists. ``x1_full``
for x1 rows.

3.2 What the schedule counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

3.3 The schedule
~~~~~~~~~~~~~~~~

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
     - **reconcile** (x1): R's ``replace(1, faces, peer, SeamRule::mean)``;
       at the pin, the x1 process-seam flux average. Rev 1.1 extends it to
       same-process seams (section 4)
     - ``_flux1`` slab, ``_face_pressure1``, ``bflux1``, ``pressure_base1``,
       tags ``0x7720``/``0x7721``
     - x1 below/above; at the pin process seams only (``:887-888``), in
       V3 every x1 seam
     - ``RegisterPhase::reconcile`` (pin: step (2), after the x1 Riemann
       solve and sedimentation)
     - raw transport (S), then R's rule: the mean of both sides
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
   * - 16r
     - **reconcile** (x2/x3, not at the pin): R's ``replace(d, faces, peer,
       SeamRule::owner_copy)`` on panel seams, adopted in R rev 1.2
     - the owner's posted x2/x3 face flux slab, all rows; momentum rows
       rotated by coord (section 4)
     - cross-panel x2/x3 neighbours
     - ``RegisterPhase::reconcile``, after x2/x3 Riemann and covariance,
       before positivity (R's phases, ``flux_register.hpp:35-44``)
     - no interp; covariant(owner) to covariant(receiver) via one Cartesian
       rotation
     - none (one face)
     - +1
     - finalize
     - new; ``SeamRule::owner_copy`` (``flux_register.hpp:46, 77-78``)
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

3.4 Lane C's four groups (Tianhao), checked at the pin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

3.5 Segment boundaries and merge proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lane X's contract: one segment boundary per blocking round. With the most
options on that can hold together on the cubed sphere (rows 1-4, 6-9, 11-30
except 17p; ``pz > 1`` across processes; not stage 0; last stage), the count
is the sum of the ``rounds`` column:

2+2+2+1 (rows 1-4) + 1+1+1+1 (6-9) + 1 (11-12) + 4 (13-16) + 1 (17) + 4 (18)
+ 1 (19) + 1 (20-21) + 4 (22-25) + 1+1 (26-27) + 2 (28) + 1+1 (29-30) = **33**.

Rows 5 and 10 are spherical-polar only and do not add to this count. The
proposed row 16r adds one round on the cubed sphere (34). The earlier local
commit 9c568b92 counted per exchange site and reported 23 points, with nine
merges to 12. That commit could not be fetched from here, so I could not
reconcile its count with this one. Counted per site, this table has 26 points
with the same options. Both counts and both merge results are
**hypotheses**. Whether
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


4. Flux-register contract
-------------------------

Lane S posts no physical flux. It owns the transport of the ``reconcile``
phase (``RegisterPhase::reconcile``, ``flux_register.hpp:37``), the rotation
that phase needs, and the wall and solid-face mask. R owns the rules:
``replace(dim, faces, peer, rule)`` with ``SeamRule::mean`` or
``SeamRule::owner_copy`` (``flux_register.hpp:46, 77-78``), and
``set_prescribed``/``apply_prescribed`` (``:61-62, 79``). Citations of R's
header are at ``3490d7e1`` (R.rst rev 1.2).

**What changed in R since 543ff84f that affects S.**

- ``replace`` now takes the faces and the peer entry plus a ``SeamRule``,
  not a name and a seam tensor (``:77-78``). S supplies ``faces`` and
  ``peer``; R applies the rule to the total, the kept parts and the
  companions (``riemann.pstar``, ``pressure_base1``).
- ``SeamRule {mean, owner_copy}`` (``:46``): ``mean`` is row 9,
  ``owner_copy`` is S's row 16r, adopted by R (R.rst:28-29, 366-370).
- ``boundary.prescribed`` (``:23-24, 61-62, 79``) replaces S's
  ``bc.solid_face``. R calls it after ``reconcile`` and again after
  ``post_implicit``.
- ``bflux1`` is gone with the cell form (``:13``). Row 9's payload under R is
  therefore the total plus ``pressure_base1``, not ``bflux1``.
- The frame is the output of coord's ``flux2global<d>_``
  (``:8-9``; ``coord/coordinate.hpp:353-361``), as S-2 says.
- New phase ``after_combine`` (``:43``) and ``begin_stage(stage,
  harp::IntegratorWeight)`` (``:64``). Neither touches S's rows.
- The replaced amount is a monitor amount named ``reconcile.x1`` or
  ``reconcile.panel`` (R.rst:372-373). This answers S's rev 1 question 3.

**Convention (adopted, S-2).** A post is per unit face area and time, in
the output frame of ``flux2global<d>_``. On the cubed sphere that is each
panel's covariant basis (``coord/gnomonic_equiangle.cpp:427-484``), so at a
panel seam the two sides' momentum rows are in different bases. Their
mass, energy, species and scalar rows are not.

**Row 9 is the existing x1 seam average.** ``replace(1, faces, peer,
SeamRule::mean)`` is the flux average at ``hydro/hydro_forward.cpp:868-946``
(comment ``:868-881``; exchange and mean ``:882-946``). Both sides hold
``0.5*(mine + theirs)`` (``:937, 942``), which is bitwise equal on both sides
because IEEE addition commutes.

**Same-process x1 seams (R's K6, requirement 5).** At the pin one process
can hold several blocks that are adjacent in x1. ``blocks_per_process``
(``layout/layout.hpp:128``) places blocks per process, and a mixed layout
(``pz > 1`` with several blocks per process) runs since #254 (commit
``5eeb9b6``; ``tests/test_pref_local_seam.cpp`` uses four blocks in one
process). Their shared x1 faces are computed independently by each block
from its own reconstruction. The average skips a same-process neighbour:
``if (above >= 0 && playout->is_local_block(above)) above = -1``
(``hydro/hydro_forward.cpp:887-888``; the comment at ``:880-881`` says so).
With no process group it does not run at all (``:882-883``). What makes the
two sides agree today is that their inputs agree: the stage-entry ghost
exchange (rows 1-3), the anchor relay through a board
(``layout/layout.cpp:850-893``) and the reference ghost rows sent to local
neighbours too (``hydro/hydro.cpp:716-776``). The code's own comment says
this makes the average "a no-op" (``hydro/hydro.cpp:722-726``). #254's test
bounds the 4-block-vs-1-block difference at 1e-12 after 200 steps (commit
message: rho 5.2e-15, p 1.2e-15, vx 5.4e-14). That compares against one
block, so it does not measure ``F_own - F_peer``.

So rev 1's rule (both sides keep the mean) **did not cover** same-process
seams: it described the pin's process seams only. Proposed rule:

- **Rule:** every x1 seam, process or same-process, is ``replace(1, faces,
  peer, SeamRule::mean)`` in phase ``reconcile``. S delivers the peer entry
  through ``Channel``: the board for a local peer, the process group for a
  remote one (``Exchange::seam_slab``). There is no owner; both sides keep
  the mean. One rule per shared x1 face, as requirement 5 asks. The
  ``has_process_group`` and ``is_local_block`` gates go.
- **Gate:** bitwise equality of the two sides after ``reconcile`` (``0.5*(a
  + b)`` is symmetric), on 2 and 4 blocks in one process and on a mixed
  layout. Before ``reconcile``, R's G-R10 records ``|F_own - F_peer|``.
- **Result change vs main:** where the two sides already agree bitwise, the
  mean returns the same bits (``0.5*(a + a) = a`` exactly, barring overflow),
  so nothing changes. Where they differ, the change is the size of that
  difference. **Hypothesis:** that is round-off, because the inputs are
  exchanged copies and only position-dependent kernels (``pow`` under AVX2,
  cross-check §3) can tell the two blocks apart. If G-R10 shows more than
  round-off, enabling the mean is a **FOR CHENG** result change.
- **Cost:** none for one block per process. For local blocks it adds one
  board round per stage, inside row 9's existing round.

**Two-sided entries for every shared face.** Declared in
``seam_register.hpp`` as ``SeamEntry``:

.. list-table::
   :header-rows: 1
   :widths: 16 18 30 36

   * - face
     - canonical owner
     - rule via ``replace``
     - gate
   * - x1 seam, process or same-process (row 9)
     - none: both keep the mean (``SeamOwner::both_mean``)
     - ``replace(1, faces, peer, SeamRule::mean)``, all rows, kept parts and
       companions
     - bitwise equal on both sides; G-R10 records the pre-``reconcile``
       difference
   * - x2/x3 edge inside one panel
     - both (``SeamOwner::self``)
     - no ``replace``. After rows 1-3 both blocks hold the same ghost bits,
       so each computes the same face flux
     - bitwise, checked in debug and CI (R's G-R2). **Hypothesis**: ``pow`` is
       position-dependent under AVX2 (492/5,462 in the cross-check), so the
       gate runs under AVX2 and the scalar setting. If it fails, the entry
       becomes ``owner_copy`` with a raw copy (bitwise by construction)
   * - cubed-sphere panel seam (row 16r)
     - the block with the smaller global rank (``SeamOwner::lower_rank``)
     - ``replace(d, faces, peer, SeamRule::owner_copy)``. S moves the owner's
       slab (``Exchange::seam_slab``). Mass, energy, species and scalar rows
       are copied raw. Momentum rows go through
       ``PanelRotation::panel_to_panel_rows_`` (owner covariant → Cartesian
       → receiver covariant)
     - non-momentum rows bitwise equal. Momentum rows rotated back to the
       owner's basis match to 1e-14 relative. Reason: two panels' covariant
       bases joined by one floating-point rotation (6.2e-16 between routes
       at 919936da, inventory I7b). **Hypothesis** until measured
   * - x1 split with a panel seam
     - row 9 first, then row 16r
     - the edge faces of a panel seam are x2/x3 faces, which row 9 never
       touches
     - as the two rows above

**What this buys.** Mass, energy, species and scalars telescope exactly
across every shared face. Today panel seams agree only to round-off
(inventory H15: mass drift -9.9e-16 over 20 steps at 919936da), and
same-process x1 seams to whatever G-R10 measures.

**Cost.** Row 16r is one more cross-panel face-slab round per stage (34
rounds with every option on, section 3.5).

**K7, the carry's momentum frame on panel seams** (R's question). The carry
adds donor velocity times withheld mass to post-rotation face rows
(``hydro/hydro_forward.cpp:1233-1236``). It runs in ``limit``, after
``reconcile``. On a panel seam the donor ghost's velocity is in the
receiver's basis after rows 1-3, so each side's carry is in its own face
basis. The two sides then differ by the rotation, and the energy part
differs until #370 syncs ``hspec`` (row 17p). **Hypothesis:** after #370
the momentum rows agree to the 16r tolerance. If R's G-R12 or a
post-``limit`` single-valuedness check fails, the fix is a second
``owner_copy`` replace in ``limit`` (+1 round). That is for lane X and EF.

**Walls and solid faces: boundary.prescribed.** S's ``bc.solid_face`` maps
onto R's ``boundary.prescribed``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - item
     - mapping
   * - name
     - ``boundary.prescribed`` (R's). S's ``bc.solid_face`` is retired
   * - who posts
     - R calls ``apply_prescribed(d)`` (``flux_register.hpp:79``). S makes no
       ``add`` call. R books the change as the named correction
       ``boundary.prescribed`` (R.rst:216-226)
   * - phase
     - ``reconcile``, after ``replace``; and again after ``post_implicit``
       (``flux_register.hpp:23-24, 37, 42``)
   * - when S supplies
     - ``set_prescribed(d, mask, value)`` (``:61-62``), redone when the solid
       field or the wall set changes (R.rst:742-744), not per stage
   * - mask
     - per direction ``d`` and per row: true where the row is prescribed on
       the face. A face is prescribed if it is a physical reflecting wall
       (``FaceKind::wall``) or either neighbour cell is solid
       (``InternalBoundaryImpl::forward``'s ``solidl``/``solidr``,
       ``bc/internal_boundary.cpp:74-76``). Rows set: mass, species, energy,
       scalars. Momentum rows: not set (below)
   * - value
     - zero on every set row
   * - normals
     - coord's face unit normal in the post frame (``face_normal(d)``,
       ``coord_rotation.hpp``). It is supplied with the mask but unused by
       the default rule, because no momentum row is prescribed. It is needed
       only if EF chooses to zero tangential momentum (below)
   * - what S also supplies
     - the cell mask: solid cells are column breaks, and are excluded from
       theta, the hit census and the neighbour minimum (EF)

**Tangential momentum at solid faces (proposal; EF decides).** Free-slip,
which is what the pin does. ``InternalBoundaryImpl::forward`` copies the
fluid side's face state into the solid side and negates only the normal
velocity component (``bc/internal_boundary.cpp:78-80, 84-91``), so the tangential
velocity is mirrored equal. Reflecting walls do the same with ghost cells
(``bc/bc_func.cpp:10-34``: only ``var[4 - dim]`` is negated). With symmetric
states the Riemann contact speed is zero. So the mass flux, and the
tangential momentum flux ``rho u* v_t``, vanish, and the momentum flux is
the pressure ``p*`` along the normal. **Hypothesis:** "zero" holds to
round-off, which is R's G-R9. Hence:

- momentum rows are **not** prescribed; they keep the mirror-Riemann flux,
  normal and tangential alike;
- zeroing the tangential covariant row would be wrong on the cubed sphere.
  On a non-orthogonal face (``g23 = cos theta``,
  ``coord/gnomonic_equiangle.cpp:336-362``) the pressure force ``p* n``
  has a component in both covariant rows (``flux2global2_``,
  ``:428-455``), and zeroing one row removes part of the pressure force;
- no-slip would need a wall stress. The inviscid solver has none, and
  diffusion owns viscous walls. That is out of scope for V3;
- caveat (**hypothesis**): on gnomonic x2/x3 faces the mirror negates the
  covariant ``IVY``/``IVZ`` component (``bc/internal_boundary.cpp:84-91``).
  ``prim2local2_`` mixes ``uu2`` into the local tangential velocity
  (``coord/gnomonic_equiangle.cpp:373-391``: ``IVZ = T32 uu2 + T33 uu3``),
  so the local tangential velocities on the two sides are not exactly
  equal there. The contact speed still vanishes by symmetry of the normal
  part. Whether the tangential flux is exactly zero there is for G-R9.

**Settled answers recorded here.**

- S-1: coord owns ``PanelRotation`` and ``AxialRotationData``
  (``coord_rotation.hpp``). Lane A reads them, and owns the Coriolis rate, the
  closure, ``omega_local`` and the m1 weights. ``RotationAxis`` refuses
  anything but z (``check_supported``). The FOR CHENG item about other axes
  sits in lane A.
- S-2: as the convention above.
- S-3: diffusion keeps its four-round seam exchange
  (``forcing/diffusion_cubed_sphere.cpp:132-143``, row 18) until phase 3. It
  then moves onto ``Exchange::seam_slab`` as a face-slab row.
- EF: solid cells are column breaks for ``fix_vapor``: never donor or
  receiver. They are excluded from theta, from the hit census
  (``hydro/hydro_forward.cpp:1195-1196``) and from the neighbour minimum. S
  supplies the cell mask with the face mask. The limiter's use of it is lane
  C's.

**Traceability to the register requirements** (names as in lane I's table):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - requirement
     - where lane S meets it
   * - R1 one owner, frame, area weighting
     - S owns no physical flux. Frame: coord's rotation, one implementation
       (``PanelRotation``). The seam slab is moved per unit area, as posted
   * - R2 named increments, declared order
     - ``boundary.prescribed`` (R's name, from S's mask), and the replaced
       amounts ``reconcile.x1`` and ``reconcile.panel``
   * - R3 stage weights
     - N/A: S posts no stage transfer
   * - R4 divergence reads the register
     - the divergence reads ``total(d)`` after ``reconcile``. Nothing in S
       rebuilds a flux
   * - R5 seams, single-valuedness, rotation in coord
     - one rule per shared face, the table above, same-process x1 seams
       included. Rotation only in ``PanelRotation``
   * - R6 no host reads in a segment
     - section 7
   * - R7 fixers and floors register what they added
     - the wall and solid-face zero as ``boundary.prescribed``, booked by R
   * - R8 FP32 dtype, capturable storage
     - section 6. Fixed seam buffers per (shape, dtype, device)
   * - R9 no added full-field copy
     - row 16r moves one face slab per seam, not a field
   * - R10 rebuilt, not in restart
     - the masks, rotations and schedule are rebuilt from the grid and the
       solid field


5. Invariants and gates
-----------------------

**The four failing identities**, cited as the PM reported them from the
919936da coord/layout/parallel/bc inventory and Xinyue's cross-check at
919936da. Not rerun here.

.. list-table::
   :header-rows: 1
   :widths: 14 30 28 28

   * - identity
     - reported at 919936da
     - at the pin, by reading
     - V3 gate
   * - H6 seam areas bitwise across panels
     - 48/384 edge-sides (24/192 seam faces) not bitwise, max rel 3.8e-14.
       Cause measured: term order on mixed x2/x3 edges. The x2f
       antisymmetry is a centre-face artefact, not a cause
     - the canonical endpoint-pair arc is in
       (``coord/gnomonic_equiangle.cpp:114-123``). That H6 now passes is a
       **hypothesis**
     - T2: seam arcs bitwise at N = 16, 48, 96
   * - H12 uniform state through the exchange
     - 96/23,040 ghost entries off by 1 ULP, density only, on nb = 1, 2 and
       CUDA. Cause measured: ``x*b1 + (1-x)*b0``
     - delta form in (``coord/gnomonic_equiangle.cpp:767-771``). Passing is a
       **hypothesis**
     - T3: uniform state bitwise through rows 1-3, CPU and CUDA
   * - H16 nb = 4 equals nb = 1 bitwise
     - 8 + 28 of 6,144 entries after 20 steps, ≤ 2.5e-16. Cause measured:
       ``torch::atan2`` is position-dependent under AVX2, and the difference
       is gone with ``ATEN_CPU_CAPABILITY=default``
     - still open: ``coord/cubed_sphere_utils.cpp:189`` feeds the exchange
       matrices
     - T1: nb = 1 vs 4 bitwise under AVX2, default and CUDA, after the
       Cartesian route replaces the spherical one (**hypothesis**)
   * - H18 CPU equals GPU
     - GPU = ``std::fma`` on all outputs. CPU = unfused, so 16.9-18.5% of
       outputs differ, max 2.8e-14
     - unchanged: nvcc contracts by default (``coord/coord_dispatch.cu:14-40``)
     - tolerance gate in the default build, bitwise in the CI configuration
       (``--fmad=false``, ``-ffp-contract=off``). Lane X owns the build flag

**Further invariants S gates.**

- **P (periodic corners).** Corner ghosts are periodic images on slab and
  cubed layouts. Today they are synthesized: 36 of 36 and 144 of 144 were
  wrong at 919936da, max 0.11. Gate: a ghost-image test after
  ``initialize`` and after one stage.
- **Same-process x1 seams.** With 2 and 4 blocks in one process and on a
  mixed layout, the two sides of every x1 seam are bitwise equal after
  ``reconcile`` (the mean is symmetric). R's G-R10 records ``|F_own -
  F_peer|`` before it. If that is beyond round-off, enabling the mean is a
  **FOR CHENG** change (section 10).
- **Seam single-valuedness.** After ``reconcile``, every shared face has the
  same non-momentum rows on both sides (bitwise). Momentum rows meet the
  section 4 tolerance.
- **Rotation involution.** The panel edge table maps every seam vertex to the
  same point from both panels (inventory I5: 0/24 failures, 3.6e-16 at
  919936da). Gate: a structural test of ``CS_FACE_EDGES``
  (``layout/cubed_sphere_layout.cpp:315-350``).
- **Transport-only rows are bitwise.** Rows that copy (4, 5, 8, 10, 13-16,
  19, 22-25 and merges M1, M2, M4-M7, M10) must give the same bits as at the
  pin. Gate: an nb = 1 vs nb = 2 run, and a pz = 1 vs pz = 2 run, under the
  scalar CPU setting.
- **Solid faces.** Zero mass, species, energy and scalar flux through every
  solid face. The ``boundary.prescribed`` amount equals the flux the mask
  removed (R's G-R9).


6. FP32 contract
----------------

- Exchanged fields travel in their storage dtype, and V3 adds no cast in a
  stage row. The only FP32-to-FP64 cast near an exchange is in
  ``initialize_local`` (``mesh/meshblock.cpp:448``), outside the stage.
- ``PanelRotation`` and ``PanelGeometry`` are built in FP64 and cast once
  per (dtype, device), as the exchange matrices are cached today
  (``layout/cubed_sphere_layout.cpp:161-191``). Building on the CPU and moving
  changes the CUDA bits relative to today's lazy build on the velocity's
  device (cross-check §3 at 919936da). That is a change at round-off.
- The reconcile slab is in the register's ``kFlux`` dtype
  (``flux_register.hpp:14-16``). The mean of row 9 and the raw copy of row 16r
  are exact in either precision. The momentum rotation is not, and its gate
  tolerance scales with the dtype's epsilon (**hypothesis**).
- The anchor scan (rows 6-7) stays in the reference's precision, FP64 under
  perturbation flux (``hydro/hydro.cpp:636-641``, which sizes the anchor
  with the fp32 gauge).


7. CUDA-graph contract
----------------------

Every row of section 3.3 is a segment boundary for lane X today. Their waits
are a process-group ``wait()`` (``layout/layout.cpp:723-733``), a board
condition variable (``layout/layout.cpp:379-416``) or a stream synchronize
(``layout/layout.cpp:757-760, 804-807``). None can be captured. S's
contract:

- **No host read inside a row.** The axial closure's ``item``
  (``mesh/meshblock.cpp:1017-1018``) and the fixer's ``.cpu()``
  (``:1058-1061``) move to X's single step-end read, as R's monitor does
  (``flux_register.hpp:21-22``).
- **Fixed buffers.** Each row's send and receive buffers are keyed by
  (row, shape, dtype, device), as ``exchange_buffer_cache`` is today
  (``layout/cubed_sphere_layout.cpp:571-574``). Row 16r's slab buffer is
  fixed per dim.
- **Declared rounds.** ``ExchangeRow::rounds`` is the number of segment
  boundaries the row costs. X may merge rows only as section 3.5 lists. A
  merge that is not bitwise (M3) is flagged in section 10.
- **Boards.** Same-process boards wait on a condition variable. Under
  capture they become device events in the ``Channel`` implementation
  (**hypothesis**: one event per row and peer is enough).


8. Cross-repository interface changes and merge order
-----------------------------------------------------

- **Python bindings.** ``cs_contra_to_sph_``/``cs_sph_to_contra_`` and
  ``cs_ab_to_lonlat`` stay public (``python/csrc/pycoord.cpp``, used by
  tests and output). The test-only ``populate_cs_l2g_vel``
  (``tests/test_coordinate.cpp:297-298``) goes with its test row.
- **commux** (Zoey). ``Channel`` sits on ``ProcessGroupContext``. Moving the
  process-group wrapper (``layout/process_group*``, ``layout/distributed*``,
  446 lines) into commux is a proposal, not a decision (section 9, move C).
- **Lane R.** ``reconcile_seams`` calls ``FluxRegister::replace(d, faces,
  peer, rule)``, and ``supply_prescribed`` calls
  ``FluxRegister::set_prescribed(d, mask, value)``, exactly as declared at
  ``3490d7e1``. R calls ``apply_prescribed``. A rename in R is a one-line
  change here.
- **Lane A.** Coriolis (``forcing/coriolis.cpp:128-142``) stops building
  frame matrices and calls ``PanelRotation``. A's rev 2 cites coord for the
  geometry.
- **Merge order.** (1) ``PanelRotation`` plus the Cartesian exchange route
  (T1). (2) FaceKind and corners (P, T4). (3) ``Exchange`` with the x1 relays
  moved out of hydro (bitwise gate). (4) Row 16r, the same-process x1 mean
  and ``boundary.prescribed``, after
  R's register lands. (5) The deletions of section 9.


9. Line budget
--------------

Baseline 8,472 (section 1). "Measured" means the lines exist at the pin and
have no caller, or are replaced as named. "Estimate" is a design estimate
for code that is not written.

.. list-table::
   :header-rows: 1
   :widths: 30 8 8 8 46

   * - file or group (pin)
     - now
     - after
     - delta
     - deletions named
   * - ``coord/cubed_sphere_utils_impl.h``
     - 83
     - 0
     - -83
     - the whole file is ``cs_lonlat_to_ab``, which has no caller (measured).
       Its include at ``coord/coord_dispatch.cpp:11`` goes too
   * - ``coord/coordgen.{hpp,cpp}``
     - 13
     - 0
     - -13
     - ``UniformCoord``, no caller. Include at ``coord/coordinate.hpp:19``
       (measured)
   * - ``coord/coordinate.{cpp,hpp}``
     - 1,297
     - 1,283
     - -14
     - ``find_cell_index`` (``.cpp:687-705``, ``.hpp:327``), the
       ``vec_from_cartesian`` default (``.hpp:329-332``; no caller, and the
       only override is in ``cylindrical.cpp_``), the coordgen include
       (measured, -24); ``face_normal(d)`` for ``boundary.prescribed``
       (+10, estimate)
   * - ``coord/coord_dispatch.{cpp,hpp,cu}``
     - 218
     - 132
     - -86
     - MPS kernels and registrations (``.cpp:31-38, 57-65, 94-104,
       120-122``), the ``cs_matrix_vec`` CPU and CUDA kernels and their
       dispatch (``.cpp:67-92, 112, 118``; ``.cu:42-65, 75-76``; ``.hpp:14``)
       (measured; the MPS part assumes lane X's MPS refusal)
   * - ``coord/cubed_sphere_utils.{cpp,hpp}``
     - 550
     - 432
     - -118
     - spherical frame route: ``cs_ab_to_xyz_tensors`` (``.cpp:146-184``),
       ``cs_ab_to_theta_phi`` (``:185-193``), ``cs_velocity_transform_matrix``
       (``:259-298``), ``cs_apply_velocity_transform_`` (``:300-334``), their
       declarations (``.hpp:206-215``). +15 for
       ``panel_to_panel_rows_`` (estimate)
   * - ``coord/gnomonic_equiangle.{cpp,hpp}``
     - 939
     - 959
     - +20
     - ``AxialRotationData`` is declared over the existing AAM members and
       gains the axis check (estimate)
   * - ``layout/cubed_sphere_layout.{cpp,hpp}``
     - 1,006
     - 816
     - -190
     - ``velocity_transform_key`` and ``_velocity_transform`` with its cache
       (``.cpp:132-142, 161-191``; ``.hpp:67-74``), the test-only
       ``populate_cs_l2g_vel`` (``.cpp:352-363``; ``.hpp:17-18``), the dead
       ``cs_clamp_inside`` (``.cpp:379-389`` and its commented call
       ``:489``), the dead ``_interpolate_to_local`` (``.cpp:866-891``;
       ``.hpp:63-65``), the intra-panel loops that repeat the base class
       (``.cpp:577-616, 756-776``), the two velocity switch blocks that
       become one call each (``.cpp:684-701, 822-839``) (measured, net of
       about +10 for the calls)
   * - ``layout/layout.{cpp,hpp}``
     - 1,338
     - 1,328
     - -10
     - board/remote fold into ``Channel`` (``.cpp:312-500, 895-940``, about
       -100, estimate). Adds ``Exchange::x1_rows``, which takes over
       ``hydro/hydro.cpp:344-377`` (R's D7; +30, estimate), ``seam_slab`` for
       rows 9 and 16r over board and process group (+25, estimate), the
       anchor scan (+5, estimate), and ``FaceKind``, ``CornerRule`` and the
       schedule table (+30, estimate). Net unchanged from rev 1
   * - ``layout/connectivity.hpp``
     - 234
     - 220
     - -14
     - ``is_edge_loc``, ``is_corner_loc`` and their comments, with no caller
       (measured)
   * - ``bc/rectify_solid.cpp``
     - 170
     - 112
     - -58
     - ``run_flip_dim1/2/3`` (``:17-107``) are one function three times. They
       become one, with a permute (measured span, estimated result)
   * - ``bc/bc_func.cpp``, ``bc/bc_dispatch.cpp``
     - 320
     - 307
     - -13
     - ``is_outflow``/``is_reflecting``/``is_custom_noop``
       (``bc_func.cpp:231-250``) become a ``FaceKind`` lookup. The MPS
       registration ``bc_dispatch.cpp:68`` goes
   * - ``bc/internal_boundary.{cpp,hpp}``
     - 225
     - 245
     - +20
     - ``PrescribedFaces::build`` and the per-row mask and value for
       ``set_prescribed`` (estimate)
   * - all other files
     - 2,079
     - 2,079
     - 0
     - unchanged
   * - **total within S**
     - **8,472**
     - **7,913**
     - **-559 (-6.6%)**
     -

The ceiling is not met within S. Line-by-line reading finds 559 lines that
S can cut by its own design. Moves that would bring it closer, each needing
its receiving owner's agreement:

.. list-table::
   :header-rows: 1
   :widths: 10 34 10 46

   * - move
     - what
     - lines
     - receiver and status
   * - A
     - YAML parsing of the options structs: ``coord/coordinate.cpp:62-240``,
       ``layout/layout.cpp:249-272``, ``bc/internal_boundary.cpp:14-36``
     - -226
     - lane X's one options reader ("Scheme struct + YAML" in the 919936da
       designs). Not agreed
   * - B
     - the spherical-polar centroid switch, ``coord/x1_centroid.{cpp,hpp}``
       (its users are hydro's x1 code, ``hydro/hydro_forward.cpp:653-667,
       949-1000``)
     - -283
     - the x1 reconstruction lane. Not agreed
   * - C
     - the process-group wrapper, ``layout/process_group*``,
       ``layout/distributed*``
     - -446
     - commux (Zoey). Not agreed

With all three moves: 7,913 - 955 = 6,958 (-17.9%), still 181 lines above
the ceiling. Reaching 6,777 needs either a deeper cut that I cannot justify
by reading (the AAM operators stay in coord per S-1), or a ruling on the
ceiling. **FOR CHENG**: the ceiling for lane S. The lines leaving hydro for
S's transport (``hydro/hydro.cpp:344-377, 727-776``,
``hydro/hydro_forward.cpp:882-946``: 149 lines, replaced by about 60 in
``layout.cpp``) are counted in the hydro lanes, not here: R's D7 (34) and
D5 (``:882-946``, 65), and C's 50 (R.rst:824-826).

**D7, ``hydro/hydro.cpp:344-377``: S takes it (yes, unconditionally).** It
is ``HydroImpl::_x1_ghost_rows`` (34 lines; declared ``hydro/hydro.hpp:402``),
the generic x1 ghost-row relay. It sends ``m`` rows (cells, or faces with
``faces = true``) over every x2/x3 column to the x1 neighbours: through a
board for a same-process block, through the process group otherwise
(``:353-367``). Its callers are schedule rows 4, 5, 10, 13-16, 19 and 22-25
(``hydro/hydro_forward.cpp:249, 491, 666, 953, 1063, 1490``). R lists it as
D7, "conditional on S": R deletes it only if S's transport takes it
(R.rst:42-43, 805, 822). It is pure transport with no physics, so S takes it
as ``Exchange::x1_rows`` (``exchange.hpp``), which also packs several fields
in one message (merges M4, M5). Nothing in it depends on X. X only places
the segment cuts at those rows, as for every row. Lines: -34 in hydro
(counted in R's D7), +30 in ``layout.cpp`` (estimate, inside the layout row
above). The S total above already includes that +30.


10. Migration
-------------

**Phase 1 (transport only, bitwise).** ``Exchange`` and ``Channel`` take
over the x1 relays and the schedule. Merges M1, M2, M4, M5, M6, M7 and M10
(section 3.5) are copies. Gate: bitwise at nb = 1, 2 and pz = 1, 2 under the
scalar CPU setting. M8 and M9 go in only if their bitwise **hypotheses**
hold.

**Phase 2 (rotation and corners, round-off).** The Cartesian route replaces
the spherical one (T1), and diagonal corners replace synthesized ones on slab
and cubed layouts (P). Both change bits: at round-off for T1, and by up to
0.11 for wrong periodic corners, which is a correction. **FOR CHENG**: the
periodic-corner correction changes results beyond round-off on decks with
periodic x2 and x3. Affected decks are listed in the PR.

**Phase 3 (register).** Row 9 becomes ``replace(1, faces, peer,
SeamRule::mean)`` under R, now on every x1 seam: the
``is_local_block``/``has_process_group`` gates
(``hydro/hydro_forward.cpp:882-888``) go, so same-process seams are averaged
too. Where both sides already agree bitwise, nothing changes. Otherwise the
change is the G-R10 difference: round-off is the **hypothesis**, and beyond
round-off it is **FOR CHENG**. Row 16r and ``boundary.prescribed`` (R calls
``apply_prescribed``; S supplies ``set_prescribed``) start. Mass, energy, species and scalars telescope exactly
across panel seams (a round-off change). Zeroing the solid-face rows changes
results by the mirror flux's leak, which is round-off if the mirror is exact
(**hypothesis**). Otherwise **FOR CHENG**. Diffusion moves onto a face-slab
row (S-3).

**Not bitwise, flagged.**

- **M3, the anchor scan.** Replacing the serial relay (rows 6-7,
  ``hydro/hydro.cpp:614-641, 704-713``) with one column gather and a prefix
  sum changes the summation order of the hydrostatic scan, so the reference
  pressure changes at round-off on every x1-split run. That is round-off.
  Anything beyond it is **FOR CHENG**. It stays a proposal for lane X.
- **The nine merges** (M1-M9) are a **hypothesis** for lane X. The earlier
  commit counted 23 → 12 blocking points per site. This table counts 33 → 17
  rounds per round (section 3.5).


11. Open questions to other lanes
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - lane, owner
     - question
   * - R, Tianhao
     - Settled by R rev 1.2: R calls ``apply_prescribed`` under the name
       ``boundary.prescribed``; panel seams use ``SeamRule::owner_copy``; the
       replaced amounts are ``reconcile.x1``/``reconcile.panel``. Open: (1)
       do you accept ``SeamRule::mean`` on same-process x1 seams (K6, section
       4), with G-R10 deciding whether it is FOR CHENG? (2) ``set_prescribed``
       takes a per-row mask: confirm that its ``mask`` may carry a row
       dimension, so the momentum rows stay unprescribed. (3) K7: accept the
       section 4 answer (consistent after #370; otherwise a second
       ``owner_copy`` in ``limit``). (4) D7: S takes it (section 9)
   * - EF, Zoey
     - Tangential momentum at solid faces: S proposes free-slip, which is the
       pin's mirror. No momentum row is prescribed, because zeroing a
       covariant tangential row would remove part of the pressure force on a
       non-orthogonal face (section 4). The final word is EF's. If EF wants
       the tangential flux zeroed, S supplies ``face_normal(d)`` so that R can
       zero only the tangential projection
   * - X
     - (1) Place segment cuts at the section 3.3 rows. Do you accept the
       merges as proposals with their bitwise labels? (2) Will you take the
       options YAML parsing (move A)? (3) Will you own the CI flags for H18?
   * - C, Tianhao
     - The theta exclusion of solid cells uses S's cell mask. Does the hit
       census (``hydro/hydro_forward.cpp:1195-1196``) take the same mask?
   * - A, Jiachen
     - Coriolis calls ``PanelRotation``. The non-z axis stays FOR CHENG in A
       (settled). Nothing open from S.
   * - commux, Zoey
     - Move C: would commux take the process-group wrapper?
   * - PM / Cheng
     - The lane S ceiling (section 9): 6,958 is reachable with three moves,
       6,777 is not by my reading. **FOR CHENG**.


12. Skeleton index
------------------

``docs/v3/skeleton/S/`` holds declarations and comments only, 216 lines
in all. It is not part of any build. It uses R's names at ``3490d7e1``
(``FluxRegister``, ``RegisterPhase::reconcile``, ``SeamRule``,
``replace(dim, faces, peer, rule)``, ``set_prescribed``,
``apply_prescribed``) and R's namespace ``snap``.

.. list-table::
   :header-rows: 1

   * - file
     - lines
     - declares
     - replaces (pin)
   * - ``coord_rotation.hpp``
     - 66
     - ``RotationAxis``, ``PanelRotation``, ``AxialRotationData``,
       ``PanelGeometry``; ``face_normal``
     - ``cs_velocity_transform_matrix`` and its callers; the layout's
       ``_velocity_transform``; Coriolis' frame
   * - ``exchange.hpp``
     - 79
     - ``FaceKind``, ``CornerRule``, ``Frame``, ``ExchangeRow``,
       ``stage_schedule``, ``Channel``, ``Exchange``
     - hydro's x1 relays (``hydro/hydro.cpp:344-377``, R's D7, as
       ``x1_rows``); the three wall predicates; the board/remote pair
   * - ``seam_register.hpp``
     - 71
     - ``SeamOwner``, ``SeamGate``, ``SeamEntry``, ``seam_entries``,
       ``reconcile_seams``, ``PrescribedFaces``, ``supply_prescribed``
     - ``hydro/hydro_forward.cpp:882-946`` (as transport, every x1 seam);
       the unmasked wall and solid-face flux
