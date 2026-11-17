V3 lane S: seams, coord, layout, exchange and boundaries
=========================================================

:Lane: S (cubed-sphere seams, coord, layout, exchange, bc)
:Owner: Xinyue Wang (backup owner, on the PM's assignment)
:Revision: 1. Item 3, the exchange schedule of one stage, went up first as
           ``48b2328d`` and is now section 3.
:Pin: chengcli/snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``
:Lane R header: happysky19/snapy ``docs/v3/skeleton/R/flux_register.hpp`` at
                ``543ff84f4adb4b79c3f43fa7c2f0a5a6eb54fa48`` (provisional,
                118 lines). Its names and signatures are used here unchanged.

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
- Lane R header ``543ff84f`` (read through the GitHub contents API, read-only).
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
  (``flux_register.hpp:5-6``). This replaces my earlier panel-frame position.
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
stay as they are. "Global components" in R's sense (``flux_register.hpp:6``)
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
   solver sweeps -- post --> FluxRegister (R) -- replace(dim,name,seam)
                                  |  enter(RegisterPhase::reconcile)
                                  |  add(dim,"bc.solid_face",rows,-F)
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
     - **reconcile** (x1): R's ``replace(dim=1, name, seam)``; at the pin, the
       x1 process-seam flux average
     - ``_flux1`` slab, ``_face_pressure1``, ``bflux1``, ``pressure_base1``,
       tags ``0x7720``/``0x7721``
     - x1 below/above, process seams only
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
     - **reconcile** (x2/x3, proposed, not at the pin): R's
       ``replace(dim, name, seam)`` on panel seams
     - the owner's posted x2/x3 face flux slab, all rows; momentum rows
       rotated by coord (section 4)
     - cross-panel x2/x3 neighbours
     - ``RegisterPhase::reconcile``, after x2/x3 Riemann and covariance,
       before positivity (R's order, ``flux_register.hpp:25-28`` at
       ``543ff84f``)
     - no interp; covariant(owner) to covariant(receiver) via one Cartesian
       rotation
     - none (one face)
     - +1
     - finalize
     - new; R's header covers only the x1 seam (``flux_register.hpp:79-80``)
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
phase (``RegisterPhase::reconcile``, ``flux_register.hpp:45``), the rotation
that phase needs, and the solid-face mask. R owns the rule
(``flux_register.hpp:79-80``).

**Convention (adopted, S-2).** A post is per unit face area and time, in
global components after coord's rotation (``flux_register.hpp:5-6``). On
the cubed sphere "global" is each panel's covariant basis
(``coord/gnomonic_equiangle.cpp:427-455``). So at a panel seam the two
sides' momentum rows are in different bases. Their mass, energy, species
and scalar rows are not.

**The reconcile row is the existing x1 seam average.**
``replace(dim, name, seam)`` on x1 is the process-seam flux average at
``hydro/hydro_forward.cpp:868-946`` (comment ``:868-881``; exchange and mean
``:882-946``). R's header orders it as "x1 seam replace :882"
(``flux_register.hpp:25``). It is row 9 of the schedule, renamed
``reconcile.x1`` and kept in place: no new row. The rule is unchanged. Both
sides hold ``0.5*(mine + theirs)`` (``:937, 942``), which is bitwise equal on
both sides because IEEE addition commutes. The payload is the flux slab,
``_face_pressure1``, ``bflux1`` and ``pressure_base1`` (``:896-917``). R
calls these the total plus kept parts.

**Proposed two-sided entry** for x2/x3 block edges and panel seams (R cites
it in ``R.rst``). Declared in ``seam_register.hpp`` as ``SeamEntry``:

.. list-table::
   :header-rows: 1
   :widths: 16 18 30 36

   * - face
     - canonical owner
     - rule via ``replace``
     - gate
   * - x1 process seam (row 9)
     - both (``SeamOwner::both_mean``)
     - mean of both sides, all rows and kept parts (today's rule)
     - bitwise equal on both sides (IEEE addition commutes)
   * - x2/x3 edge inside one panel
     - both (``SeamOwner::self``)
     - no ``replace``. After rows 1-3 both blocks hold the same ghost bits,
       so each computes the same face flux
     - bitwise, checked in debug and CI. **Hypothesis**: the cross-check
       found ``pow`` position-dependent under AVX2 (492/5,462), so the gate
       runs under AVX2 and the scalar setting. If it fails, the entry becomes
       ``lower_rank`` with a raw copy (bitwise by construction)
   * - cubed-sphere panel seam (new row 16r)
     - the block with the smaller global rank (``SeamOwner::lower_rank``)
     - S moves the owner's slab (``Exchange::seam_slab``). Mass, energy,
       species and scalar rows are copied raw. Momentum rows go through
       ``PanelRotation::panel_to_panel_rows_`` (owner covariant → Cartesian
       → receiver covariant). The receiver calls ``replace(dim, name,
       seam)`` on the total and the kept parts
     - non-momentum rows bitwise equal across the seam. Momentum rows:
       rotating the receiver's value back to the owner's basis gives the
       owner's value to a declared relative tolerance of 1e-14. Reason: two
       panels' covariant bases joined by one floating-point rotation. The two
       routes differ by 6.2e-16 at 919936da (inventory I7b). The tolerance is
       a **hypothesis** until measured
   * - x1 split with a panel seam (both)
     - row 9 first, then row 16r
     - the edge faces of a panel seam are x2/x3 faces, which row 9 never
       touches
     - as the two rows above

**What this buys.** Mass, energy, species and scalars telescope exactly
across panel seams. Today each side computes its own seam flux from
transformed states, so the sums agree only to round-off (inventory H15:
mass drift -9.9e-16 over 20 steps at 919936da). This is a result change at
round-off, not beyond it. The monitor gets the replaced amount (owner minus
own) through R's ``RegisterMonitor``, so the size of the old disagreement
stays visible.

**Cost.** Row 16r is one more cross-panel face-slab round per stage (34
rounds with every option on, section 3.5). It cannot share rows 11-12,
because the fluxes do not exist yet when the face states move. Whether it
can share the theta sync (row 17) is a lane X question: theta reads the
reconciled flux, so it cannot share it today.

**Solid faces.** S's coord/bc supplies ``SolidFaceMask``
(``seam_register.hpp``): a face is solid where either neighbour cell is solid.
The zero is posted as one named boundary increment, ``bc.solid_face``,
through ``FluxRegister::add(dim, name, rows, increment)``
(``flux_register.hpp:74-75``). The increment is ``-F`` on the zeroed rows
(mass, species, energy, scalars), so the monitor sees what the mask removed.
Normal momentum keeps the mirror-Riemann flux
(``bc/internal_boundary.cpp:70-95``). It is not zeroed. The increment is
posted in phase ``sweep``, after the solver's ``post_slot``. Whether R or S
makes the call is Tianhao's (section 11).

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
     - ``bc.solid_face``, plus the replaced amounts that R's monitor records
   * - R3 stage weights
     - N/A: S posts no stage transfer
   * - R4 divergence reads the register
     - the divergence reads ``total(d)`` after ``reconcile``. Nothing in S
       rebuilds a flux
   * - R5 seams, single-valuedness, rotation in coord
     - the table above. Rotation only in ``PanelRotation``
   * - R6 no host reads in a segment
     - section 7
   * - R7 fixers and floors register what they added
     - the solid-face zero as ``bc.solid_face``
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
  solid face. The ``bc.solid_face`` amount equals the flux the mask removed.


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
  (``flux_register.hpp:13-15``). The mean of row 9 and the raw copy of row 16r
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
- **Lane R.** ``reconcile_seams`` calls ``FluxRegister::replace`` and
  ``FluxRegister::add`` exactly as declared at ``543ff84f``. A rename in R's
  rev 1 is a one-line change here.
- **Lane A.** Coriolis (``forcing/coriolis.cpp:128-142``) stops building
  frame matrices and calls ``PanelRotation``. A's rev 2 cites coord for the
  geometry.
- **Merge order.** (1) ``PanelRotation`` plus the Cartesian exchange route
  (T1). (2) FaceKind and corners (P, T4). (3) ``Exchange`` with the x1 relays
  moved out of hydro (bitwise gate). (4) Row 16r and ``bc.solid_face``, after
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
     - 1,273
     - -24
     - ``find_cell_index`` (``.cpp:687-705``, ``.hpp:327``), the
       ``vec_from_cartesian`` default (``.hpp:329-332``; no caller, and the
       only override is in ``cylindrical.cpp_``), the coordgen include
       (measured)
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
       -100, estimate). Adds x1 transport and the anchor scan (+60,
       estimate), and ``FaceKind``, ``CornerRule`` and the schedule table
       (+30, estimate)
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
     - 240
     - +15
     - ``SolidFaceMask::build`` (estimate)
   * - all other files
     - 2,079
     - 2,079
     - 0
     - unchanged
   * - **total within S**
     - **8,472**
     - **7,898**
     - **-574 (-6.8%)**
     -

The ceiling is not met within S. Line-by-line reading finds 574 lines that
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

With all three moves: 7,898 - 955 = 6,943 (-18.0%), still 166 lines above
the ceiling. Reaching 6,777 needs either a deeper cut that I cannot justify
by reading (the AAM operators stay in coord per S-1), or a ruling on the
ceiling. **FOR CHENG**: the ceiling for lane S. The lines leaving hydro for
S's transport (``hydro/hydro.cpp:344-377, 727-776``,
``hydro/hydro_forward.cpp:882-946``: 149 lines, replaced by about 60 in
``layout.cpp``) are counted in the hydro lanes, not here.


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

**Phase 3 (register).** Row 9 becomes ``reconcile.x1`` under R. Row 16r and
``bc.solid_face`` start. Mass, energy, species and scalars telescope exactly
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
     - (1) Who makes the ``bc.solid_face`` ``add()`` call, R or S? Asked in
       the lane R thread. (2) Do you accept the section 4 entry (owner =
       lower rank, raw copy plus coord rotation, the 1e-14 momentum gate) for
       ``R.rst`` at 18:00? (3) Should the replaced amount (owner minus own) be
       a monitor amount under a declared name?
   * - EF, Zoey
     - Is the mask-zeroed versus mirror-Riemann split per row as written in
       section 4 (mass, species, energy, scalars zeroed; normal momentum
       mirrored)? Should tangential momentum be zeroed or mirrored?
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
     - The lane S ceiling (section 9): 6,943 is reachable with three moves,
       6,777 is not by my reading. **FOR CHENG**.


12. Skeleton index
------------------

``docs/v3/skeleton/S/`` holds declarations and comments only, 200 lines in
all. It is not part of any build. It uses R's names (``FluxRegister``,
``RegisterPhase::reconcile``, ``replace``, ``add``) and R's namespace
``snap``.

.. list-table::
   :header-rows: 1

   * - file
     - lines
     - declares
     - replaces (pin)
   * - ``coord_rotation.hpp``
     - 64
     - ``RotationAxis``, ``PanelRotation``, ``AxialRotationData``,
       ``PanelGeometry``
     - ``cs_velocity_transform_matrix`` and its callers; the layout's
       ``_velocity_transform``; Coriolis' frame
   * - ``exchange.hpp``
     - 77
     - ``FaceKind``, ``CornerRule``, ``Frame``, ``ExchangeRow``,
       ``stage_schedule``, ``Channel``, ``Exchange``
     - hydro's x1 relays; the three wall predicates; the board/remote pair
   * - ``seam_register.hpp``
     - 59
     - ``SeamOwner``, ``SeamGate``, ``SeamEntry``, ``seam_entries``,
       ``reconcile_seams``, ``SolidFaceMask``, ``post_solid_faces``
     - ``hydro/hydro_forward.cpp:882-946`` (as transport); the unmasked
       solid-face flux
