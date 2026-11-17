V3 lane I: vertical implicit solver and gravity work
=====================================================

:Lane: I (vertical implicit solver and gravity work)
:Owner: Xinyue Wang (alternate: Tianhao Le)
:Revision: 1 (round 1; written without reading the other lanes' drafts)
:Pin: chengcli/snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``

Every ``file:line`` below is at the pin unless it says otherwise. Lines
cited for #369 are at that PR's head
``c382f0da7b84dbc14219053758054f874669cc84`` (open, merge base
``cbf86830``). A claim I have not tested is marked **hypothesis**. A change in
results beyond round-off is marked **FOR CHENG** until Cheng rules. The
section titles and the
register requirements R1-R10 follow the brief as Cheng resent it in the lane
I thread.

**Rulings (Cheng's decision, 2026-11-17).** (a) ``gravity-work: cell`` is
removed. (b) The global gravity-work fixer is removed. Both are settled
design below, no longer FOR CHENG. (c) The per-column FP64 retry is still
open: it stays as proposed and keeps its FOR CHENG row in section 10.

.. contents::
   :local:
   :depth: 1


1. Scope and pins
-----------------

**Scope.** Lane I covers ``src/implicit/*`` (12 files),
``src/hydro/gravity_work.hpp``, ``src/hydro/gravity_work_radial.hpp`` and
``src/forcing/const_gravity.cpp``. The lane reads its callers but does not
own them: ``src/hydro/hydro_forward.cpp:1412-1690``,
``src/hydro/hydro.cpp:479-549`` and ``src/mesh/meshblock.cpp:1036-1109``.

**Pins.**

- snapy ``9e5de2d4`` (checked with ``git rev-parse``).
- #369 ``c382f0da`` (open). It is the only open PR this lane builds on.

**The brief's numbers, checked at the pin.**

.. list-table::
   :header-rows: 1

   * - number
     - brief
     - at the pin
     - command
   * - ``src`` lines
     - 42,677
     - 42,677, confirmed
     - ``git ls-files 'src/*.cpp' 'src/*.hpp' 'src/*.h' 'src/*.cu' 'src/*.cuh' | xargs cat | wc -l``
       (all tracked files give 43,629: that also counts ``CMakeLists.txt`` and ``*.cpp_``)
   * - lane baseline
     - 2,655
     - 2,655, confirmed (2,337 + 26 + 224 + 68)
     - ``wc -l src/implicit/* src/hydro/gravity_work.hpp src/hydro/gravity_work_radial.hpp src/forcing/const_gravity.cpp``
   * - ``.item<>``
     - 103
     - 103 lines (105 occurrences), confirmed
     - ``grep -E '\.item<' <C++ files above> | wc -l``
   * - hard FP64 dtype
     - 135
     - 135 only with ``torch::kFloat64|torch::kDouble|at::kDouble|kFloat64``. ``kFloat64`` alone gives 134. ``kFloat64|kDouble`` gives 136, because it also counts ``c10::kDouble`` at ``src/coord/coordinate.hpp:405``.
     - ``grep -E`` with the pattern on the left
   * - ``set_``/``set_data``
     - 41
     - 41 only with ``\.set_|set_data``. That pattern counts 3 comment lines in ``src/utils/tensor_cast_cache.hpp`` and misses ``p->set_(v)``. ``(\.|->)set_(data)?\(`` gives 39 real calls.
     - ``grep -E`` with the pattern on the left

In the lane's own files: 7 ``.item<>`` lines (one of them commented out, at
``implicit_hydro.cpp:205``), 11 FP64 literals and 2 ``set_data`` calls
(``implicit_hydro.cpp:157,183``).

**Not checked.**

- ``gh`` has no valid credentials for the API here. I read the PRs through
  the public REST API (bodies) and their squash commits on main, and #369
  through its fetched head.
- The Slack threads 1794169013.322299 (gravity-work booking) and
  1794212148.557499 (amplification analysis) and the earlier implicit
  inventory were not readable. I could not reproduce the inventory's "191
  copy lines" in the partial/full Jacobian pair. My own count:
  ``vic_assemble_full_impl.h`` has 118 lines, and 68 of them also appear in
  ``vic_assemble_partial_impl.h`` (206 lines). The assembly drivers add
  ``implicit_dispatch.cpp:24-137`` (114 lines) and ``.cu:32-143`` (112).


2. Concepts and current sites
-----------------------------

**VIC.** The vertically implicit correction solves the x1 column linearised
about the stage state. Its 3x3 (partial: rho, m1, E) or 5x5 (full) block
tridiagonal system is ``(I/dt + J) dq = du/dt``. J is built from Roe ``|A|``
and the flux Jacobians.

**Gravity work G.** G is the energy the x1 mass transfer gains or loses in
the potential :math:`\Phi=-g_1 x_1`. It has two parts: the face work W and,
where the grid has an x1 measure, the corrected-PE work K (#297, #345).
Section 4 defines it.

**Wall and solid rows.** These are the column ends: reflecting walls and
solid surfaces. **Well-balanced reference.** This is the hydrostatic part
taken out of the explicit RHS.

Current sites:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - concept
     - sites at the pin
   * - stage call, stage-weighted operator dt
     - ``hydro_forward.cpp:1554-1616`` (dt weight ``:1568-1583``, rk with 3 or more stages only)
   * - FP32/FP64 boundary, whole-block FP64 retry
     - ``hydro.cpp:502-543``
   * - frame projection / de-projection
     - ``implicit_hydro.cpp:254-259`` / ``:558-561``
   * - grid caches (three copies of one set of factors)
     - ``implicit_hydro.cpp:262-294`` (``_g_*``), ``implicit_hydro.hpp:127-133`` (``_rx_*``), ``hydro.hpp:434`` (``_pe_work``)
   * - matrix choice, a bit test of ``scheme``
     - ``implicit_hydro.hpp:36-42``, ``implicit_hydro.cpp:97-111, 427, 489``
   * - face-work flag bits in ``dir``
     - ``implicit_dispatch.hpp:27-30``, ``implicit_hydro.cpp:330-337``
   * - Jacobian assembly, partial / full
     - ``vic_assemble_partial_impl.h:42-200`` / ``vic_assemble_full_impl.h:17-112``
   * - gravity-work rows of the energy row
     - ``vic_gravity_work_impl.h:229-254``; corrected-PE rows ``implicit_hydro.cpp:365-425``; Woodbury wall third points ``:436-472``
   * - solve, back-substitution, redistribution
     - ``forward_sweep_impl.h:29-132``; ``vic_redistribute_impl.h:30-225``
   * - post-solve gravity-work remainders
     - projection/clamp ``implicit_hydro.cpp:522-556``; corrected-PE ``:569-594``
   * - host reads in the stage
     - ``implicit_hydro.cpp:477, 479, 485, 487``; ``implicit_hydro.hpp:104``; ``implicit_dispatch.cpp:255``; ``hydro.cpp:530``
   * - device failure latch
     - ``implicit_hydro.cpp:599-615``; read in ``check_redo`` at ``meshblock.cpp:1549-1551``
   * - explicit face work (two formulas) and corrected-PE work
     - ``hydro_forward.cpp:1447-1452`` (Cartesian, ``gravity_work.hpp:12-24``), ``:1453-1465`` (curved), ``:1470-1495`` (K)
   * - ``gravity-work: cell`` (non-Riemann flux split, Roe-diffusion work)
     - ``const_gravity.cpp:54-58``; ``hydro_forward.cpp:579-580, 846-849, 1426-1434``; ``vic_gravity_work_impl.h:243-247``
   * - global E+PE fixer
     - ``const_gravity.cpp:59-64`` (lane I); lane R: ``hydro.hpp:273-274, 296-314, 480``, ``hydro.cpp:126-135, 138-145, 287-293, 322-327``, ``hydro_forward.cpp:1416-1417, 1507-1533, 1584-1598, 1608, 1682-1693``; lane X: ``meshblock.hpp:217-222``, ``meshblock.cpp:724-730, 993-1004, 1036-1109, 1336-1337, 1477, 1502``, ``mesh.cpp:363, 384-402`` (section 10, table)
   * - readers of the implicit transfers
     - Coriolis/metric m1 ``hydro_forward.cpp:1637-1653`` (#355); AAM ``:1668-1669``; tracer donor transfer ``meshblock.cpp:786-806``
   * - solid and wall rows
     - ``implicit_dispatch.cpp:118-130, 216-235``; ``vic_assemble_full_impl.h:107-111``; ``implicit_hydro.cpp:405-420``; #369 surface rows ``_rs_*``
   * - well-balanced reference in the RHS
     - ``hydro_forward.cpp:837, 1380-1382, 1394-1400``; operator gravity ``implicit_hydro.cpp:326-329``

Default: ``gravity-work: face`` since #303 (``const_gravity.cpp:54``). No
``*.yaml`` file in the repository sets ``gravity-work``
(``grep -rln gravity-work --include=*.yaml examples tests`` is empty), so
every shipped deck runs ``face``.


3. Target design
----------------

**Decisions.**

#. **One gravity-work form, the default and the only choice.** It is the
   face work W plus the corrected-PE work K, with K = 0 where the grid has no
   measure (cylindrical, ``X1Measure::none``: plain face work and a warning,
   as now). Its operator row is generated from the booked work (section 4).
   ``gravity-work: cell`` is refused, and the global fixer is removed.
   **Settled: Cheng's decision, 2026-11-17** (rulings (a), (b)). Why:

   - it is what every shipped deck runs;
   - E+P closes by construction (#297, #345);
   - the fixer that ``cell`` needs has a documented tall-column instability
     (#284 "Limit", #285 "Limits") and adds a global reduction per step.
#. **``scheme`` is no longer a bit field.** Only 0, 1 and 9 are accepted
   (``implicit_hydro.cpp:97-111``), and bit 3 is the only bit with a
   meaning. V3 has ``enum class VicMatrix {partial = 3, full = 5}``.
   ``implicit-scheme: 1 | 9`` maps onto it in ``from_yaml``; 0 means no
   module. With one gravity form, the ``dir`` flag bits disappear and
   ``dir`` is the direction again, inside ``VicParams``.
#. **One Jacobian assembly.** The partial kernel copies {IDN, IVX+dir, IPR}
   out of the same 5x5 ``Am``, ``Ap`` and ``FluxJacobian`` objects the full
   kernel uses (``partial:80-137`` against ``full:52-88``). Its ``Phi``
   and ``Bnd`` are restrictions of the full ones (``partial:70-74``,
   ``full:39-46``). The block sums are entrywise (``partial:144-149``,
   ``full:95-100``). So V3 assembles 5x5 once and, for N = 3, restricts
   and folds the tangential momenta (``partial:159-199``) under
   ``if constexpr``.
#. **Wall and solid rows live in the column operator**, through one
   ``ColumnSegments`` per block built from #369's ``X1Wall`` depth: fluid
   ``[lo, hi)`` per column and the kind of floor. The assembly flags, the
   Woodbury cells ``k0``/``k1`` and the per-depth ``GravityWork`` rows all
   read it.
#. **No reference row** in the operator. A contract test replaces it
   (section 5).
#. **One kernel definition.** The plumbing that is duplicated across 9
   CPU and 9 CUDA stubs becomes one ``vic_stage<Launch>`` template.
   ``CpuColumns`` (also used for MPS through ``OnHost``) and
   ``CudaColumns`` instantiate it. Two stubs remain: ``vic_stage`` and
   ``donor_cap``.

**Types and ownership.**

.. list-table::
   :header-rows: 1

   * - type (skeleton)
     - owner
     - holds
   * - ``GravityWork`` (``gravity_work.hpp``)
     - ``HydroImpl``, one per block
     - grid-only coefficients of G; ``apply``, ``apply_seam``, ``rows``, ``potential``
   * - ``VerticalImplicitImpl`` (``vertical_implicit.hpp``)
     - ``HydroImpl`` (submodule ``icorr``)
     - options, ``ColumnSegments``, scratch, device ``status``, meters
   * - ``VicIncrement``
     - ``VerticalImplicitImpl``, posted to the register (lane R)
     - face transfers (mass, species, partial-matrix tangential momenta) and cell increments
   * - ``ColumnSegments``
     - ``VerticalImplicitImpl``, rebuilt per solid-mask version
     - fluid ``[lo, hi)``, floor kind
   * - ``VicParams``, ``VicColumn``, ``vic_stage`` (``vic_kernels.hpp``)
     - ``src/implicit`` dispatch
     - kernel scalars, unpacked per-column operands, the launcher template

**Data flow of one stage** (the solve's position in the stage does not
change):

.. code-block:: text

   explicit x1 fluxes --post--> [ x1 face register (lane R) ] <--post-- VicIncrement
          |                         |        |        |                 ^
          |                 read M_exp|  read M (final)|                 |
          v                         v        v        v                 |
   forcings (w-free     GravityWork.apply   m1 (A)  tracer      VerticalImplicit::forward
   Coriolis rows)       G[M_exp] -> du_E             donor        assemble(N; rows())
          \______________________ du ______________________/      solve (+Woodbury in kernel)
                                   \____________________________>  redistribute -> post
                                                                   remainder G[M_imp] - R dq -> du_E
                                                                   status bits (device)


4. Flux-register contract
-------------------------

**Posts.** The solve posts after de-projection and redistribution:

.. list-table::
   :header-rows: 1

   * - name (declared order)
     - kind
     - source at the pin
   * - ``vic.mass_dry``
     - x1 face transfer, after the availability clamp
     - ``vic_redistribute_impl.h:119-147`` (``MASS(IVX+(dir+1)%3)``)
   * - ``vic.mass_species[n]``
     - x1 face transfer, after the clamp
     - ``vic_redistribute_impl.h:156-175``
   * - ``vic.mass_total``
     - x1 face transfer, equal to dry plus species (checked)
     - ``MASS(IVX+(dir+2)%3)``, ``:144, 172``
   * - ``vic.mass_clamped``
     - x1 face: requested minus moved
     - ``implicit_hydro.cpp:527-528``
   * - ``vic.momentum_t[2]`` (partial matrix)
     - x1 face transfer at the donor velocity
     - ``vic_redistribute_impl.h:186-201``
   * - ``vic.cell``
     - cell increment: x1 momentum, energy without gravity work, full-matrix tangential momenta
     - ``vic_redistribute_impl.h:205-225``

The x1 momentum has a gravity source and wall pressure, so it cannot be a
telescoping face transfer. Energy stays a cell increment in round 1 because
the linearised wall flux leaves a column residual. The mass row shows this
residual: the redistribution removes it in proportion to mass
(``vic_redistribute_impl.h:65-71``). **Hypothesis:** the energy residual at
a sealed wall is round-off. If so, energy could become a face transfer too
(question to lane R).

**Frame and units.** The two cells across an x1 face share (j, k), so the
cell frame is the face frame. The solve posts after ``flux2global1_``
(``implicit_hydro.cpp:561``), in the explicit x1 flux's components. The
solve's transfers are A·dt-integrated (``vic_redistribute_impl.h:82-89``).
The explicit flux is per area per time (``hydro_forward.cpp:1641, 1646``).
The solve converts once, to whatever lane R declares.

**Reads.**

- ``GravityWork::apply`` reads the register's explicit total x1 mass before
  the solve, and the implicit part after it. This replaces the rebuilt
  ``vertical_mass_flux1`` at ``hydro_forward.cpp:1422-1425``.
- Coriolis/metric m1 (lane A) reads the final total.
- The tracer donor transfer reads ``vic.mass_dry``.
- ``mass_correction()`` and ``correction()`` stop being public.

**Corrects.** Only the energy row: the gravity-work remainder after the
solve. It is the one place a booked energy depends on the registered
transfer.

**G as one function of the registered x1 mass.** With :math:`M_f` the x1
mass through face :math:`f` (kg per stage) and cell :math:`i` between faces
:math:`l` and :math:`u`:

.. math::

   G_i[M] = \underbrace{\frac{g_1 (x_u - x_i)\,M_u - g_1 (x_l - x_i)\,M_l}{V_i}}_{W_i[M]}
          + \underbrace{g_1\sigma_i^2 s_i[\Delta\rho] + g_1(r_{c,i}-x_{1v,i})\Delta\rho_i}_{K_i[\Delta\rho]},
   \qquad \Delta\rho_i = \frac{M_l - M_u}{V_i}.

- W is ``gravity_work_from_face_mass`` (``gravity_work.hpp:12-24``).
- K is ``corrected_pe_work`` (``gravity_work_radial.hpp:190-205``); its
  second term exists only on gnomonic-equiangle (``:172-176``).
- The curved formula at ``hydro_forward.cpp:1453-1465`` equals W
  algebraically:
  :math:`[\Phi_i(M_u-M_l) - (\Phi_u M_u - \Phi_l M_l)]/V_i = [(\Phi_i-\Phi_u)M_u-(\Phi_i-\Phi_l)M_l]/V_i`.

G is linear in M.

**The energy row equals the booked work by construction.** The linearised
implicit transfer is
:math:`M^{\rm lin}_f(\delta q)=\Delta t A_f[\tfrac12(m_{f-}+m_{f+})-\tfrac12B^\rho_f(\delta q_{f+}-\delta q_{f-})]`
(#288 derivation). V3 assembles the row as
:math:`R=\sum_f (\partial G/\partial M_f)(\partial M^{\rm lin}_f/\partial\delta q)`.
The first factor comes from ``GravityWork::rows()``, the same tensors
``apply()`` uses. The second is the mass row of the assembled ``|A|`` and
flux Jacobians, which ``vic_gravity_work_impl.h`` already receives as
``am``/``ap``. The booked total for the stage is

.. math::

   G[M^{\rm exp}] + R\,\delta q + \big(G[M^{\rm imp}] - R\,\delta q\big)
   = G[M^{\rm exp}+M^{\rm imp}] = G[M^{\rm register}].

The last bracket is the one post-solve call. It vanishes wherever no clamp or
projection acted, because :math:`R\,\delta q = G[M^{\rm lin}(\delta q)]`
there. It generalises #345's identity for K to W and K together and
replaces ``implicit_hydro.cpp:522-556`` and ``:569-594``.

**How the four PRs broke the pairing.** Each time the booked work gained a
term, the row was edited separately:

- **#284.** Face work was booked outside the operator while the matrix had
  the cell work. A tall rest column blew up at Courant about 66 (#283).
- **#285.** It put the Cartesian face work in the row. Curved grids still
  "linearize cell work and replace it after the solve" (#288 body).
- **#288.** It added curved weights. The matrix energy defect fell from
  6.64e-4 to below 3e-16.
- **#297.** It added K to the booked work but not to the row, so a rest
  column grew at Courant 197-250 (#296). The tridiagonal part entered the
  row in f256ab5, and the wall third points only in #345 (wall mode from
  :math:`N\Delta t\approx 119`-149 before).

**Ordering constraints.**

#. The explicit x1 fluxes are posted, and the flux reconciliation of lane R
   is done, before ``G[M_exp]`` is booked into the RHS.
#. The w-free Coriolis rows enter ``du`` before the solve (#355 commit 6).
#. The solve posts before any reader of the final x1 transfer runs (G
   remainder, m1, tracer transfer).
#. The register's stage accumulation (R3) takes the posted stage
   transfer. The solve's own ``dt`` stays stage-weighted for 3 or more
   stages (``hydro_forward.cpp:1568-1583``), unchanged.
#. A failed column posts zero transfers and sets its status bit. The
   restore of ``du`` and ``w`` (``implicit_hydro.cpp:606-611``) happens
   before the post.

**Traceability to the ten register requirements.**

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - requirement
     - where lane I meets it
   * - R1 one owner, frame, area weighting
     - Lane I owns no face flux. The solve posts its transfers into the register (this section, "Posts") instead of keeping ``_mass_corr``. Frame: global components after ``flux2global1_`` (``implicit_hydro.cpp:561``). Area: the solve converts its A·dt-integrated transfers to the register's declared weighting at the post (question R-3).
   * - R2 named increments, declared order, total and parts
     - The six ``vic.*`` names in the declared order above. Parts are reportable: ``vic.mass_total`` = dry + species (a check), and the register total = explicit + ``vic``.
   * - R3 stage weights into the step transfer
     - Lane I posts the stage transfer. The register accumulates it with the integrator weights (``w2_s * prod w1_t``), the weights the fixer uses today (``hydro_forward.cpp:1682-1691``); that fixer code is deleted (section 10).
   * - R4 divergence and flux-derived terms read the register
     - G reads the register's x1 mass, explicit and implicit, and no longer rebuilds ``vertical_mass_flux1`` (``hydro_forward.cpp:1422-1425``). m1 and the tracer transfer read it too. The linearised fluxes inside the operator are its unknowns, not a rebuilt registered flux.
   * - R5 seams, reconciliation, single-valuedness, rotation in coord
     - N/A for the implicit post: the solve refuses an x1 split (``hydro.cpp:486-491``), so it posts no seam face. The explicit G at an x1 seam reads the reconciled register M; ``apply_seam`` takes the neighbour's Δρ from the existing ghost exchange (``hydro_forward.cpp:1484-1493``). Rotation: the solve calls coord's ``prim2local1_``/``flux2global1_`` and owns none.
   * - R6 device monitor, no host reads in a segment
     - Lane I contributes, all on device: ``status`` bits, the clamp residual, dry clamp and FP64-fallback meters, and ``GravityWork::potential`` for the E+P term. In-stage host reads go from 6 to 0 (section 7).
   * - R7 fixers and floors register what they added
     - The availability clamps, including the 4096-ulp species keep (``vic_redistribute_impl.h:151-155``), are floors: their change is posted as ``vic.mass_clamped``. The global gravity-work fixer is removed, so there is nothing of it to register.
   * - R8 FP32 dtype, graph-capturable storage
     - The solve dtype follows section 6. The posts are written into register storage in the register's dtype. Scratch is fixed per (shape, dtype, device) (section 7).
   * - R9 no added full-field copy per stage
     - V3 adds no copy. It removes ``w_in = w.clone()`` (``implicit_hydro.cpp:480``) and the ``rx_mass0`` clone (``:421``). ``_du0`` (one ``du``-sized copy, ``:212``) already exists and stays. Memory: the register slots for lane I are (ny + 4) face fields replacing ``_mass_corr`` (nvar cell fields), so the net is ≤ 0. Cost: not measured (**hypothesis**: ≤ the #358 spread).
   * - R10 rebuilt, not in restart
     - Met at the pin and kept. Restart writes the state variables and times only (``restart.cpp:69-100``), not the module buffers. V3's scratch is unregistered and ``ColumnSegments`` and ``GravityWork`` are rebuilt from the grid and the mask.


5. Invariants and gates
-----------------------

**Invariants.**

- I1 rows = apply: for random :math:`\delta q`, ``rows()·δq == apply(M_lin(δq))``
  to round-off, for every ``X1Measure`` and every solid depth.
- I2 booked = G[register M] per stage. With the post-solve remainder, the
  energy increment from gravity equals ``apply`` of the register's total
  x1 mass.
- I3 mass telescoping: ``vic.mass_total`` closes to zero at both walls of
  every fluid segment (the clamp-residual meter, ``implicit_hydro.cpp:495-518``).
- I4 rest is exact: zero explicit increment gives a correction that is
  zero bit for bit (``vic_assemble_partial_impl.h:40-41``,
  ``implicit_hydro.cpp:421-424``).
- I5 the Jacobian matches a finite difference of the split RHS. This holds
  because the reference terms do not depend on the state within a stage:
  ``face_gradient(pressure_base1)`` (``hydro_forward.cpp:837, 1394-1400``)
  and ``background_force1`` (``:1380-1382``). It is a **hypothesis** for any
  reference rebuilt from the stage state (#369 ``upward``, #361 FP32 gauge);
  that question goes to lane C.

**Gates (existing).**

- the #348 harness: 16 implicit decks x CPU/CUDA, ``hydro_u`` bytes;
- ``test_conservation_budget`` (E+P rows);
- the large-Courant tall column (#345);
- #369's "floor == wall" and rest-with-terrain rows;
- ``test_fp32_implicit.py`` and ``test_fp32_jupiter.py`` (fallback counts).

**Gates (new).** I1 and I2 as unit tests that go RED if one side is edited
alone. I4 on each reference mode for schemes 1 and 9. A sync-counter test:
no host read between assemble and post.


6. FP32 contract
----------------

- **Policy unchanged:** ``auto | float32 | float64``. FP32 is allowed only
  on Cartesian ideal-moist (``hydro.cpp:502-507``). The RHS master and the
  energy offset stay FP64 at the boundary (``:508-511``).
- **G in FP32.** The coefficients difference the geometry before the cast
  (``gravity_work.hpp:9-11``), for all grids. The curved explicit formula,
  which differences the potential after the cast, is gone.
- **Fallback, per column.** Today a failed FP32 block is re-solved whole in
  FP64 from ``rhs64`` (``hydro.cpp:529-535``). In V3, after the FP32 stage,
  the FP64 stage is launched once more without a condition. A column
  predicate makes every column whose status bit is clear return at once.
  The FP64 launch takes ``rhs64`` and the FP64 primitive ``wi64``, as today.
  I chose a second launch over an FP64 path inside the FP32 kernel because
  the K = 2 kernel is already register-bound (#358 runs it at 64 threads).
  The result change is in section 10(c).
- **FP64 literals.** The registered FP64 scratch (``implicit_hydro.cpp:126-136``)
  becomes unregistered scratch on ``w.options()``. ``GeoKey`` goes, and
  with it ``implicit_hydro.hpp:141``. That leaves one literal, the MPS-aware
  accumulation type (``implicit_hydro.cpp:24-26``), which the status kernel
  takes over. ``set_data`` (``:157, 183``) goes.


7. CUDA-graph contract
----------------------

- **No host read inside the solve segment.** Removed: ``implicit_hydro.cpp:477,
  479, 485, 487``, ``implicit_hydro.hpp:104`` and ``implicit_dispatch.cpp:255``
  (``donor_cap``: an always-run sweep on CPU, device-predicated on CUDA).
  ``hydro.cpp:530`` moves into the predicate. The 0-d ``int32`` status
  (``VicStatusBit``) is read once per step by ``check_redo``, which already
  reads the latch (``meshblock.cpp:1549-1551``). The failure log
  (``implicit_hydro.cpp:219-239``) runs on a redo only.
- **Fixed storage.** a, b, c, delta, z0, z1 and du0 are allocated once per
  (shape, dtype, device) and never ``set_data``-replaced.
  ``ColumnSegments`` and the ``GravityWork`` rows change only when the mask
  version or the grid key changes, which ends a captured segment.
- **Fixed launches.** Assemble, solve (Woodbury 2x2 in the kernel, in
  place of the ~12 small host launches at ``implicit_hydro.cpp:446-472``),
  redistribute, the predicated FP64 stage, the remainder. The count does
  not depend on the data.
- **Segment boundaries** are for lane X to choose (section 11). The lane
  needs none inside the solve.


8. Cross-repository interface changes and merge order
-----------------------------------------------------

**Python (snapy bindings).**

- ``ConstGravityOptions.gravity_work`` keeps working; ``"cell"`` is refused
  when the Hydro module is built.
- ``gravity_work_fixer`` is removed with the option: the binding
  ``python/csrc/pyforcing.cpp:29`` and the stubs ``python/snapy/forcing.pyi:61-70``
  go, and the ``"cell"`` docstring at ``forcing.pyi:54`` is edited. The C++
  setter call in ``test_lu_failure.cpp:228`` is removed with it.
- ``ImplicitOptions.scheme`` (``python/snapy/implicit.pyi:55-60``) stays an
  int spelling of ``VicMatrix``.
- The buffers ``hydro.icorr.{a,b,c,delta,du0,corr,mass_corr}`` leave
  ``named_buffers``. User stage forcings see the module's buffers
  (``meshblock.cpp:58-67``), and ``test_fp32_implicit.py:49`` reads
  ``buffer("a")``. The meters (``fp64_fallbacks``, ``clamp_residual``,
  ``dry_clamp_step``) stay.

**YAML.**

- ``forcing/const-gravity/gravity-work: cell`` is refused.
- ``gravity-work-fixer: false`` is still parsed by ``const_gravity.cpp``
  (lane I) with a deprecation warning, so decks that spell it keep running.
  ``true`` is refused. Inline test YAML uses ``false``
  (``test_lu_failure.cpp:109``).
- ``integration/implicit-scheme: 1|9`` is unchanged.

**pyharp** reads the ``integration`` block too (``implicit_hydro.cpp:33-38``).
Its keys do not change, so nothing changes for pyharp. kintera,
pyminichem and pydisort are not touched.

**Merge order.**

#. #369 (open; it keeps priority). V3 rebases on it.
#. Lane R's register.
#. The lane I PRs in the order of section 10, the ones that need no
   register first.
#. Lane A's m1 register read, after I-4.


9. Line budget
--------------

Baseline: 2,655 lines (section 1). The "after" numbers are estimates for the
design, not counts of written code.

.. list-table::
   :header-rows: 1

   * - file (pin)
     - now
     - after
     - delta
     - deletions named
   * - ``implicit/implicit_hydro.hpp``
     - 158
     - 95
     - -63
     - ``GeoKey`` (:137-152), ``_rx_*`` (:127-133), ``_g_*``, ``_solve_failed`` and ``solve_failed()`` (:101-116), the scheme bit test (:36-42)
   * - ``implicit/implicit_hydro.cpp``
     - 627
     - 420
     - -207
     - scheme-0 path (:179-188); grid caches (:262-294); host screens and ``reject`` (:219-252, 474-488); two remainders become one call (:522-556, 569-594); Woodbury 2x2 moves into the kernel (:446-472); ``set_data`` resizes (:139-168)
   * - ``implicit/implicit_dispatch.hpp``
     - 37
     - 25
     - -12
     - 9 stubs become 2; ``kVicFaceWork``, ``kVicDiffusiveCell``, ``kVicCartesianFaceWork``
   * - ``implicit/implicit_dispatch.cpp``
     - 313
     - 200
     - -113
     - per-stub operand unpacking (:24-235), replaced by one ``vic_stage<CpuColumns>``
   * - ``implicit/implicit_dispatch.cu``
     - 307
     - 90
     - -217
     - the CUDA copies of the same stubs (:32-289); the launcher, instantiation and registration remain
   * - ``vic_assemble_partial_impl.h`` + ``vic_assemble_full_impl.h``
     - 324
     - 220
     - -104
     - the second assembly (``full`` 118 lines) and the 3x3 extractions (``partial:91-93, 109-111, 124-126, 135-137``)
   * - ``implicit/vic_gravity_work_impl.h``
     - 50
     - 25
     - -25
     - the Cartesian 0.25 branch (:235-242) and the diffusive-cell branch (:243-247)
   * - ``implicit/forward_sweep_impl.h``
     - 138
     - 140
     - +2
     - (the column predicate is added)
   * - ``implicit/flux_decomposition_impl.h``
     - 130
     - 130
     - 0
     -
   * - ``implicit/vic_redistribute_impl.h``
     - 232
     - 232
     - 0
     - (named slots replace the ``MASS(IVX+(dir+k)%3)`` reuse)
   * - ``implicit/vic_solve_failure.h``
     - 21
     - 25
     - +4
     - (status bits are added)
   * - ``hydro/gravity_work.hpp`` + ``gravity_work_radial.hpp``
     - 250
     - 215
     - -35
     - two headers merge into one; free ``corrected_pe_work``/``_seam`` (:197-222) become ``GravityWork`` methods
   * - ``forcing/const_gravity.cpp``
     - 68
     - 50
     - -18
     - the ``cell`` and fixer options and checks (:54-64)
   * - **total**
     - **2,655**
     - **1,867**
     - **-788 (-29.7%)**
     - target: at least 20% (at most 2,124)

The deletions that rulings (a) and (b) cause outside lane I's files are
listed by owner in section 10 ("Outside-lane deletions"). They are counted
in lane R's and lane X's tables, not here. None of their lines is in the
2,655-line baseline (checked: no range falls in ``src/implicit``,
``gravity_work*.hpp`` or ``const_gravity.cpp``), so the lane I total above
needs no correction.

Revision 1 listed these sites only roughly, and two of its pointers were
wrong. ``meshblock.cpp:997-1003`` is really ``:993-1004``. "The explicit
face-work block ``hydro_forward.cpp:1412-1537`` becomes about 30 lines" also
overlapped the fixer and ``cell`` lines inside that block. Neither was in
lane I's -788. The remaining rewrite of that block around
``GravityWork::apply`` sits in lane R's file and is lane R's line count.


10. Migration
-------------

**PR sequence.** Each line gives: what changes; bitwise, rounding or beyond
round-off; and why.

#. **I-1 one assembly, ``VicMatrix``.** Bitwise (**hypothesis**): each entry
   of the partial blocks is the same scalar expression as before. Gate: the
   #348 harness.
#. **I-2 one kernel definition (``vic_stage<Launch>``, 2 stubs).** Bitwise
   (**hypothesis**): it changes the plumbing only, and the per-column bodies
   are untouched.
#. **I-3 ``GravityWork``: one cache, one W formula, rows from the
   coefficients, one remainder call.** Rounding (**hypothesis**). Three
   things move round-off: the Cartesian 0.25 branch becomes the general
   weights (equal values, other operation order); the curved explicit W
   becomes difference-first; and the remainder is computed in another order.
   It should reduce the large-origin cancellation #288 measured (3e-12 at
   r = 7e7 m).
#. **I-4 posting to the register** (after lane R). Bitwise if the register
   stores the same A·dt-integrated values that are read today; rounding if
   it converts units.
#. **I-5 device status and the Woodbury 2x2 in the kernel.** Rounding on
   CUDA (**hypothesis**: FMA contraction); bitwise on CPU.
#. **I-6 per-column FP64 retry.** See (c).
#. **I-7 removing ``gravity-work: cell`` and the global fixer** (settled,
   Cheng 2026-11-17). See (a) and (b). The lane I part is
   ``const_gravity.cpp`` and ``src/implicit``. The lane R and lane X parts
   are in their own PRs (table below), merged together with I-7 so that no
   commit leaves a dangling ``gravity_work_fixer()`` call.

**Options and paths removed.**

- ``gravity-work: cell``;
- ``gravity-work-fixer: true`` (``false`` is a deprecated no-op);
- ``kVicFaceWork``, ``kVicDiffusiveCell`` and ``kVicCartesianFaceWork``;
- the scheme-0 module (Python only: from YAML it already returns ``nullptr``,
  ``implicit_hydro.cpp:91``);
- the whole-block FP64 retry;
- the ``fixgrav=`` cycle meter (``meshblock.cpp:1336-1337``).

**What each removal changes in results.**

(a) **Refusing ``gravity-work: cell``.** **Settled: Cheng's decision,
2026-11-17.**

- Decks on ``face``, which is every shipped ``*.yaml``: bitwise
  (**hypothesis**, gated by #348). Every removed branch is gated on
  ``gw_cell`` or ``gravity_work() == "cell"``
  (``hydro_forward.cpp:579-580, 1427, 1430, 1470, 1505``;
  ``implicit_hydro.cpp:335-337``).
- A deck or script that selects ``cell``: refused at Hydro construction. If
  moved to ``face``, it changes **beyond round-off**:

  - energy, entropy drift and the conserved quantity (E+P instead of
    E+ρΦ). The #284 table, moving isentropic column at nz 64 / 128:
    entropy drift 1.97 / 0.473 with cell + fixer against 1.11 / 0.235 with
    face (measured in #284 on the face form of that time, before
    corrected PE; at the pin face also books K);
  - tall-column behaviour (#285: 4.49x top density with cell + fixer on a
    coarse 40H column, against a stable face column at the pin per the
    #345 large-Courant test).

  Periodic-x1 gravity decks lose the alternative the current warning names
  (``hydro.cpp:146-156``).
- Affected tests: all of these select ``cell`` and are re-targeted to
  ``face`` or deleted.

  - ``test_conservation_budget.cpp:178-180, 190`` (rows ``a_cell_s0/1/9``,
    ``c_cell_s9``);
  - ``test_flux_covariance_seams.py:59-60``;
  - ``test_implicit_gravity_tall_column.py:61-63, 202-212``
    (``cell-global-fixer`` arm);
  - ``test_implicit_stratified_solid.py:163, 170, 222, 252-299``;
  - ``test_mps_matches_cpu.py:203-204`` ("bubble cell work fixer");
  - ``test_gravity_work_fixer.py`` (whole file, 247 lines;
    ``tests/CMakeLists.txt:351, 828``);
  - ``run_straka_redo.cmake:1-5`` (``tests/CMakeLists.txt:613``), which
    runs straka on cell + fixer at CFL 1.5. Its outcome depends on the
    form ("a 1-ulp change flips the outcome", ``:5``), so it is re-recorded
    on face. The result is a **hypothesis** until run.

(b) **Removing the global gravity-work fixer.** **Settled: Cheng's
decision, 2026-11-17.**

- With ``face`` the fixer is already off: ``hydro.cpp:128`` forces it false
  and ``gravity_work_fixer()`` requires ``cell`` (``hydro.cpp:322-326``).
  Removing it is therefore **bitwise** for every face deck (**hypothesis**;
  the code it removes is not executed for face, ``meshblock.cpp:997-1004``,
  ``hydro_forward.cpp:1507, 1598, 1608, 1682``).
- For cell decks it goes away with (a), so there is no case where the fixer
  is removed and cell kept.
- Lost: the ``fixgrav=`` meter and the boundary-mass refusal
  (``meshblock.cpp:1097-1103``); the E+PE defect diagnostic becomes the E+P
  monitor (R6).
- The redo signal is not lost, because ``check_redo`` reads the solve latch
  itself (``meshblock.cpp:1549-1551``).
- Decks that spell ``gravity-work-fixer: false`` keep running bitwise,
  with a deprecation warning: ``test_lu_failure.cpp:109``,
  ``test_stencil_cache_small_block.py:54`` and
  ``test_implicit_gravity_tall_column.py:57``. The C++ setter call
  ``test_lu_failure.cpp:228`` is deleted, because the option goes.

**Outside-lane deletions (rulings (a), (b)).** These lines are counted in
lane R's or lane X's table, **not here**. They are exact inclusive ranges at
the pin, one entry per line, with no line in two rows (checked by a script
over the ranges). "Edited" lines change but are not deleted, and they are
not counted.

.. list-table::
   :header-rows: 1
   :widths: 10 50 8 32

   * - owner
     - range at the pin
     - lines
     - what
   * - R
     - ``src/hydro/hydro.hpp:273-274``
     - 2
     - ``gravity_work_fixer()`` declaration
   * - R
     - ``src/hydro/hydro.hpp:296-314``
     - 19
     - ``gravity_work_defect``, ``gravity_work_fix``, ``add_``/``commit_``/``drop_gravity_work_fix``, ``gravity_work_wall_mass``
   * - R
     - ``src/hydro/hydro.hpp:480``
     - 1
     - ``_gwfix_d, _gwfix_total, _gwfix_pending, _gwfix_wall``
   * - R
     - ``src/hydro/hydro.cpp:126-135, 138-145``
     - 18
     - fixer reset, the grav2/grav3 and periodic-x1 checks (``:136-137`` stay: the face warning reads ``nm``, ``wrap``)
   * - R
     - ``src/hydro/hydro.cpp:287-293``
     - 7
     - ``_gwfix_*`` ``register_buffer``
   * - R
     - ``src/hydro/hydro.cpp:322-327``
     - 6
     - ``HydroImpl::gravity_work_fixer()``
   * - R
     - ``src/hydro/hydro_forward.cpp:1416-1417, 1507-1533, 1584-1598, 1608, 1682-1693``
     - 57
     - the stage E+PE defect, wall-mass meter, implicit ``epe`` lambda and the stage-weight accumulation into ``_gwfix_d``
   * - R
     - ``src/hydro/hydro_forward.cpp:575-580, 846-849, 1426-1434, 1505-1506, 1534, 1536``
     - 23
     - ``cell``: ``bflux1``, ``gw_cell``, ``F - F^R`` split, the ``cell`` branch (``:1535`` stays as the only branch)
   * - **R total**
     -
     - **133**
     - hydro.hpp 22, hydro.cpp 31, hydro_forward.cpp 80
   * - X
     - ``src/mesh/meshblock.hpp:217-222``
     - 6
     - ``gravity_work_fixer_sums``, ``apply_gravity_work_fixer`` declarations, ``defer_gravity_work_fixer``
   * - X
     - ``src/mesh/meshblock.cpp:724-730``
     - 7
     - step-start reset and commit
   * - X
     - ``src/mesh/meshblock.cpp:993-1004``
     - 12
     - single-block step-end call (``:992`` stays: it is the ``apply_boundaries`` comment)
   * - X
     - ``src/mesh/meshblock.cpp:1036-1109``
     - 74
     - ``gravity_work_fixer_sums`` and ``apply_gravity_work_fixer``
   * - X
     - ``src/mesh/meshblock.cpp:1336-1337``
     - 2
     - ``fixgrav=`` print (if cycle printing is lane O's, it moves there; count it once)
   * - X
     - ``src/mesh/meshblock.cpp:1477, 1502``
     - 2
     - ``check_redo``: ``drop_`` / ``commit_gravity_work_fix``
   * - X
     - ``src/mesh/mesh.cpp:363, 384-402``
     - 20
     - ``defer_gravity_work_fixer``; multi-block sum, reduction and apply
   * - **X total**
     -
     - **123**
     - meshblock.hpp 6, meshblock.cpp 97, mesh.cpp 20
   * - not R or X
     - ``src/forcing/forcing.hpp:48, 56-58, 60-62``; ``python/csrc/pyforcing.cpp:29``; ``python/snapy/forcing.pyi:61-70``
     - 18
     - option report, the ``cell`` doc, the ``gravity_work_fixer`` option, its binding and stubs; owner to be named by Cheng (the forcing header and the Python bindings are outside every claimed lane)

Edited, not counted:

- ``hydro.cpp:123-125`` (accept ``face`` only), ``:146-147, 157`` (the
  warning no longer names ``cell``), ``:324, 331``;
- ``hydro.hpp:290-291`` (comment);
- ``hydro_forward.cpp:1412-1414`` (comment), ``:1470`` (``!gw_cell &&``
  goes), ``:1535`` (dedent);
- ``meshblock.hpp:225``, ``meshblock.cpp:1476`` (comments);
- ``forcing.hpp:47``, ``pyforcing.cpp:28`` (the line terminator);
- ``forcing.pyi:54`` (docstring).

``docs/derivations/curved-gravity-work-weight.md:429, 433, 521, 525`` name
the fixer and ``cell``; they are docs, not ``src``, and are edited with I-7.

(c) **Per-column FP64 retry.** **FOR CHENG** (still open with Cheng; kept
as proposed)

- A step where no FP32 column fails: **bitwise**. The predicated FP64
  launch does no column work.
- A step where a column fails: the failed columns get an FP64 solve from
  the same ``rhs64``/``wi64``, as today, so they are bitwise to today's
  retry for that column (**hypothesis**: the column solve does not depend on
  its neighbours, ``implicit_dispatch.cpp:137-160``). The other columns of
  the block keep their FP32 solution where today they are re-solved in
  FP64, so they change **beyond round-off**. The size of the change is the
  FP32-FP64 solve gap: #361 measured 5.2e-6 in w after 1000 steps (CPU),
  and ``test_fp32_implicit.py:52-54`` bounds it per solve at 2e-6
  (relative). The size per failing step is a **hypothesis**.
- The ``fp64_fallbacks`` meter changes meaning from blocks to columns.
- Affected tests: ``test_fp32_implicit.py:58-60``. It asserts
  ``fallbacks > 0`` and finiteness at ``dt = 1e9``; both should still hold
  (**hypothesis**).
- Unaffected by construction: ``test_fp32_compute.py:48``,
  ``test_fp32_jupiter.py:147`` and the ``test_fp32_science.py`` fallback
  check, which require zero fallbacks.
- No shipped deck runs the FP32 solve by default: ``precision: auto``
  picks FP32 only under ``dynamics/precision: float32`` (``hydro.cpp:502-503``).


11. Open questions to other lanes
---------------------------------

**Lane R (Linfeng): register interface.**

- R-1: Which call posts an x1 face transfer, and are the explicit and
  ``vic`` parts kept apart?
- R-2: Do you accept the six ``vic.*`` names in the order of section 4,
  and can the register hold cell increments next to face transfers?
- R-3: Does the register store A·dt-integrated transfers (kg per stage)
  or flux densities? Are the momentum components global on the cubed
  sphere?
- R-4: Would you take energy as a face transfer if the wall residual turns
  out to be round-off?

**Lane C (Tianhao): the well-balanced reference row.**

- Is any reference rebuilt from the stage state inside a stage (#369
  ``upward``, #361 FP32 gauge)? If so, the Jacobian misses that reference's
  derivative (I5).
- Do you accept a contract test in place of a row, and should it live in
  your lane or mine?

**Lane A (Jiachen): rho w for Coriolis and metric.**

- Can m1 read the register's final total instead of
  ``mass_correction()[IVZ]``?
- Is the cell increment enough for ``axial_x1_source_`` under the full
  matrix?

**Lane S (Jiheng): wall and solid rows against #369.**

- Is ``ColumnSegments``, built from ``X1Wall`` depth, the right owner of the
  surface rows and of ``k0``/``k1``?
- Is "floor == wall" the bitwise gate for I-1 to I-3?

**Lane X (Xi): FP64 fallback, device status, graph segments.**

- A predicated second FP64 launch, or CUDA conditional nodes?
- Is one ``int32`` read per step in ``check_redo`` the G3 target?
- Where do the graph segments start and end around the solve?
- Should the FP32 restriction to Cartesian ideal-moist stay?

**Cheng:** (c) of section 10. (a) and (b) were ruled on 2026-11-17.


12. Skeleton index
------------------

``docs/v3/skeleton/I/`` holds declarations and comments only, 298 lines in
all. It is not part of any build.

.. list-table::
   :header-rows: 1

   * - file
     - declares
     - replaces (pin)
   * - ``vertical_implicit.hpp``
     - ``VicMatrix``, ``VicStatusBit``, ``VerticalImplicitOptionsImpl``, ``ColumnSegments``, ``VicIncrement``, ``VerticalImplicitImpl``
     - ``implicit_hydro.hpp``; the ``scheme`` bit test; ``_solve_failed``; ``mass_correction()``/``correction()``
   * - ``gravity_work.hpp``
     - ``X1Measure``, ``X1GridKey``, ``CentroidSlopeStencil``, ``GravityWorkRows``, ``GravityWork``
     - ``gravity_work.hpp``, ``gravity_work_radial.hpp``; ``_pe_work``, ``_rx_work``, ``_g_work_lo/hi``
   * - ``vic_kernels.hpp``
     - ``VicParams``, ``VicColumn``, per-column kernel templates, ``CpuColumns``/``CudaColumns``, ``vic_stage``, 2 dispatch stubs
     - ``implicit_dispatch.hpp``; ``vic_assemble_{partial,full}_impl.h``; ``vic_gravity_work_impl.h``; the stub bodies of ``implicit_dispatch.cpp``/``.cu``
