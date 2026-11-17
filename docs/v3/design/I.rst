V3 lane I: vertical implicit solver and gravity work
=====================================================

:Lane: I (vertical implicit solver and gravity work)
:Owner: Xinyue Wang (alternate: Tianhao Le)
:Revision: 1 (round 1, written independently of the other lanes' drafts)
:Pin: chengcli/snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``

Every ``file:line`` below is at the pin unless it says otherwise. ``#369``
lines are at that PR's head ``c382f0da7b84dbc14219053758054f874669cc84``
(open; merge base ``cbf86830``). A claim I have not tested is marked
**hypothesis**. The results that change beyond round-off are marked
**FOR CHENG**.

The kickoff brief's list of twelve section titles was not in the context this
draft was written from. Sections 9 (line budget), 11 (open questions) and 12
(skeleton index) are where the lane message puts them. The other sections
follow the order of the lane message's design questions.

.. contents::
   :local:
   :depth: 1


1. Scope, pin and numbers
-------------------------

Lane I covers ``src/implicit/*`` (12 files), ``src/hydro/gravity_work.hpp``,
``src/hydro/gravity_work_radial.hpp`` and ``src/forcing/const_gravity.cpp``.
Their callers are read but not owned: ``src/hydro/hydro_forward.cpp:1412-1690``,
``src/hydro/hydro.cpp:479-549`` and ``src/mesh/meshblock.cpp:1036-1109``.

The brief's numbers, checked at the pin:

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
       (all tracked files under ``src`` give 43,629, because they also count ``CMakeLists.txt`` and ``*.cpp_``)
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
     - 135 only with ``grep -E 'torch::kFloat64|torch::kDouble|at::kDouble|kFloat64'``.
       ``kFloat64`` alone gives 134. ``kFloat64|kDouble`` gives 136, because it also counts ``c10::kDouble`` at ``src/coord/coordinate.hpp:405``.
     - see the left column
   * - ``set_``/``set_data``
     - 41
     - 41 only with ``grep -E '\.set_|set_data'``, which also counts 3 comment lines in ``src/utils/tensor_cast_cache.hpp`` and misses ``p->set_(v)``. The real call sites (``(\.|->)set_(data)?\(``) number 39.
     - see the left column

The same greps, run on the lane's files alone, give 7 ``.item<>`` lines (one is
commented out, at ``implicit_hydro.cpp:205``), 11 FP64 literals and 2
``set_data`` calls (``implicit_hydro.cpp:157,183``).

**Not checked.** ``gh`` has no valid credentials on this machine, so I read the
PRs through the public REST API (bodies) and through their squash commits on
main. The two Slack threads (gravity-work booking 1794169013.322299,
amplification analysis 1794212148.557499) and the earlier implicit inventory
were not in my context, and Slack cannot be read from here. In particular, I
could not reproduce the inventory's "191 copy lines in the partial/full
Jacobian pair". My own count: ``vic_assemble_full_impl.h`` has 118 lines, and
68 of them also appear (after trimming whitespace) in
``vic_assemble_partial_impl.h`` (206 lines). The two assembly drivers add 114
lines in ``implicit_dispatch.cpp:24-137`` and 112 in
``implicit_dispatch.cu:32-143``.


2. The code at the pin
----------------------

One stage of the vertical implicit correction (``hydro_forward.cpp:1554-1616``,
then ``implicit_hydro.cpp:175-616``):

#. The explicit face gravity work is computed from the x1 mass flux after
   positivity limiting and sedimentation (``hydro_forward.cpp:1412-1537``). There are two
   formulas: a Cartesian one that differences the geometry first
   (``gravity_work.hpp:12-24``, called at ``hydro_forward.cpp:1447-1452``),
   and a curved-grid one written as ``phi_cell div(M) - div(phi_face M)``
   (``:1453-1465``). The corrected-PE work is added on top
   (``:1470-1495``).
#. In face mode with a solve, that work goes into ``du[IPR]`` *before* the
   solve (``:1601-1606``). The stage weight is applied to the operator's dt
   for integrators with 3 or more stages (``:1568-1583``).
#. ``_apply_implicit_correction`` handles the FP32/FP64 boundary, the energy
   offset and the whole-block FP64 retry (``hydro.cpp:502-543``).
#. ``forward_masked`` projects to the local orthonormal frame (``implicit_hydro.cpp:254-259``).
   It builds grid caches (``:262-294``) and assembles a, b, c, with the face-work
   flags in ``dir`` bits 16/32/64 (``:330-337``, ``implicit_dispatch.hpp:27-30``).
   It then adds the corrected-PE tridiagonal rows (``:365-425``), solves, and
   applies the Woodbury wall third points (``:436-472``). Next come a host
   screen (``:474-488``) and the redistribution (``:489-493``). Two post-solve
   gravity-work remainders follow: projection/clamp (``:522-556``) and
   corrected-PE (``:569-594``). The function ends with the de-projection
   (``:558-561``) and a device-latched failure (``:599-615``).
#. After the solve, the registered x1 mass (explicit flux plus
   ``mass_correction()[IVZ]``) feeds Coriolis and metric m1
   (``hydro_forward.cpp:1624-1680``, #355). The dry transfer
   ``mass_correction()[IVY]`` feeds the tracer donor transfer
   (``meshblock.cpp:786-806``).

Gravity-work variants at the pin: ``gravity-work: face`` (the default since
#303, ``const_gravity.cpp:54``) books the plain face work plus the corrected-PE
work wherever the grid has a measure (``hydro.cpp:328-333``,
``gravity_work_radial.hpp:20-25``). ``gravity-work: cell`` books the cell work
``rho g v``. It adds the face work of the non-Riemann mass flux only
(``hydro_forward.cpp:579-580, 846-849, 1426-1434``), puts the Roe-diffusion
work into the operator (``kVicDiffusiveCell``, ``vic_gravity_work_impl.h:243-247``)
and turns on the global E+PE fixer by default (``const_gravity.cpp:59``,
``meshblock.cpp:1036-1109``). ``face-wallc`` and the environment switch were
removed in #345. No YAML file in the repository sets ``gravity-work``
(``grep -rhn "gravity-work:" --include=*.yaml .`` returns nothing), so every
shipped deck runs ``face``.


3. The implicit correction as a register increment (question 1)
---------------------------------------------------------------

**What the solve posts.** The solve posts what the column update moves
across faces, and posts the rest as cell increments.

.. list-table::
   :header-rows: 1

   * - quantity
     - posted as
     - source at the pin
   * - dry-gas mass, each species
     - x1 face transfer (kg per stage), after the availability clamps
     - ``vic_redistribute_impl.h:119-175`` (``MASS(IVX+(dir+1)%3)`` dry, species via ``q``)
   * - total mass
     - x1 face transfer, the sum of the above (the "moved" mass)
     - ``MASS(IVX+(dir+2)%3)``, ``vic_redistribute_impl.h:88-89, 144, 172``
   * - tangential momenta (partial matrix)
     - x1 face transfer at the donor velocity
     - ``vic_tangential_momentum_column``, ``vic_redistribute_impl.h:186-201``
   * - x1 momentum; energy; tangential momenta (full matrix)
     - cell increment
     - ``vic_redistribute_cell``, ``vic_redistribute_impl.h:205-225``
   * - gravity work
     - not posted by the solve; booked once from the registered total x1 mass (section 4)
     - today: ``implicit_hydro.cpp:522-556, 569-594``

Mass, species and partial-matrix tangential momentum are face transfers by
construction at the pin. The redistribution prefix-sums the zero-sum part of
the solved mass change (``vic_redistribute_impl.h:110-116``), so a face moves
exactly what one cell loses and its neighbour gains. The x1 momentum has a
gravity source and wall pressure, so its column sum is not conserved and it
stays a cell increment. Energy stays a cell increment in V3 round 1. The
linearised Roe flux at a mirrored wall face need not vanish: the mass row
shows this through the "column residual R" that the redistribution removes
in proportion to mass (``vic_redistribute_impl.h:65-71``). The energy row has
the same kind of residual. **Hypothesis:** the energy residual at a sealed wall
is of round-off size, which would let energy become a face transfer too. I
have not measured it, and it is listed for lane R (section 11).

**Frame.** The solve runs in the local orthonormal frame (``implicit_hydro.cpp:254-259``)
and de-projects (``:558-561``, ``flux2global1_``). The two cells across an x1
face share (j, k), so cell frame and face frame agree for x1 faces. The solve
posts after de-projection, in the same frame and components as the explicit
x1 fluxes in the register. Mass and energy are frame-free. **Hypothesis:** on
the cubed sphere the de-projected tangential transfer is what the register's
explicit x1 momentum flux uses. This depends on lane R's component
convention.

**Units and area.** The solve's transfers are area- and dt-integrated (kg per
stage, ``vic_redistribute_impl.h:82-89``). The explicit fluxes are per area
per time (``hydro_forward.cpp:1641``: ``phi = area * f1``; ``:1646``:
``mc[IVZ] / dt``). Whether the register stores integrated transfers or flux
densities is lane R's choice (section 11). The solve converts once at the post.

**When in the stage.** At the same point as now. The order is: explicit x1
fluxes posted; forcings, including the w-free Coriolis rows that must pass
through the scheme-9 solve (#355 commit 6); explicit gravity work
``G[M_exp]`` into the RHS; the solve; the implicit transfers posted. Then the
readers of the final x1 transfers run: the gravity-work remainder (section
4), Coriolis/metric m1 (section 8) and the tracer donor transfer. The
register's x1 mass slot is final only after the solve posts. Every reader
listed here runs after that point.


4. Gravity work as one function of the registered x1 mass (question 2)
-----------------------------------------------------------------------

**Definition.** Let :math:`M_f` be the x1 mass moved through face :math:`f`
in the stage (kg, all constituents), :math:`\Phi=-g_1 x_1`, and let cell
:math:`i` have lower face :math:`l` and upper face :math:`u`. Then

.. math::

   G_i[M] = \underbrace{\frac{g_1 (x_u - x_i)\,M_u - g_1 (x_l - x_i)\,M_l}{V_i}}_{W_i[M]}
          + \underbrace{g_1\,\sigma_i^2\, s_i\!\left[\Delta\rho\right]
          + g_1\,(r_{c,i}-x_{1v,i})\,\Delta\rho_i}_{K_i[\Delta\rho]},
   \qquad \Delta\rho_i = \frac{M_l - M_u}{V_i}.

:math:`W` is ``gravity_work_from_face_mass`` (``gravity_work.hpp:12-24``).
:math:`K` is ``corrected_pe_work`` (``gravity_work_radial.hpp:190-205``),
whose second term exists on gnomonic-equiangle only (``:172-176``). The
curved formula at ``hydro_forward.cpp:1453-1465`` is algebraically the same
:math:`W`:

.. math::

   \Phi_i\,\frac{M_u - M_l}{V_i} - \frac{\Phi_u M_u - \Phi_l M_l}{V_i}
   = \frac{(\Phi_i-\Phi_u)M_u - (\Phi_i-\Phi_l)M_l}{V_i} = W_i[M].

So V3 keeps one formula for all grids: the difference-first one, which
``gravity_work.hpp:9-11`` documents as safe at large origin. :math:`G` is
linear in :math:`M`.

**One object.** ``GravityWork`` (skeleton ``gravity_work.hpp``) owns the
grid-only coefficients:

- the face weights :math:`g_1(x_f - x_i)/V_i`;
- the slope bands of :math:`K` (``CentroidSlopeStencil::band``, ``gravity_work_radial.hpp:61-67``) and its two wall third points.

It has two outputs from those same tensors: ``apply(M)`` returns
:math:`G[M]`, and ``rows()`` returns the coefficients the operator needs.
Today the factors are cached twice, in ``HydroImpl::_pe_work``
(``hydro.hpp:434``) and ``ImplicitHydroImpl::_rx_work``
(``implicit_hydro.hpp:133``). The face weights are built a third time as
``_g_work_lo/_g_work_hi`` (``implicit_hydro.cpp:279-284``). V3 has one
instance, owned by Hydro.

**The Jacobian's energy row equals the booked work by construction.** In the
operator, the linearised implicit transfer through face :math:`f` is
:math:`M^{\rm lin}_f(\delta q) = \Delta t A_f\,[\tfrac12(m_{f-}+m_{f+}) - \tfrac12 B^\rho_f(\delta q_{f+}-\delta q_{f-})]`
(#288 derivation). Because :math:`G` is linear,

.. math::

   \frac{\partial G_i[M^{\rm lin}]}{\partial \delta q_j}
   = \sum_f \frac{\partial G_i}{\partial M_f}\,\frac{\partial M^{\rm lin}_f}{\partial \delta q_j}.

V3 assembles the energy row as exactly this product. The first factor comes
from ``GravityWork::rows()``. The second is the mass row of the same
:math:`|A|` and flux Jacobian blocks the assembly already builds
(``vic_gravity_work_impl.h:229-254`` passes ``am``/``ap`` as the mass rows
today). The booked work for the stage is

.. math::

   \underbrace{G[M^{\rm exp}]}_{\text{RHS, before the solve}}
   + \underbrace{R\,\delta q}_{\text{in the operator}}
   + \underbrace{G[M^{\rm imp}] - R\,\delta q}_{\text{after the solve}}
   = G[M^{\rm exp} + M^{\rm imp}] = G[M^{\rm register}],

where :math:`R` is the assembled energy row and :math:`M^{\rm imp}` is the
posted (clamped, projected) implicit transfer. The after-solve term is one
call: :math:`R\,\delta q` evaluated on the solved vector, subtracted from
``apply`` of the posted transfer. It is zero wherever no clamp or projection
acted, because then :math:`M^{\rm imp} = M^{\rm lin}(\delta q)` and
:math:`R\,\delta q = G[M^{\rm lin}(\delta q)]`. This generalises #345's
identity :math:`K[\Delta\rho_0] + K[\delta\rho_{\rm raw}-\Delta\rho_0] + (K[\Delta\rho-\Delta\rho_0] - K[\delta\rho_{\rm raw}-\Delta\rho_0]) = K[\Delta\rho]`
to :math:`W` and :math:`K` together. It replaces the two hand-written
remainders at ``implicit_hydro.cpp:522-556`` and ``:569-594``. The identity
holds because both sides read the same coefficient tensors. A unit test
checks it with random :math:`\delta q`: ``apply(M_lin(dq)) == rows() * dq``
to round-off. That test fails if anyone edits one side alone.

**How the four PRs broke the pairing.** Each time the booked work gained a
term, the operator row had to be edited separately by hand:

- **#284.** The face work was booked outside the operator while the matrix
  linearised the cell work. A tall rest column blew up at acoustic Courant
  about 66 (#283). The PR switched the default to cell plus a fixer (#284 body,
  "Why").
- **#285.** It put the Cartesian face work into the energy row. Curved grids
  still "linearize cell work and replace it after the solve" (#288 body).
- **#288.** It added curved weights to the rows. The matrix energy-budget
  defect fell from 6.64e-4 to below 3e-16 (#288 body). It kept a separate
  bitwise-preserving Cartesian path (``kVicCartesianFaceWork``,
  ``vic_gravity_work_impl.h:235-242``).
- **#297.** It added :math:`K` to the booked work. The operator lacked it, so
  part of it became explicit and a rest column grew at Courant about 197-250
  (#296). Fix f256ab5 put the tridiagonal part into the row, but the two wall
  third points stayed after the solve until #345 added them by Woodbury
  (wall mode from :math:`N\Delta t \approx 119`-149 before, #345 body).

**Decision: the surviving gravity-work form and its default.** One form:
the face work :math:`W` plus the corrected-PE work :math:`K`, with
:math:`K\equiv 0` where the grid has no measure (``X1Measure::none``,
cylindrical: plain face work and a warning, as now). It is the default and
the only choice. The ``gravity-work`` key is still accepted with the value
``face``. ``cell`` is refused, and the message points to this note. Why:

- it is what every shipped deck runs (section 2);
- E+P closes by construction (#297, #345);
- it is the only form whose operator row is derived from the booked work;
- the fixer that cell needs has a documented tall-column instability (#284
  "Limit", #285 "Limits") and costs a global reduction per step.


5. Which variants survive (question 3)
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 34 14 34

   * - variant
     - code at the pin
     - V3
     - what changes in results
   * - face + corrected-PE ("radial-exact")
     - ``gravity_work_radial.hpp``; ``implicit_hydro.cpp:348-425, 436-472, 569-594``; ``hydro_forward.cpp:1470-1495``
     - survives as ``GravityWork`` (W + K)
     - none intended; the one-call remainder is round-off (**hypothesis**, bitwise harness of #348 to confirm)
   * - face, Cartesian 0.25 path
     - ``vic_gravity_work_impl.h:235-242``; ``implicit_hydro.cpp:333-334``; ``kVicCartesianFaceWork``
     - merged into the general weights (on a uniform grid ``work_lo = work_hi = 1/4``, ``implicit_hydro.cpp:279-282``)
     - round-off on Cartesian implicit decks (**hypothesis**: other operation order, the same values)
   * - face, curved explicit formula
     - ``hydro_forward.cpp:1453-1465``
     - replaced by the difference-first :math:`W`
     - round-off; it should reduce the large-origin cancellation #288 measured (3e-12 at r = 7e7 m) (**hypothesis**)
   * - cell
     - ``const_gravity.cpp:54-64``; ``hydro_forward.cpp:579-580, 846-849, 1426-1434, 1505-1533``; ``kVicDiffusiveCell``
     - **removed**
     - any deck or script that sets ``cell`` moves to face: different energy and entropy drift (#284 table: moving column entropy drift 1.97 → 1.11 at nz 64). **FOR CHENG**
   * - global E+PE fixer
     - ``meshblock.cpp:1036-1109``; ``hydro_forward.cpp:1416-1417, 1505-1533, 1584-1598, 1608, 1682-1693``; ``hydro.hpp:290-314``; ``hydro.cpp:120-145``
     - **removed** (it existed only with cell)
     - none for face decks; see the note below
   * - scheme 0 no-op module
     - ``implicit_hydro.cpp:179-188`` (reachable from Python only; YAML returns ``nullptr``, ``:91``)
     - removed: no solve means no module
     - none

Note on the fixer. ``gravity_work_fixer_sums`` also carries the solve's
redo status and the limiter marks in its one host copy
(``meshblock.cpp:1050-1056``). Nothing is lost when it goes:
``check_redo`` already reads the solve latch in its own copy
(``meshblock.cpp:1549-1551``). The fixer sums run only with the fixer on
(``meshblock.cpp:997-1003``).

Tests to retire or re-target with ``cell``/fixer: ``test_gravity_work_fixer.py``,
``run_straka_redo.cmake`` (fixer on), and the cell arms of
``test_implicit_gravity_tall_column.py``, ``test_implicit_stratified_solid.py``,
``test_lu_failure.cpp``, ``test_stencil_cache_small_block.py`` and
``test_mps_matches_cpu.py`` (from ``grep -rlE`` for ``cell``/``fixer`` in ``tests``).


6. One Jacobian assembly; ``scheme`` (question 4)
-------------------------------------------------

**One assembly.** Both kernels build the same 5x5 objects: ``Am``, ``Ap``
(``Rmat Lambda Rimat``) and the three ``FluxJacobian`` results
(``vic_assemble_partial_impl.h:80-137``, ``vic_assemble_full_impl.h:52-88``).
The partial kernel copies the rows and columns {IDN, IVX+dir, IPR} out of
them (``partial:91-93, 109-111, 135-137``). Its ``Phi`` and ``Bnd`` are the
full ones restricted to those indices (``partial:70-74`` against
``full:39-46``). The block sums ``a = (Am A_l + Ap A_u + (A_u - A_l) J_c) / 2V - Phi + I/dt``
and the like (``partial:144-149``, ``full:95-100``) are entrywise. So each
entry of the partial block is computed by the same scalar expression as the
matching entry of the full block. V3 assembles the 5x5 blocks once and, for
N = 3, restricts them and runs the partial-only tangential-momentum fold
(``partial:159-199``) under ``if constexpr (N == 3)``. The gravity-work rows
are already shared (#348, ``vic_gravity_work_impl.h``). **Hypothesis:** the
partial matrix stays bitwise, since Eigen's fixed-size coefficient-wise sums
evaluate each entry independently. The #348 harness (16 decks, CPU and CUDA)
is the gate. The partial kernel then does the full 5x5 block sum before
restricting, which costs a few more flops per cell. It is not timed; it
should be well inside the run-to-run spread #348 measured, but that is a
**hypothesis**.

Two kinds of kernel must stay, and their reason is not a second assembly:

- The partial solve has 3 unknowns and a right-hand-side correction
  (``forward_sweep_impl.h:63-69``).
- The partial redistribution carries tangential momentum
  (``implicit_dispatch.cpp:227-231``).

Both stay as ``if constexpr`` branches of the one N-templated kernel, as
``ForwardSweep`` already is.

**Decision: ``scheme`` is no longer a bit field.** At the pin, only
``scheme`` 0, 1 and 9 are accepted (``implicit_hydro.cpp:97-111``). Bit 3
selects the matrix (``implicit_hydro.hpp:37``, ``implicit_hydro.cpp:427, 489``),
and no other bit has a meaning. The face-work flags live in a different
bit field, the ``dir`` argument (``implicit_dispatch.hpp:27-30``). V3 has
``enum class VicMatrix {partial = 3, full = 5}``, where the value is the
block size. YAML ``implicit-scheme: 1 | 9`` keeps working, mapped once in
``from_yaml``. With one gravity-work form, the ``dir`` flags disappear and
``dir`` is the direction again, inside a ``VicParams`` struct. Results do not
change. The Python ``ImplicitOptions.scheme()`` int (``python/snapy/implicit.pyi:55-60``)
can stay as a compatibility spelling. That is the Python lane's call.


7. Wall rows, solid rows and the well-balanced reference (questions 5, 6)
--------------------------------------------------------------------------

**Where wall and solid rows live at the pin.**

- assembly: a solid cell is an identity/dt row, and a fluid cell next to a
  solid mirrors it (``implicit_dispatch.cpp:118-130``, and the same in ``.cu``);
- physical walls: ``a += b Bnd`` (``vic_assemble_full_impl.h:107-111``);
- redistribution: per fluid segment (``implicit_dispatch.cpp:216-235``);
- corrected-PE rows: masked by ``rx_fluid`` (``implicit_hydro.cpp:405, 412-420``).

#369 adds surface rows from each column's wall frame (``_rs_mid/_rs_lo/_rs_hi/_rs_ts``,
``implicit_hydro.cpp`` +82 lines at c382f0da). It also moves the lower Woodbury
row ``k0`` to the first fluid cell (``implicit_dispatch.cpp``/``.cu``,
+6/+7). In the solid case it computes the work per column through
``pib->per_column``.

**After the refactor.** One ``ColumnSegments`` descriptor per block, built
once per solid-mask version from #369's ``X1Wall`` depth. For each column it
holds the fluid segment ``[lo, hi)`` and the wall kind at each end (physical
wall or solid surface). Its owners and readers:

- the column operator owns it (skeleton ``vertical_implicit.hpp``);
- the assembly reads ``lower_wall``/``upper_wall`` per cell from it;
- the Woodbury rows read their cell indices ``k0``/``k1`` from it (#369's
  ``k0`` scan disappears from both dispatch files);
- ``GravityWork::rows()`` is built per wall depth from #369's ``x1_depth``
  frames. The ``_rs_*`` and ``rx_fluid`` tensors become one per-segment row
  set.

Hydro and meshblock keep nothing of the column structure. Order: #369 merges
first and keeps priority. V3 rebases onto it and does not reorder its
arithmetic. The bitwise "floor == wall" checks of #369 are the gate.

**The well-balanced reference (question 6).** **Decision:** V3 does *not* add
a reference row to the operator. It adds a contract and a test. Reasons:

- The operator acts on increments. A rest column whose explicit increment is
  exactly zero gets a zero correction bit for bit. The RHS is zero, the
  m_t fold adds nothing for zero ``du`` (``vic_assemble_partial_impl.h:40-41``)
  and ``rx_tri(rx_mass0)`` is zero (``implicit_hydro.cpp:421-424``). The
  balance therefore comes from the RHS, which lane C owns.
- The reference terms in the explicit RHS do not depend on the state
  within a stage: ``face_gradient(pressure_base1)`` (``hydro_forward.cpp:837, 1394-1400``)
  and ``background_force1`` with ``density_delta`` (``:1380-1382``). Their
  derivative with respect to :math:`\delta q` is zero, so the Jacobian of
  the split RHS equals the Jacobian of the full RHS, which is what the
  assembly builds (full gravity, ``implicit_hydro.cpp:326-329``). This is a
  **hypothesis** for any reference that is rebuilt from the stage state
  inside the stage: the ``upward`` wall reference of #369 and the FP32 gauge
  of #361. If such a reference moves with :math:`\delta q`, a row is missing.
- The contract test: on every reference mode (main, upward, ref4), a balanced
  rest column with scheme 1 and 9 gives a correction that is exactly zero,
  and the Jacobian matches a finite-difference Jacobian of the split RHS.
  This question goes to lane C (section 11).


8. FP32, device status, ``rho w``, one kernel definition (questions 7, 8)
-------------------------------------------------------------------------

**Solve precision.** No change to the policy: ``auto | float32 | float64``.
FP32 is allowed only on Cartesian ideal-moist (``hydro.cpp:502-507``), and
the energy offset and RHS masters stay FP64 at the boundary (``:508-511``).

**Stage-local FP64 fallback.** At the pin a failed FP32 block is re-solved
whole in FP64 from ``rhs64`` (``hydro.cpp:529-535``). That needs two host
reads: ``solve_failed()`` (``implicit_hydro.hpp:103-105``) and
``isfinite(rhs64).all().item`` (``hydro.cpp:530``). V3 instead launches the
FP64 stage once more in the same stage, unconditionally, with a column
predicate: the kernel returns at once for each column whose status bit is
clear. This needs no host read and fits a graph. Its cost when nothing
failed is one launch per stage that does no column work (**hypothesis**; to
be timed on C48 as in #358). **FOR CHENG:** in a block with a failed column,
the columns that did not fail keep their FP32 result. Today the whole block
is re-solved in FP64, so on failing steps those columns differ beyond
round-off. Decks that never fall back are unchanged; #361 reports zero
fallbacks on its acceptance decks. I chose a predicated second launch over an
FP64 path inside the FP32 kernel because the K = 2 kernel is already
register-bound (#358 dropped it to 64 threads).

**Device status in place of host booleans (G3).** A 0-d ``int32`` status per
block, written by the kernels, replaces the host booleans and reads below.
Its bits are: nonfinite input, singular column, nonfinite result, FP64
retried, FP64 failed.

- ``_solve_failed`` (``implicit_hydro.hpp:120``);
- the screen read (``implicit_hydro.cpp:477``);
- the column reads (``:479, 485, 487``);
- the ``donor_cap`` read ``binds.any().item`` (``implicit_dispatch.cpp:255``),
  which becomes an always-run sweep, or a device-predicated one on CUDA.

The input check moves into the assembly kernel: a column with a nonfinite
input sets the bit and is not solved. The failure path, with its column log
(``implicit_hydro.cpp:219-239``), runs in ``check_redo``. That function
already does one device read per step (#306) and already reads the solve
latch in it (``meshblock.cpp:1549-1551``). In-stage host reads in the lane go from 6 to 0. ``check_redo`` keeps its
one read per step.

**FP64 literals in the lane.** The scratch buffers are registered as FP64
and then ``set_data``-resized (``implicit_hydro.cpp:126-136, 155-158``). In
V3 they become unregistered scratch, allocated once per shape on
``w.options()``. That removes 9 of the 11 FP64 literals and both
``set_data`` calls. The meters (clamp residual, dry clamp, fallback count)
stay registered, in the accumulation dtype. ``GeoKey::st = torch::kDouble``
(``implicit_hydro.hpp:141``) goes with ``GeoKey``, whose caching
``GravityWork`` takes over. **Open for lane X:** graph replay needs stable
scratch addresses, so the buffers must not be reallocated between replays
(section 11).

**``rho w`` for Coriolis and metric (#355).** #355 builds m1 from
``area * flux1 + mass_correction()[IVZ] / dt`` (``hydro_forward.cpp:1637-1653``).
In V3 this becomes one register read: the final x1 total mass transfer,
explicit plus implicit, after the solve has posted (section 3). The m1
weights and the Coriolis call order (the w-free rows before the solve, the m1
rows after) stay with lane A. ``mass_correction()`` and ``correction()`` stop
being public accessors. The tracer donor transfer reads the dry transfer
from the register too (``meshblock.cpp:786-806``). The one exception is the
AAM path, which reads ``picorr->correction()`` (``hydro_forward.cpp:1668-1669``):
it needs the tangential-momentum transfers, which the register holds for the
partial matrix. For the full matrix, which has cell increments, lane A
decides (section 11).

**CPU, CUDA and MPS from one kernel definition (question 8).** The column
bodies are already shared ``DISPATCH_MACRO`` templates in the ``*_impl.h``
files. What is duplicated is the plumbing. ``implicit_dispatch.cpp``
(313 lines) and ``.cu`` (307) each unpack 13-15 iterator operands per stub,
for 9 stubs (``implicit_dispatch.hpp:174-182``). The two files differ in 350
lines once ``_cpu``/``_cuda`` are normalised (``diff`` count). V3:

- one ``vic_stage<Launch>`` template in a shared header unpacks the operands
  once into a ``VicColumn`` view;
- ``Launch`` is ``CpuColumns`` (``iter.for_each``) or ``CudaColumns``
  (``gpu_kernel``, 64 threads for K = 2 as in #358);
- ``implicit_dispatch.cpp`` instantiates and registers CPU and MPS;
- ``implicit_dispatch.cu`` instantiates and registers CUDA;
- 2 stubs remain: one ``vic_stage`` stub that takes the stage
  (assemble, solve, redistribute) as an argument, and ``donor_cap``. N and K
  become runtime arguments that pick a template instance;
- the Woodbury 2x2 runs in the solve kernel after the back-substitution
  (``implicit_hydro.cpp:446-472`` is per-column scalar algebra). That removes
  about a dozen small host launches (#358 "Not in this PR": ~1 ms per step).
  **Hypothesis:** the change is round-off only (FMA contraction on CUDA).

MPS has no kernel of its own: it runs the CPU kernels on host copies
(``OnHost``, ``implicit_dispatch.cpp:295-311``, since a9d6063). V3 keeps that
and says so. An MPS kernel is out of scope.

The ``*_impl.h`` files remain:

- ``flux_decomposition_impl.h``: unchanged;
- ``forward_sweep_impl.h``: unchanged;
- one ``vic_assemble_impl.h``, replacing partial + full + gravity-work;
- ``vic_redistribute_impl.h``: named output slots instead of
  ``MASS(IVX+(dir+k)%3)`` reuse;
- ``vic_solve_failure.h``: it sets the status bit.


9. Line budget
--------------

The baseline is the 2,655 lines measured in section 1. The V3 numbers are
estimates from the design above, not counts of written code.

.. list-table::
   :header-rows: 1

   * - file (pin)
     - pin
     - V3
     - delta
     - why
   * - ``implicit/implicit_hydro.hpp``
     - 158
     - 95
     - -63
     - ``VicMatrix``; no ``GeoKey``, ``_rx_*``, ``_g_*`` (``GravityWork`` holds them); status tensor
   * - ``implicit/implicit_hydro.cpp``
     - 627
     - 420
     - -207
     - no scheme-0 path; one remainder call instead of 2 blocks (-75); Woodbury in kernel (-27); no host screens/reject (-45); one cache
   * - ``implicit/implicit_dispatch.hpp``
     - 37
     - 25
     - -12
     - 2 stubs (``vic_stage``, ``donor_cap``); no ``dir`` flags
   * - ``implicit/implicit_dispatch.cpp``
     - 313
     - 200
     - -113
     - one launcher-templated stage set; operand unpacking once
   * - ``implicit/implicit_dispatch.cu``
     - 307
     - 90
     - -217
     - CUDA launcher + instantiation + registration only
   * - ``vic_assemble_partial_impl.h`` + ``_full_impl.h``
     - 324
     - 220
     - -104
     - one N-templated assembly, restriction for N = 3
   * - ``implicit/vic_gravity_work_impl.h``
     - 50
     - 25
     - -25
     - one form: rows from ``GravityWork`` weights
   * - ``implicit/forward_sweep_impl.h``
     - 138
     - 140
     - +2
     - column predicate for the FP64 retry
   * - ``implicit/flux_decomposition_impl.h``
     - 130
     - 130
     - 0
     -
   * - ``implicit/vic_redistribute_impl.h``
     - 232
     - 232
     - 0
     - named slots replace the comment table
   * - ``implicit/vic_solve_failure.h``
     - 21
     - 25
     - +4
     - status bits
   * - ``hydro/gravity_work.hpp`` + ``gravity_work_radial.hpp``
     - 250
     - 215
     - -35
     - one header, one ``GravityWork`` class (``apply``, ``apply_seam``, ``rows``, ``potential``)
   * - ``forcing/const_gravity.cpp``
     - 68
     - 50
     - -18
     - no ``cell`` / fixer keys and checks
   * - **total**
     - **2,655**
     - **1,867**
     - **-788 (-29.7%)**
     - the target is at least 20% (at most 2,124)

There are also deletions outside the baseline, in files other lanes own and
count. These are estimates from the line ranges in section 5:

- fixer: ``meshblock.cpp:1036-1109`` (74 lines) and about 70 lines of
  ``hydro_forward.cpp``, ``hydro.cpp`` and ``hydro.hpp``;
- the explicit face-work block ``hydro_forward.cpp:1412-1537`` (126 lines)
  becomes about 30 lines around ``GravityWork::apply``.

I do not count either in the lane's 29.7%.


10. Results that change; verification
-------------------------------------

**FOR CHENG** (beyond round-off):

#. **``gravity-work: cell`` and the fixer are removed.** A deck or script
   that selects ``cell`` is refused and must move to ``face``. Its energy,
   entropy drift and E+PE diagnostics change (#284 table). No YAML file in
   the repository selects it. 8 test files do (section 5). Decks with
   periodic x1 and gravity lose the alternative the face warning suggests
   today (``hydro.cpp:146-156``: "use x1 walls or gravity-work: cell").
#. **The FP64 fallback is per column.** On a step where an FP32 column
   fails, the other columns of that block keep FP32 results, where today
   they are re-solved in FP64 (section 8).

Round-off only, all **hypotheses** to be gated bitwise or at round-off by the
#348 harness (16 implicit decks x CPU/CUDA):

- the Cartesian 0.25 path merged into the general weights;
- the curved explicit :math:`W` in difference-first form;
- the single remainder call;
- the Woodbury 2x2 in the kernel;
- the partial assembly as a restriction (expected bitwise).

Unchanged on purpose: the stage-weighted operator dt for 3 or more stages
only (``hydro_forward.cpp:1568-1583``; generalising it would change rk1/rk2
results); the ``nb1 = 1`` refusal (``hydro.cpp:486-491``); and the full
gravity in the operator, whatever ``non-hydrostatic`` is (``implicit_hydro.cpp:326-329``).

New tests (RED/GREEN where they guard a defect):

- ``GravityWork`` rows-vs-apply identity, random :math:`\delta q`, every
  measure. It goes RED if either side is edited alone;
- G of the register's mass equals the booked work, per stage, on a tall
  column, scheme 1 and 9;
- well-balanced contract: zero correction on rest columns for each reference
  mode; Jacobian against a finite difference of the split RHS (section 7);
- device status: a NaN column under FP32 is retried in FP64 with no host
  read inside the stage (counted with a sync counter);
- the existing E+P budget (``test_conservation_budget``), tall-column
  large-Courant (#345) and floor == wall (#369) gates, unchanged.


11. Open questions to the other lanes
-------------------------------------

**Lane R (Linfeng): register interface.**

- What call does the solve use to post an x1 face transfer, and does the
  register keep explicit and implicit contributions apart or only their sum?
- Increment names and order. I need slots for: total mass, dry mass, each
  species, two tangential momenta, and cell increments for x1 momentum and
  energy (and, for the full matrix, the tangential momenta). Can the
  register hold cell increments (sources) next to face transfers, or should
  they stay outside it?
- Frame and area weighting. Does the register store area- and dt-integrated
  transfers (kg per stage, like the solve) or flux densities (like
  ``_flux1``)? Are momentum components global or local-frame on the cubed
  sphere?
- Would you accept energy as a face transfer if the wall residual turns out
  to be round-off (the hypothesis in section 3)?

**Lane C (Tianhao): the well-balanced reference row.**

- Is any reference rebuilt from the stage state inside the stage (the
  #369 ``upward`` reference, the #361 FP32 gauge)? If so, the Jacobian is
  missing that reference's derivative.
- Do you agree that V3 needs a contract test rather than a reference row
  (section 7)?
- Should the contract test live in your lane or mine?

**Lane A (Jiachen): rho w for Coriolis and metric.**

- Can m1 read the register's final x1 total mass transfer instead of
  ``mass_correction()[IVZ]``, and do you keep the w-free rows before the
  solve?
- The AAM path needs the tangential-momentum transfers. Under the full
  matrix they are cell increments: is the cell increment enough for
  ``axial_x1_source_``?

**Lane S (Jiheng): wall and solid rows against #369.**

- Is a per-segment ``ColumnSegments`` (fluid ``[lo, hi)`` and the wall kind at
  each end, built from ``X1Wall`` depth) the right owner of the surface rows
  and of the Woodbury ``k0``/``k1``?
- What should V3 do with #369's per-column ``per_column`` work path, and will
  the "floor == wall" bitwise rows be the gate?

**Lane X (Xi): FP64 fallback, device status, graph segments.**

- Is a predicated second FP64 launch acceptable inside a captured graph, or
  do you want CUDA conditional nodes?
- Is one ``int32`` status read in ``check_redo`` per step the G3 target?
- Can the scratch buffers be graph-static per shape, and where do the
  graph segment boundaries fall around the solve: before the assembly,
  after the redistribution?
- Should the FP32 restriction (Cartesian ideal-moist,
  ``hydro.cpp:502-507``) stay in V3?

**For Cheng:** the two FOR CHENG items in section 10.


12. Skeleton index
------------------

``docs/v3/skeleton/I/`` holds declarations and comments only. It is not added
to any build.

.. list-table::
   :header-rows: 1

   * - file
     - declares
     - replaces (pin)
   * - ``vertical_implicit.hpp``
     - ``VicMatrix``, ``VicStatusBit``, ``VerticalImplicitOptionsImpl``, ``ColumnSegments``, ``VicIncrement``, ``VerticalImplicitImpl``
     - ``implicit_hydro.hpp``; ``ImplicitOptionsImpl::scheme`` bit test; ``_solve_failed``; ``mass_correction()``/``correction()``
   * - ``gravity_work.hpp``
     - ``X1Measure``, ``X1GridKey``, ``CentroidSlopeStencil``, ``GravityWorkRows``, ``GravityWork``
     - ``gravity_work.hpp``, ``gravity_work_radial.hpp``; ``_pe_work``, ``_rx_work``, ``_g_work_lo/hi``
   * - ``vic_kernels.hpp``
     - ``VicParams``, ``VicColumn``, per-column kernel templates, ``CpuColumns``/``CudaColumns``, ``vic_stage``, 2 dispatch stubs
     - ``implicit_dispatch.hpp``; ``vic_assemble_{partial,full}_impl.h``; ``vic_gravity_work_impl.h``; the stub bodies of ``implicit_dispatch.cpp``/``.cu``
