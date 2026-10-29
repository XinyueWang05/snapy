Configuration
=============

Snapy simulations are configured using YAML files that specify mesh parameters, time integration settings, physical models, and output options.

Basic Configuration
-------------------

A minimal configuration file looks like this:

.. code-block:: yaml

    mesh:
      nx1: 100        # Grid points in x1 direction
      nx2: 100        # Grid points in x2 direction
      nx3: 1          # Grid points in x3 direction (1 for 2D)
      x1min: 0.0      # Domain minimum in x1
      x1max: 1.0      # Domain maximum in x1
      x2min: 0.0      # Domain minimum in x2
      x2max: 1.0      # Domain maximum in x2

    time:
      cfl: 0.8        # CFL number for time step control
      tlim: 1.0       # Simulation end time

    hydro:
      gamma: 1.4      # Adiabatic index
      riemann: hllc   # Riemann solver type

    output:
      dt: 0.1         # Output interval

Mesh Configuration
------------------

The mesh section defines the computational domain:

.. code-block:: yaml

    mesh:
      # Grid dimensions
      nx1: 100
      nx2: 100
      nx3: 50

      # Domain extent
      x1min: 0.0
      x1max: 10.0
      x2min: 0.0
      x2max: 10.0
      x3min: 0.0
      x3max: 5.0

      # Boundary conditions
      bc_x1: periodic
      bc_x2: periodic
      bc_x3: outflow

Coordinate Systems
~~~~~~~~~~~~~~~~~~

Snapy supports multiple coordinate systems:

* **Cartesian**: Standard rectangular coordinates
* **Cylindrical**: (r, θ, z) coordinates
* **Spherical**: (r, θ, φ) coordinates

.. code-block:: yaml

    coordinate:
      type: cartesian  # or cylindrical, spherical

Time Integration
----------------

Configure time stepping and integration schemes:

.. code-block:: yaml

    time:
      cfl: 0.8              # CFL number (0 < CFL < 1)
      tlim: 100.0           # Simulation end time
      nlim: 10000           # Maximum number of steps
      integrator: rk3       # Integration scheme (rk2, rk3, rk4)

Hydrodynamics
-------------

Hydrodynamic solver configuration:

.. code-block:: yaml

    hydro:
      # Equation of state
      gamma: 1.4            # Adiabatic index
      eos: ideal            # EOS type (ideal, real)

      # Riemann solver
      riemann: hllc         # Solver type (hllc, roe, hll)

      # Reconstruction
      reconstruction: plm    # Scheme (plm, ppm, weno)

      # Flux calculation
      disable_flux_x1: false
      disable_flux_x2: false
      disable_flux_x3: false

Physical Forcing
----------------

Add gravitational and Coriolis forcing:

.. code-block:: yaml

    hydro:
      gravity:
        enabled: true
        g: 9.81             # Gravitational acceleration
        direction: -1       # Direction (-1 for downward in x1)

      coriolis:
        enabled: true
        omega: 7.27e-5      # Rotation rate (rad/s)
        latitude: 45.0      # Latitude (degrees)

Add constant isotropic viscosity and heat conduction on a Cartesian mesh:

.. code-block:: yaml

    forcing:
      diffusion:
        nu_iso: 0.0         # Kinematic viscosity, or dynamic viscosity mu if dynamic
        kappa_iso: 0.0      # Thermal diffusivity (length^2 / time), or conductivity k if dynamic
        dynamic: false      # Read the two above as DYNAMIC coefficients
        nu_scale_x1:        # optional x1 profile multiplying nu_iso
          x1: [0., 5.e4, 1.e5]
          scale: [1., 10., 100.]
        kappa_scale_x1:     # optional x1 profile multiplying kappa_iso
          x1: [0., 1.e5]
          scale: [1., 10.]

Diffusion is integrated explicitly and contributes a parabolic time-step
limit, which uses the largest scale of each profile. Curved coordinates and
anisotropic coefficients are not supported.

``nu_scale_x1`` and ``kappa_scale_x1`` multiply the kinematic coefficients by
a profile in the x1 coordinate: a table of x1 knots (strictly increasing, at
least two) and scale values (finite, strictly positive), linear between knots
and held beyond the ends, interpolated onto each block's cell centres. The
table works with any number of blocks along x1. A profile is refused with
``dynamic: true``. From Python the same table is the ``(2, n)`` tensor
``nu_scale_x1_table``; a per-cell tensor ``nu_scale_x1`` (one value per x1
cell of the block, ghosts included) is accepted only when a single block
spans x1. Either must be set before the MeshBlock is built.

With ``dynamic: false`` (the default) the coefficients are kinematic: the
viscous flux carries the face-averaged density and heat conduction uses the
energy flux ``-rho * cv * kappa_iso * grad(T)``, where ``cv`` is the local
mixture specific heat supplied by the equation of state. An EOS without a
positive reference specific heat at constant volume cannot enable heat
conduction.

With ``dynamic: true`` the same two numbers are read as the dynamic viscosity
``mu`` and the conductivity ``k``: the fluxes are ``-mu * stress`` and
``-k * grad(T)`` with no face density, and the time-step bound uses
``mu / rho_min`` and ``k / (rho_min * cv)``.

Implicit Correction
-------------------

To treat the vertical (x1) direction implicitly, which removes its acoustic
time-step bound, set an implicit scheme under ``integration``:

.. code-block:: yaml

    integration:
      implicit-scheme: 1          # 0 none, 1 vic-partial, 9 vic-full
      implicit-advection-cfl: 1.0 # optional, > 0: advective bound in x1
      shear-cfl: 0.0              # optional, >= 0; 0 switches it off

See :class:`~snapy.ImplicitOptions` for what each key does.

hevi-ark2 (vic-full under rk3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``implicit-scheme: 9`` and ``type: rk3`` and no ``implicit-integrator`` key, the
step is the ARS(2,3,2) HEVI step ``hevi-ark2``: the horizontal tendency explicit, the
vertical one linearly implicit, each implicit stage solved by corrections
``D <- D + (I + g dt J)^-1 (g dt V(R + D) - D)``. A one-line notice is printed when this
default applies. ``implicit-integrator: rk`` keeps the previous rk path.

.. code-block:: yaml

    integration:
      type: rk3
      implicit-scheme: 9
      # implicit-integrator: rk            # opt out: the previous rk path
      # implicit-integrator-iterations: 5  # stage solves (default 5, fixed)
      # implicit-integrator-tolerance: 0   # > 0: stop on a tolerance (4 to 8
      #                                    # solves, and when the correction
      #                                    # stops contracting); 0 or absent: fixed

Limits
^^^^^^

Each stage solve inverts a first-order, three-point Roe linearisation of the vertical
operator against the full high-order residual, so the stage iteration is a defect
correction: it contracts at a rate that depends on the vertical acoustic Courant number,
not to a fixed tolerance. The default of 5 solves was chosen against the converged ARS
stage (implicit-integrator-iterations 20) in the cp5 linear model of the 22 wind and box
arms at dt0, 2 dt0 and 3 dt0 (growth minus converged ARS, worst horizontal mode):

1. At dt0, 5 solves match converged ARS on all 22 arms.
2. Above dt0, on the dx 400 km boxes, 5 solves grow weakly: +0.0067 per day at 2 dt0, an
   e-folding time of about 150 days.
3. Converged ARS itself grows at grid scale at 3 dt0 on the LMARS u 400 m/s boxes with
   dx 200 km and dx 400 km (+8.76 and +7.16 per day, m 32); the rk path does not. No
   number of solves removes this growth.
4. Near-floor coarse columns at large vertical Courant number redo (retry the step at a
   halved dt) under the default and under the rk path alike. On the coarse
   face-work column of ``tests/test_implicit_stratified_solid.py`` (45 levels over 40
   scale heights, dt 1500 s, vertical acoustic Courant 27.8, balanced to rtol 1e-12), run
   for 60,000 s with redos as dt halvings: the rk path first redoes at step 33 (corrected-PE
   switch on) and step 34 (off), with 25 and 40 rejected tries and max\|w\| 494 and
   437 m/s; 5 solves first redo at step 12 and step 19, with 138 and 80 rejected tries and
   max\|w\| 1042 and 671 m/s. Every run reaches 60,000 s; none aborts. The test cases are
   that column (ctest ``test_implicit_stratified_solid_coarse_face_re0`` and ``_re1``) and
   ``tests/test_hevi_coarse_column.py``; both pin that a redo restores the step input.

.. warning::

   Raising ``implicit-integrator-iterations`` does not converge monotonically, so it is
   not a tuning knob. In the linear model the number of cells out of 64 whose growth
   exceeds converged ARS by more than 1e-4 per day is 31, 7, 7, 22, 0, 3, 0 and 0 at 4,
   5, 6, 8, 10, 12, 16 and 20 solves; 4 solves run the
   coarse column above without a redo but grow on the wind arms (+0.35 per day on HLLC
   u 250 m/s at 2 dt0, against -0.43 at 5).

Output Configuration
--------------------

Control simulation output:

.. code-block:: yaml

    output:
      # Basic output settings
      dt: 0.1               # Output interval
      format: netcdf        # Output format (netcdf, hdf5)

      # Variables to output
      variables:
        - density
        - velocity
        - pressure
        - temperature

      # Output file settings
      basename: simulation
      directory: output/

Advanced Output
~~~~~~~~~~~~~~~

Configure multiple output streams:

.. code-block:: yaml

    output:
      streams:
        - id: 1
          dt: 0.1
          variables: [density, velocity, pressure]

        - id: 2
          dt: 1.0
          variables: [temperature, energy]

        - id: 3
          dt: 10.0
          variables: [diagnostics]

Loading Configuration
---------------------

Load configuration in Python:

.. code-block:: python

    import snapy

    # Load from YAML file
    options = snapy.MeshBlockOptions.from_yaml("config.yaml")

    # Create mesh block
    block = snapy.MeshBlock(options)

You can also configure individual components:

.. code-block:: python

    # Configure hydrodynamics
    hydro_opts = snapy.HydroOptions.from_yaml("config.yaml")
    hydro_opts.riemann().type("hllc")
    hydro_opts.eos().gamma(1.4)

    # Configure EOS separately
    eos_opts = snapy.EquationOfStateOptions()
    eos_opts.type("ideal")
    eos_opts.gamma(1.4)
    hydro_opts.eos(eos_opts)
