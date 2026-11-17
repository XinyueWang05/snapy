#pragma once
// V3 lane I skeleton (declarations only; not part of any build).
// Design: docs/v3/design/I.rst s4. Replaces src/hydro/gravity_work.hpp and
// src/hydro/gravity_work_radial.hpp, and the three caches of the same factors
// (HydroImpl::_pe_work, ImplicitHydroImpl::_rx_work, _g_work_lo/_g_work_hi).

#include <string>

#include <torch/torch.h>

namespace snap {

//! as at the pin (gravity_work_radial.hpp:13-25)
enum class X1Measure { none, plain, radial, radial_mid };
X1Measure x1_measure(std::string const& coordinate_type);

//! as at the pin (gravity_work_radial.hpp:120-149): grid, range, gravity
struct X1GridKey {
  torch::Tensor x1f, x1v;
  int64_t vf = -1, vv = -1;
  int is = 0, ie = 0, lo = 0, hi = 0;
  double grav1 = 0.;
  X1Measure measure = X1Measure::none;
  bool operator==(X1GridKey const& o) const;
};

//! the 3-point centroid slope as a stencil (gravity_work_radial.hpp:49-68)
struct CentroidSlopeStencil {
  torch::Tensor idx[3], wt[3];
  torch::Tensor operator()(torch::Tensor const& q) const;
  torch::Tensor band(int off) const;  //!< one diagonal of the slope matrix
};

//! The coefficients of G, laid out for the column operator. They are the
//! tensors apply() uses, so the operator's energy row is dG/dM times the
//! mass row of the flux Jacobian by construction (I.rst s4).
struct GravityWorkRows {
  //! W: g1 (x_l - x_i) / V_i and g1 (x_u - x_i) / V_i, [nc1] (interior)
  torch::Tensor face_lo, face_hi;
  //! K on d rho: three bands of g1 sigma^2 s (plus g1 (r_c - x1v) on the
  //! diagonal, radial_mid), [nc1]; undefined when measure == none
  torch::Tensor k_lo, k_mid, k_hi;
  //! the one-sided wall slopes' third points, solved by Woodbury in-kernel
  torch::Tensor k_t0, k_t1;
};

//! G[M] = W[M] + K[-div M / V]: the x1 gravity work, per unit volume, of the
//! x1 mass M moved through faces [is, ie] in a stage. One form for every
//! grid and for the explicit and implicit paths. Linear in M. Geometry is
//! differenced before the cast to the working dtype (FP32-safe at large r).
class GravityWork {
 public:
  GravityWork() = default;
  GravityWork(double grav1, X1Measure measure);

  //! rebuild the grid-only factors when key, dtype or device changed
  void update(X1GridKey const& key, torch::TensorOptions const& opts);

  //! G[M]; face_mass [..., nc1+1] in kg (area- and dt-integrated)
  torch::Tensor apply(torch::Tensor const& face_mass) const;

  //! G[M] of a block next to an internal x1 seam: drho carries the
  //! neighbour's value one cell past it (#303), so the seam cell takes the
  //! centred slope (explicit path only; the solve refuses an x1 split)
  torch::Tensor apply_seam(torch::Tensor const& face_mass,
                           torch::Tensor const& drho_ghosts, bool seam_lo,
                           bool seam_hi) const;

  //! the same coefficients for the operator, per solid depth (#369 frames)
  GravityWorkRows const& rows(int depth = 0) const;

  //! the corrected potential energy P = rho phi(r_c) - g1 sigma^2 drho/dx1,
  //! for the E + P diagnostics (meshblock.cpp:1270-1282)
  torch::Tensor potential(torch::Tensor const& rho, bool seam_lo,
                          bool seam_hi) const;

  X1Measure measure() const;

 private:
  double _grav1 = 0.;
  X1Measure _measure = X1Measure::none;
  X1GridKey _key;
  CentroidSlopeStencil _slope;
  torch::Tensor _gvar, _goff;      //!< g1 sigma^2, g1 (r_c - x1v)
  std::vector<GravityWorkRows> _rows;  //!< [0]: no solid; [d]: depth d
};

}  // namespace snap
