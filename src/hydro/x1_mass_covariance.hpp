#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! SNAP_X1_MASS_COVARIANCE (HydroImpl::x1_mass_covariance): the x1
//! reconstruction treats the cell velocity m1/rho as the cell average of w; in
//! a stratified column that adds dz^2/12 rho_z w_z to the face mass flux.
//! hydro_forward.cpp subtracts it from the reconstructed velocity (the x1
//! analogue of SNAP_FLUX_COVARIANCE), with the x1 derivatives below.
//! `stretched` is the grid key (CoordinateOptions::stretched_x1, #280), not a
//! probe of the widths, so every block of a keyed grid applies one operator.

//! centred x1 derivative over x1v for every cell that has both neighbours; the
//! first and last array cells copy their neighbour's value
inline torch::Tensor x1_cov_d1_centred(torch::Tensor const& a,
                                       torch::Tensor const& x1v,
                                       bool stretched) {
  int n1 = a.size(-1);
  auto d = torch::empty_like(a);
  if (stretched) {
    // #280: the slope at x1v_i of the quadratic through cells i-1, i, i+1
    auto hm = x1v.narrow(0, 1, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
    auto hp = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 1, n1 - 2);
    d.narrow(-1, 1, n1 - 2)
        .copy_(-hp / (hm * (hm + hp)) * a.narrow(-1, 0, n1 - 2) +
               (hp - hm) / (hm * hp) * a.narrow(-1, 1, n1 - 2) +
               hm / (hp * (hm + hp)) * a.narrow(-1, 2, n1 - 2));
  } else {
    auto dx = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
    d.narrow(-1, 1, n1 - 2)
        .copy_((a.narrow(-1, 2, n1 - 2) - a.narrow(-1, 0, n1 - 2)) / dx);
  }
  d.select(-1, 0).copy_(d.select(-1, 1));
  d.select(-1, n1 - 1).copy_(d.select(-1, n1 - 2));
  return d;
}

//! d a / d x1 at cell i from cells i, i + s, i + 2 s (s = +1 inward from the
//! lower wall, -1 from the upper), second order; next to a reflecting wall the
//! even ghost density is not the stratified column
inline torch::Tensor x1_cov_one_sided(torch::Tensor const& a,
                                      torch::Tensor const& x1v, int i, int s,
                                      bool stretched) {
  if (stretched) {
    // #280: the slope at x1v_i of the quadratic through the three cells
    auto p = x1v[i + s] - x1v[i], q = x1v[i + 2 * s] - x1v[i];
    return -(p + q) / (p * q) * a.select(-1, i) +
           q / (p * (q - p)) * a.select(-1, i + s) -
           p / (q * (q - p)) * a.select(-1, i + 2 * s);
  }
  auto h = x1v[i + s] - x1v[i];
  return s *
         (-3. * a.select(-1, i) + 4. * a.select(-1, i + s) -
          a.select(-1, i + 2 * s)) /
         (2. * s * h);
}

}  // namespace snap
