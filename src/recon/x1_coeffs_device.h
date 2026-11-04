#pragma once

// base
#include <configure.h>

//! \file x1_coeffs_device.h
//! The two route A row widths, in a header with no torch dependency so that
//! the device kernels (`interp_impl_nu.cuh`) can see them without pulling in
//! <torch/torch.h>.  `x1_coeffs.hpp` includes this, so there is one
//! definition.  The meaning of every column is in x1_coeffs.hpp and in
//! docs/derivations/280-route-a-layout.md.

namespace snap {

//! columns of one per-face WENO5 row: 9*5 candidate/smoothness + 3 linear
constexpr int kWeno5NuCoeff = 48;
//! columns of one per-face cp5 row
constexpr int kPoly5NuCoeff = 5;

//! const-correct twin of `_vvdot` (interp_impl.h).
/*!
 * The coefficient row is read-only on the route A path: it points into the
 * module's registered buffer, which `left()`/`right()` must not modify.
 *
 * ONE definition for both backends.  The host and the device kernel used to
 * carry a copy each, differing only in the `__device__` qualifier
 * (snapy#280 review item 4); `DISPATCH_MACRO` is that qualifier under nvcc
 * and empty otherwise, so this header gives each translation unit the form it
 * needs.
 */
template <int N, typename T>
DISPATCH_MACRO inline T _vvdot_c(T const* v1, T const* v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

}  // namespace snap
