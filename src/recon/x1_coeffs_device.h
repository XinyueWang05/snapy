#pragma once

// base
#include <configure.h>

//! \file x1_coeffs_device.h
//! The two route A row widths, in a torch-free header so the device kernels
//! can see them; column meanings in docs/derivations/280-route-a-layout.md.

namespace snap {

//! columns of one per-face WENO5 row: 9*5 candidate/smoothness + 3 linear
constexpr int kWeno5NuCoeff = 48;
//! columns of one per-face cp5 row
constexpr int kPoly5NuCoeff = 5;

//! const-correct twin of `_vvdot` (interp_impl.h), one definition for both
//! backends via `DISPATCH_MACRO` (snapy#280 review item 4).
template <int N, typename T>
DISPATCH_MACRO inline T _vvdot_c(T const* v1, T const* v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

}  // namespace snap
