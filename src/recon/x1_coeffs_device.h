#pragma once

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

}  // namespace snap
