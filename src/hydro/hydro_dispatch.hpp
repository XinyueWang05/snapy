#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! The x1 grid class the reference kernel is handed. A bool `uniform` (true:
//! the six-face rows; false: the log-mean) converts to the first two; a keyed
//! stretched x1 (#280) takes the six-face cell average with per-cell weights
//! (hydro_ref_x1_stretched_rows).
enum X1RefGrid : int {
  kX1RefNonUniform = 0,
  kX1RefUniform = 1,
  kX1RefStretched = 2
};

//! #280 (iv): the per-cell rows of the stretched-x1 cell pressure, built once
//! from the cell widths `dx1f` (nc1,): row i is (start, w_0..w_5), the cell
//! average over cell i of the degree-5 Lagrange interpolant through faces
//! start..start+5, the same faces the uniform branch uses (start = -1: the
//! two-face mean). Weights depend on widths only, so a block and
//! balance_column handed the same widths get the same rows. Returns (nc1, 7)
//! float64 on the CPU.
torch::Tensor hydro_ref_x1_stretched_rows(torch::Tensor const& dx1f, int iu,
                                          bool phys_in, bool phys_out,
                                          bool wall_clamp);

}  // namespace snap

namespace at::native {

using hydro_ref_x1_fn = void (*)(
    torch::Tensor const& w, torch::Tensor const& dx1f,
    torch::Tensor const& anchor, torch::Tensor const& psf_lo,
    torch::Tensor const& psf_hi, torch::Tensor const& pref,
    torch::Tensor const& dsf, torch::Tensor const& dref, int iu, double grav,
    int x1_grid, bool phys_in, bool phys_out, bool wall_clamp,
    torch::Tensor const& rows);

DECLARE_DISPATCH(hydro_ref_x1_fn, call_hydro_ref_x1);

}  // namespace at::native
