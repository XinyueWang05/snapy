#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! The x1 grid class the reference kernel is handed. A bool `uniform` (true:
//! the six-face rows; false: the log-mean) converts to the first two; a keyed
//! stretched x1 (#280) takes the log-mean in its log1p form.
enum X1RefGrid : int {
  kX1RefNonUniform = 0,
  kX1RefUniform = 1,
  kX1RefStretched = 2
};

}  // namespace snap

namespace at::native {

using hydro_ref_x1_fn = void (*)(
    torch::Tensor const& w, torch::Tensor const& dx1f,
    torch::Tensor const& anchor, torch::Tensor const& psf_lo,
    torch::Tensor const& psf_hi, torch::Tensor const& pref,
    torch::Tensor const& dsf, torch::Tensor const& dref, int iu, double grav,
    int x1_grid, bool phys_in, bool phys_out, bool wall_clamp);

DECLARE_DISPATCH(hydro_ref_x1_fn, call_hydro_ref_x1);

}  // namespace at::native
