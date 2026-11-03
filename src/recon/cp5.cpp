// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"
#include "x1_coeffs.hpp"

namespace snap {

void Center5InterpImpl::reset() {
  cm = register_buffer("cm", torch::tensor({-1. / 20., 9. / 20., 47. / 60.,
                                            -13. / 60., 1. / 30.},
                                           torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));

  // Route A tables (snapy#280 stage 1); see Weno5InterpImpl::reset for why
  // they are registered empty here rather than created on demand.
  cm_x1 = register_buffer("cm_x1",
                          torch::zeros({0, kPoly5NuCoeff}, torch::kFloat64));
  cp_x1 = register_buffer("cp_x1",
                          torch::zeros({0, kPoly5NuCoeff}, torch::kFloat64));
}

void Center5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  auto rows = cp5_x1_coeffs(x1f);

  auto opts = cm.options();
  cm_x1.set_data(rows.first.to(opts).contiguous());
  cp_x1.set_data(rows.second.to(opts).contiguous());
}

void Center5InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cm, dim);
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cp, dim);
}
}  // namespace snap
