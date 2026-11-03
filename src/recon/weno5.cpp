// torch
#include <ATen/TensorIterator.h>

// snap
#include "interpolation.hpp"
#include "recon_dispatch.hpp"
#include "x1_coeffs.hpp"

namespace snap {

void Weno5InterpImpl::reset() {
  cm = register_buffer("cm",
                       torch::tensor({{-1. / 6., 5. / 6., 1. / 3., 0., 0.},
                                      {0., 1. / 3., 5. / 6., -1. / 6., 0.},
                                      {0., 0., 11. / 6., -7. / 6., 1. / 3.},
                                      {1., -2., 1., 0., 0.},
                                      {1., -4., 3., 0., 0.},
                                      {0., 1., -2., 1., 0.},
                                      {0., -1., 0., 1., 0.},
                                      {0., 0., 1., -2., 1.},
                                      {0., 0., 3., -4., 1.}},
                                     torch::kFloat64));

  cp = register_buffer("cp", cm.flip({1}));

  // Route A tables (snapy#280 stage 1).  Registered empty here so that the
  // buffer count is stable for torch::nn::Cloneable::clone() and a harness can
  // find the names before they are filled.  Stage A only builds and exposes
  // them; left()/right() below are the shipped uniform path, untouched.
  cm_x1 = register_buffer("cm_x1",
                          torch::zeros({0, kWeno5NuCoeff}, torch::kFloat64));
  cp_x1 = register_buffer("cp_x1",
                          torch::zeros({0, kWeno5NuCoeff}, torch::kFloat64));
}

void Weno5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  auto rows = weno5_x1_coeffs(x1f);

  // follow whatever dtype and device the module was moved to, as cm/cp do
  auto opts = cm.options();
  cm_x1.set_data(rows.first.to(opts).contiguous());
  cp_x1.set_data(rows.second.to(opts).contiguous());
}

void Weno5InterpImpl::left(torch::Tensor w, int dim, torch::Tensor const& out) {
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

  at::native::call_weno5(out.device().type(), iter, cm, dim, options->scale());
}

void Weno5InterpImpl::right(torch::Tensor w, int dim,
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

  at::native::call_weno5(out.device().type(), iter, cp, dim, options->scale());
}

}  // namespace snap
