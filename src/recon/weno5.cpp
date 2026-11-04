// torch
#include <ATen/TensorIterator.h>

// snap
#include "interpolation.hpp"
#include "recon_dispatch.hpp"
#include "recon_dispatch_nu.hpp"
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

  // Route A tables: registered only once a face vector arrives, so a uniform
  // deck's buffer list is the pre-#280 one (x1_coeffs.hpp, review item 2).
  x1_reset_tables(*this, cm_x1, cp_x1, kWeno5NuCoeff);
}

void Weno5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  x1_set_coeffs(*this, x1f, &weno5_x1_coeffs);
}

//! Route A: run the per-face kernel with `src` as the coefficient table.
/*!
 * Factored out so that `left()` and `right()` differ only in which table they
 * pass, exactly as they differ only in `cm` vs `cp` on the uniform path.  The
 * refusals and the iterator are shared with cp5 (x1_coeffs.hpp).
 */
void Weno5InterpImpl::forward_nu(torch::Tensor const& w, int dim,
                                 torch::Tensor const& out,
                                 torch::Tensor const& src,
                                 torch::Tensor& cache) {
  x1_forward_nu(w, dim, out, src, cache, "weno5",
                [&](at::TensorIterator& iter, torch::Tensor const& c) {
                  at::native::call_weno5_nu(out.device().type(), iter, c, dim,
                                            options->scale());
                });
}

void Weno5InterpImpl::left(torch::Tensor w, int dim, torch::Tensor const& out) {
  // route A (snapy#280): only when a face vector has been installed.  With no
  // table the function below is byte for byte the shipped one.
  if (has_x1_coeffs()) {
    forward_nu(w, dim, out, cm_x1, cm_x1_cast);
    return;
  }

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
  if (has_x1_coeffs()) {
    forward_nu(w, dim, out, cp_x1, cp_x1_cast);
    return;
  }

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
