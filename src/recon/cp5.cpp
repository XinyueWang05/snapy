// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"
#include "recon_dispatch_nu.hpp"
#include "x1_coeffs.hpp"

namespace snap {

void Center5InterpImpl::reset() {
  cm = register_buffer("cm", torch::tensor({-1. / 20., 9. / 20., 47. / 60.,
                                            -13. / 60., 1. / 30.},
                                           torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));

  // Route A tables (snapy#280 stage 1); see Weno5InterpImpl::reset for why
  // they are registered in set_x1_coeffs rather than here.
  x1_reset_tables(*this, cm_x1, cp_x1, kPoly5NuCoeff);
}

void Center5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  auto rows = cp5_x1_coeffs(x1f);

  auto opts = cm.options();
  x1_install_tables(*this, cm_x1, cp_x1, rows.first.to(opts).contiguous(),
                    rows.second.to(opts).contiguous());

  // the dtype/device casts belong to the old table
  cm_x1_cast = torch::Tensor();
  cp_x1_cast = torch::Tensor();
}

//! Route A: run the per-face kernel with `src` as the coefficient table.
void Center5InterpImpl::forward_nu(torch::Tensor const& w, int dim,
                                   torch::Tensor const& out,
                                   torch::Tensor const& src,
                                   torch::Tensor& cache) {
  // see Weno5InterpImpl::forward_nu: a size test cannot tell x2 from x1 on a
  // block with nc2 == nc1, so the dimension itself is checked
  TORCH_CHECK(dim == out.dim() - 1,
              "[cp5] route A covers the x1 dimension only, so a module "
              "holding per-face x1 coefficients must be given the last "
              "dimension of a (nvar, nc3, nc2, nc1) state; got dim ", dim,
              " of a ", out.dim(), "-dimensional output (note Sec. 4.7)");

  TORCH_CHECK(out.size(dim) == src.size(0),
              "[cp5] per-face x1 coefficients are installed but dimension ",
              dim, " has ", out.size(dim), " outputs while the table has ",
              src.size(0),
              " rows; route A covers the x1 dimension only, so `recon23` must "
              "not be given a face vector (note Sec. 4.7)");

  auto const& c = x1_coeffs_for(src, cache, w);
  auto iter = x1_nu_iterator(w, dim, out, c);
  at::native::call_poly5_nu(out.device().type(), iter, c, dim);
}

void Center5InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor const& out) {
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

  at::native::call_poly5(out.device().type(), iter, cm, dim);
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
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

  at::native::call_poly5(out.device().type(), iter, cp, dim);
}
}  // namespace snap
