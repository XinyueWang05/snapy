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

  // Route A tables (snapy#280 stage 1): REGISTERED ONLY ONCE A FACE VECTOR
  // HAS BEEN INSTALLED, in set_x1_coeffs below.  A uniform deck therefore
  // keeps exactly the buffer list it had before #280, which matters because
  // meshblock.cpp copies every named buffer into the user-forcing input
  // dictionary.  They stay defined-but-empty so that `has_x1_coeffs()` and
  // the harnesses that read them need no null test.
  x1_reset_tables(*this, cm_x1, cp_x1, kWeno5NuCoeff);
}

void Weno5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  auto rows = weno5_x1_coeffs(x1f);

  // follow whatever dtype and device the module was moved to, as cm/cp do
  auto opts = cm.options();
  x1_install_tables(*this, cm_x1, cp_x1, rows.first.to(opts).contiguous(),
                    rows.second.to(opts).contiguous());

  // the dtype/device casts belong to the old table
  cm_x1_cast = torch::Tensor();
  cp_x1_cast = torch::Tensor();
}

//! Route A: run the per-face kernel with `src` as the coefficient table.
/*!
 * Factored out so that `left()` and `right()` differ only in which table they
 * pass, exactly as they differ only in `cm` vs `cp` on the uniform path.
 */
void Weno5InterpImpl::forward_nu(torch::Tensor const& w, int dim,
                                 torch::Tensor const& out,
                                 torch::Tensor const& src,
                                 torch::Tensor& cache) {
  // x1 is the LAST dimension of (nvar, nc3, nc2, nc1) everywhere in snap, so
  // this is the whole of "dim is x1".  The row-count test below cannot stand
  // in for it: on a block with nc2 == nc1 the x2 output has exactly as many
  // elements as the table has rows, and the x1 coefficients would then be
  // applied along x2 with no complaint (note Sec. 4.7).
  TORCH_CHECK(dim == out.dim() - 1,
              "[weno5] route A covers the x1 dimension only, so a module "
              "holding per-face x1 coefficients must be given the last "
              "dimension of a (nvar, nc3, nc2, nc1) state; got dim ", dim,
              " of a ", out.dim(), "-dimensional output (note Sec. 4.7)");

  TORCH_CHECK(out.size(dim) == src.size(0),
              "[weno5] per-face x1 coefficients are installed but dimension ",
              dim, " has ", out.size(dim), " outputs while the table has ",
              src.size(0),
              " rows; route A covers the x1 dimension only, so `recon23` must "
              "not be given a face vector (note Sec. 4.7)");

  auto const& c = x1_coeffs_for(src, cache, w);
  auto iter = x1_nu_iterator(w, dim, out, c);
  at::native::call_weno5_nu(out.device().type(), iter, c, dim,
                            options->scale());
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
