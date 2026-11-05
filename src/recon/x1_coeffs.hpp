#pragma once

// C/C++
#include <utility>

// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

// snap
#include "x1_coeffs_device.h"

//! \file x1_coeffs.hpp
//! Route A per-face WENO5 / cp5 coefficients for a non-uniform x1 grid
//! (snapy#280 stage 1, design note Sec. 4.2-4.3).
//!
//! Face and window convention, the 48-column WENO5 row layout, why 48 and not
//! the shipped 45, and what the smoothness rows c4..c9 hold (undivided, so the
//! kernel's own 13/12 and 1/4 keep eps = 1e-6 meaningful): design note
//! Sec. 4.2 and docs/derivations/280-route-a-layout.md Sec. 1-2.
//!
//!   row m of `cm`  ->  target face m+2  (the IRT state, written by `left()`)
//!   row m of `cp`  ->  target face m+3  (the ILT state, written by `right()`)

namespace snap {

// kWeno5NuCoeff / kPoly5NuCoeff live in x1_coeffs_device.h, included above.

//! Per-face WENO5 rows from a 1-D increasing face vector (note Sec. 4.2):
//! {cm, cp}, each (x1f.numel() - 5, 48), float64, CPU; cm targets x1f[m+2].
std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f);

//! Per-face cp5 rows, same convention, (x1f.numel() - 5, 5): the sum_r d_r p_r
//! of the WENO5 row above (note Sec. 4.2, layout note Sec. 3).
std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(torch::Tensor const& x1f);

//! Put the route A tables back in the state a fresh `reset()` leaves them in;
//! not a no-op, a clone's non-empty tables are re-registered here (review
//! item 2, torch cloneable.h).  `ncoeff` is kWeno5NuCoeff or kPoly5NuCoeff.
void x1_reset_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                     torch::Tensor& cp_x1, int ncoeff);

//! Install one pair of tables, registering the buffers on first use.
void x1_install_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                       torch::Tensor& cp_x1, torch::Tensor const& cm_rows,
                       torch::Tensor const& cp_rows);

//! The body of `set_x1_coeffs`: weno5 and cp5 differ only in `build`, the
//! builder that turns the faces into rows (snapy#280 review item 4).
template <typename Self, typename Builder>
void x1_set_coeffs(Self& self, torch::Tensor const& x1f, Builder build) {
  auto rows = build(x1f);

  // follow whatever dtype and device the module was moved to, as cm/cp do
  auto opts = self.cm.options();
  x1_install_tables(self, self.cm_x1, self.cp_x1,
                    rows.first.to(opts).contiguous(),
                    rows.second.to(opts).contiguous());

  // the dtype/device casts belong to the old table
  self.cm_x1_cast = torch::Tensor();
  self.cp_x1_cast = torch::Tensor();
}

//! The coefficient table in the state's dtype and device, cached; same cast
//! pattern as `x1v`/`dx1f` at `src/hydro/hydro_forward.cpp:426-427`.
torch::Tensor const& x1_coeffs_for(torch::Tensor const& src,
                                   torch::Tensor& cache,
                                   torch::Tensor const& w);

//! A zero-stride view of `c` whose element at face f is `c[f][0]`, so the CPU
//! row pointer comes out of the iterator's own offset arithmetic; costs no
//! memory (layout note Sec. 8.2).
torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim);

//! The `TensorIterator` of one route A `left()`/`right()` call: the shipped
//! `Weno5InterpImpl::left` configuration (src/recon/weno5.cpp:66-72) plus, ON
//! THE CPU ONLY, the zero-stride row view above as a third operand, because
//! the CPU loop counter is not the face index while CUDA's `threadIdx.x` is
//! (snapy#280 review item (b); layout note Sec. 8.1-8.2).
at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c);

//! The coefficient table must have one contiguous row per output element;
//! shared by both dispatches so they refuse a malformed table alike.
void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what);

//! The body of `forward_nu`: route A's two refusals, then the kernel.  `src`
//! is `cm_x1` or `cp_x1`, `cache` its dtype/device cast, `what` "weno5" or
//! "cp5" for the messages, `call` the backend kernel on the built iterator.
template <typename Call>
void x1_forward_nu(torch::Tensor const& w, int dim, torch::Tensor const& out,
                   torch::Tensor const& src, torch::Tensor& cache,
                   char const* what, Call call) {
  // x1 is the LAST dimension everywhere in snap; the row-count test below
  // cannot stand in for this one when nc2 == nc1 (note Sec. 4.7).
  TORCH_CHECK(dim == out.dim() - 1, "[", what,
              "] route A covers the x1 dimension only, so a module holding "
              "per-face x1 coefficients must be given the last dimension of "
              "a (nvar, nc3, nc2, nc1) state; got dim ", dim, " of a ",
              out.dim(), "-dimensional output (note Sec. 4.7)");

  TORCH_CHECK(out.size(dim) == src.size(0), "[", what,
              "] per-face x1 coefficients are installed but dimension ", dim,
              " has ", out.size(dim), " outputs while the table has ",
              src.size(0),
              " rows; route A covers the x1 dimension only, so `recon23` must "
              "not be given a face vector (note Sec. 4.7)");

  auto const& c = x1_coeffs_for(src, cache, w);
  auto iter = x1_nu_iterator(w, dim, out, c);
  call(iter, c);
}


}  // namespace snap
