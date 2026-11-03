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
//! FACE / WINDOW CONVENTION (read off the code at 2d9fa419, do not change
//! without re-reading these lines):
//!
//! * `src/coord/cartesian.cpp:26-27` makes face `k` the LOWER face of cell `k`,
//!   so a block with `nc1` cells has `nc1 + 1` faces.
//! * `src/recon/interpolation.hpp:64-76` gives the interpolator an output of
//!   `nc1 - 4` elements; output element `m` is the 5-cell window `m .. m+4`,
//!   bounded by the six faces `m .. m+5`.
//! * `src/recon/reconstruct.cpp:54-55` writes `left()` into `wlr[IRT]` at index
//!   `il-1+m = m+2` and `right()` into `wlr[ILT]` at index `il+m = m+3`.
//!
//! So the two sides of one output element target TWO DIFFERENT faces out of the
//! SAME five cells: they are the lower and the upper face of the window's
//! central cell `m+2`.
//!
//!   row m of `cm`  ->  target face m+2  (the IRT state, written by `left()`)
//!   row m of `cp`  ->  target face m+3  (the ILT state, written by `right()`)
//!
//! On a uniform grid `cp = cm.flip` hides this, which is why a face-index
//! off-by-one is invisible to every uniform test.
//!
//! WENO5 ROW LAYOUT, 48 columns.  Columns 0-44 are the nine 5-entry rows the
//! kernel walks at `src/recon/interp_impl.h:77-85`, in that order:
//!
//!   cols  0- 4  c1  candidate p_0   (linear weight d_0)
//!   cols  5- 9  c2  candidate p_1   (linear weight d_1)
//!   cols 10-14  c3  candidate p_2   (linear weight d_2)
//!   cols 15-19  c4  A_0  | beta_0 = 13/12 (c4.u)^2 + 1/4 (c5.u)^2
//!   cols 20-24  c5  B_0  |
//!   cols 25-29  c6  A_1  | beta_1
//!   cols 30-34  c7  B_1  |
//!   cols 35-39  c8  A_2  | beta_2
//!   cols 40-44  c9  B_2  |
//!   col  45     d_0 | the three linear weights, which replace the literals
//!   col  46     d_1 | .3/.6/.1 at `src/recon/interp_impl.h:115-117`
//!   col  47     d_2 |
//!
//! Within a 5-entry row the index is the window cell offset n = 0..4, i.e. the
//! coefficient of the cell average of block cell `m + n`.
//!
//! 45 vs 48: 45 is the shipped coefficient count (`kWeno5Coeff`,
//! `src/recon/interp_impl.cuh:221`).  On a stretched grid the three optimal
//! linear weights are face dependent too, so route A carries them in the same
//! row: 45 + 3 = 48, exactly as the design note Sec. 4.2 "Storage" says.
//!
//! The smoothness rows ARE pre-scaled: c4..c9 store A_r and B_r divided by
//! sqrt(13/12) and sqrt(1/4) respectively, so the kernel's own 13/12 and 1/4
//! multipliers reproduce the Jiang & Shu indicator
//!   beta_r = sum_{m=1,2} h_i^{2m-1} int_{I_i} (p_r^{(m)})^2 dx
//! unchanged.  beta is therefore dimensionless on any grid and the shipped
//! absolute eps = 1e-6 keeps its uniform-grid meaning (note Sec. 4.2).

namespace snap {

// kWeno5NuCoeff / kPoly5NuCoeff live in x1_coeffs_device.h, included above.

//! Per-face WENO5 coefficient rows built from a face vector.
/*!
 * \param[in] x1f  1-D, strictly increasing, at least 6 faces
 * \return         {cm, cp}, each (x1f.numel() - 5, 48), float64, on the CPU.
 *                 Row m belongs to the 5-cell window bounded by
 *                 x1f[m .. m+5]; cm targets x1f[m+2], cp targets x1f[m+3].
 */
std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f);

//! Per-face cp5 coefficient rows built from a face vector.
/*!
 * \param[in] x1f  1-D, strictly increasing, at least 6 faces
 * \return         {cm, cp}, each (x1f.numel() - 5, 5), float64, on the CPU.
 *                 Same face convention as weno5_x1_coeffs.  Each row is
 *                 sum_r d_r p_r of the WENO5 row above (note Sec. 4.2), which
 *                 is the degree-4 (5-cell) reconstruction at the target face.
 */
std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(torch::Tensor const& x1f);

//! The coefficient table in the state's dtype and device, cached.
/*!
 * Buffers follow `module.to(...)`, so the fast path is a plain reference.  A
 * mismatch is only possible when the state was converted without the module;
 * then the cast is built once and reused, the same pattern as the `x1v`/`dx1f`
 * casts at `src/hydro/hydro_forward.cpp:425-426`.
 */
torch::Tensor const& x1_coeffs_for(torch::Tensor const& src,
                                   torch::Tensor& cache,
                                   torch::Tensor const& w);

//! A zero-stride view of `c` whose element at face f is `c[f][0]`.
/*!
 * Added as a third `TensorIterator` input, it makes the CPU kernel's row
 * pointer come out of the iterator's own offset arithmetic instead of a loop
 * counter.  Every dimension has stride 0 except `dim`, where one step advances
 * a whole row, so the view costs no memory.
 */
torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim);

//! The `TensorIterator` of one route A `left()`/`right()` call.
/*!
 * Same configuration as the shipped `Weno5InterpImpl::left`
 * (src/recon/weno5.cpp:51-57) plus, ON THE CPU ONLY, a third operand: the
 * zero-stride row view above.
 *
 * WHY THE THIRD OPERAND (snapy#280 review item (b)).  `call_weno5_cpu` walks
 * its sub-range with a plain counter (src/recon/recon_dispatch.cpp:114-123).
 * `iter.for_each` hands out sub-ranges by grain size, a sub-range may begin
 * part-way along a line, and the iterator is free to reorder and coalesce
 * dimensions, so that counter is NOT the x1 index.  With the row view as an
 * operand the kernel instead reads `data[2] + i * strides[2]`, which
 * `TensorIterator` computes with the SAME offset arithmetic it uses for the
 * state.  That is correct for any start, any order and any thread count.
 *
 * It also pins the iteration structure: coalescing dimension `k` into `dim`
 * would need `stride[dim] == stride[k] * shape[k]`, i.e. `NCoeff == 0`, so the
 * x1 dimension can no longer be merged into another one.
 *
 * On CUDA the iterator keeps two operands and `dim` stays squashed: there the
 * kernel recovers the face index from `threadIdx.x`, which is exact because
 * `stencil_kernel` gives one block to one whole line.
 */
at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c);

//! The coefficient table must have one contiguous row per output element.
/*!
 * Shared by the CPU and the CUDA dispatch so that both refuse the same
 * malformed table with the same message.
 */
void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what);

}  // namespace snap
