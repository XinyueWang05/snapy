#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

//! \file recon_dispatch_nu.hpp
//! Route A dispatch stubs (snapy#280 stage 1).  Deliberately a separate file
//! and separate stubs from `recon_dispatch.hpp`: the shipped stubs and their
//! `REGISTER_*_DISPATCH` lines stay byte identical, which is the structural
//! half of the G3 argument.
//!
//! `payload` is the per-face coefficient table, (nout, 48) for WENO5 and
//! (nout, 5) for cp5, in the state's dtype and on the state's device, with row
//! m belonging to output element m.
//!
//! The two backends recover the face index differently, and both are exact:
//!  * CUDA -- `stencil_kernel` gives one block to one whole line with
//!    `blockDim.x == nc1` and `grid.x == 1` (src/utils/loops.cuh:91-108), so
//!    `threadIdx.x` IS the output index; the tiled launcher keeps `grid.x == 1`
//!    (src/recon/recon_dispatch.cu:82-83) and uses `start + threadIdx.x`.
//!  * CPU  -- `iter.for_each` hands out sub-ranges that may begin mid-line and
//!    does not track the x1 index (src/recon/recon_dispatch.cpp:114-123), so
//!    the loop counter is NOT the face index.  The iterator therefore carries
//!    a third operand, a zero-stride view of the table whose element at face f
//!    is `payload[f][0]`; the row pointer then comes out of the iterator's own
//!    offset arithmetic.  See `snap::x1_nu_iterator` in x1_coeffs.hpp.

namespace at::native {

using recon_poly_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim);

using recon_weno_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim, bool scale);

DECLARE_DISPATCH(recon_poly_nu_fn, call_poly5_nu);
DECLARE_DISPATCH(recon_weno_nu_fn, call_weno5_nu);

}  // namespace at::native
