#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

//! \file recon_dispatch_nu.hpp
//! Route A dispatch stubs (snapy#280 stage 1), deliberately separate from
//! `recon_dispatch.hpp` so the shipped stubs stay byte identical (the
//! structural half of the G3 argument).
//!
//! `payload` is the per-face coefficient table, (nout, 48) for WENO5 and
//! (nout, 5) for cp5, in the state's dtype and device, row m for output m.
//!
//! How each backend recovers the face index, and why both are exact (CUDA
//! `threadIdx.x`; CPU the zero-stride third operand, since the `for_each`
//! loop counter is not the face index): design note Sec. 4.3 and
//! docs/derivations/280-route-a-layout.md Sec. 8.1-8.2.

namespace at::native {

using recon_poly_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim);

using recon_weno_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim, bool scale);

DECLARE_DISPATCH(recon_poly_nu_fn, call_poly5_nu);
DECLARE_DISPATCH(recon_weno_nu_fn, call_weno5_nu);

}  // namespace at::native
