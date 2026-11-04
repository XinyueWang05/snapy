//! \file test_x1_dim_guard.cpp
//! #280 review item 1: route A must refuse a dimension that is not x1.
//!
//! Once a face vector is installed, `Weno5InterpImpl::left/right` (weno5.cpp)
//! and `Center5InterpImpl::left/right` (cp5.cpp) take the per-face path for
//! ANY `dim` they are handed.  The only guard was `out.size(dim) == rows`, a
//! SIZE test: a block whose x2 extent happens to equal its x1 extent passes it
//! and the x1 coefficients are then applied along x2, silently and with no
//! error -- the rows of a stretched x1 have nothing to do with the x2 spacing.
//!
//! nc2 == nc1 is not exotic: a square block is the default of several example
//! decks.  These tests pin the refusal down on exactly that shape, for both
//! interpolators and through both entry points (`left`/`right` and the
//! `forward` that allocates its own outputs).

// C/C++
#include <cmath>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/recon/interpolation.hpp>
#include <snap/recon/x1_coeffs.hpp>

using namespace snap;

namespace {

// (nvar, nc3, nc2, nc1), as everywhere in snap; x1 is the LAST dimension
enum { DIM3 = 1, DIM2 = 2, DIM1 = 3 };

//! a cubic block: nc3 == nc2 == nc1, the shape that defeats the size-only
//! guard on both of the dimensions route A does not cover
constexpr int kNvar = 2, kN = 14;

//! kN + 1 geometrically stretched faces, so the rows are genuinely per-face
torch::Tensor faces(double q) {
  auto f = torch::empty({kN + 1}, torch::kFloat64);
  auto a = f.accessor<double, 1>();
  double x = 0.3, h = 0.11;
  for (int k = 0; k <= kN; ++k) {
    a[k] = x;
    x += h;
    h *= q;
  }
  return f;
}

//! a positive, smooth state; the values do not matter, the refusal does
torch::Tensor state() {
  auto w = torch::empty({kNvar, kN, kN, kN}, torch::kFloat64);
  auto flat = w.view({-1});
  for (int64_t i = 0; i < flat.numel(); ++i) {
    flat[i] = 1.5 + 0.1 * std::sin(0.37 * static_cast<double>(i));
  }
  return w;
}

//! the output `InterpImpl::forward` would allocate for this `dim`
torch::Tensor out_for(torch::Tensor const& w, int dim, int stencils) {
  auto vec = w.sizes().vec();
  vec[dim] -= stencils - 1;
  return torch::empty(vec, w.options());
}

//! every interpolator route A covers, already holding this block's rows
std::vector<Interp> installed(torch::Tensor const& x1f) {
  Weno5Interp weno;
  Center5Interp cp5;
  weno->set_x1_coeffs(x1f);
  cp5->set_x1_coeffs(x1f);
  EXPECT_TRUE(weno->has_x1_coeffs());
  EXPECT_TRUE(cp5->has_x1_coeffs());
  return {weno.ptr(), cp5.ptr()};
}

}  // namespace

//! The size-only guard cannot tell x2 from x1 on a square block: the rows have
//! nc1 - 4 entries and the x2 output has nc2 - 4 == nc1 - 4 of them.
TEST(X1DimGuard, a_square_block_defeats_the_row_count_guard) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_EQ(out.size(DIM2), x1f.numel() - 5)
        << p->options->type()
        << ": this test is only meaningful while the x2 output and the table "
           "have the same length";
  }
}

TEST(X1DimGuard, left_refuses_x2_on_a_square_block) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_THROW(p->left(w, DIM2, out), c10::Error)
        << p->options->type()
        << ": x1 coefficients were applied along x2 without complaint";
  }
}

TEST(X1DimGuard, right_refuses_x2_on_a_square_block) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_THROW(p->right(w, DIM2, out), c10::Error)
        << p->options->type()
        << ": x1 coefficients were applied along x2 without complaint";
  }
}

//! `forward` allocates its own outputs, so it reaches the same guard by the
//! path `Reconstruct::_apply_inplace` actually uses.
TEST(X1DimGuard, forward_refuses_x2_and_x3) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    EXPECT_THROW(p->forward(w, DIM2), c10::Error) << p->options->type();
    EXPECT_THROW(p->forward(w, DIM3), c10::Error) << p->options->type();
  }
}

//! and x1 itself still runs: the guard refuses a dimension, not the path
TEST(X1DimGuard, x1_still_runs) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto lr = p->forward(w, DIM1);
    EXPECT_EQ(lr.first.size(DIM1), kN - 4) << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.first).all().item<bool>())
        << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.second).all().item<bool>())
        << p->options->type();
  }
}

//! with no table installed the uniform kernels serve every dimension, exactly
//! as they ship; the guard must not touch that path
TEST(X1DimGuard, an_uninstalled_interpolator_still_serves_x2) {
  auto w = state();
  Weno5Interp weno;
  Center5Interp cp5;
  std::vector<Interp> bare = {weno.ptr(), cp5.ptr()};
  for (auto const& p : bare) {
    ASSERT_FALSE(p->has_x1_coeffs());
    auto lr = p->forward(w, DIM2);
    EXPECT_EQ(lr.first.size(DIM2), kN - 4) << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.first).all().item<bool>())
        << p->options->type();
  }
}
