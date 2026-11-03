//! \file test_x1_coeffs.cpp
//! Author unit checks for route A per-face WENO5/cp5 coefficients (#280
//! stage 1, piece 2).  These are the implementer's own checks, NOT the G1
//! gate: G1 lives in Xinyue's harness on the grids of note Sec. 8.2.

// C/C++
#include <algorithm>
#include <cmath>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>

// tests

using namespace snap;

namespace {

constexpr double kEps = 2.220446049250313e-16;

//! Xi's G1 form: |sum_k c_k - 1| <= 8 * eps * sum_k |c_k|.
void expect_row_sum(torch::Tensor const& row, char const* what) {
  double s = row.sum().item<double>();
  double scale = row.abs().sum().item<double>();
  EXPECT_LE(std::abs(s - 1.), 8. * kEps * scale)
      << what << ": sum = " << s << ", 8*eps*sum|c| = " << 8. * kEps * scale;
}

torch::Tensor faces_uniform(int n, double h = 1.) {
  return torch::arange(n, torch::kFloat64) * h;
}

torch::Tensor faces_geometric(int n, double q, double h0 = 1.) {
  std::vector<double> y(n);
  y[0] = 0.;
  double h = h0;
  for (int i = 1; i < n; ++i) {
    y[i] = y[i - 1] + h;
    h *= q;
  }
  return torch::tensor(y, torch::kFloat64);
}

//! Exact cell averages of a polynomial, by 3-point Gauss-Legendre.
torch::Tensor cell_avg_poly(torch::Tensor const& x1f,
                            std::vector<double> const& c) {
  auto f = x1f.accessor<double, 1>();
  int64_t n = x1f.numel() - 1;
  std::vector<double> u(n);
  const double g[3] = {-std::sqrt(3. / 5.), 0., std::sqrt(3. / 5.)};
  const double wg[3] = {5. / 18., 8. / 18., 5. / 18.};
  for (int64_t k = 0; k < n; ++k) {
    double a = f[k], b = f[k + 1], acc = 0.;
    for (int i = 0; i < 3; ++i) {
      double x = .5 * (a + b) + .5 * (b - a) * g[i];
      double p = 0.;
      for (int d = static_cast<int>(c.size()) - 1; d >= 0; --d) p = p * x + c[d];
      acc += wg[i] * p;
    }
    u[k] = acc;
  }
  return torch::tensor(u, torch::kFloat64);
}

double poly_at(std::vector<double> const& c, double x) {
  double p = 0.;
  for (int d = static_cast<int>(c.size()) - 1; d >= 0; --d) p = p * x + c[d];
  return p;
}

//! Independent re-derivation of the 5-cell row, straight from Shu (1998)
//! eq. 2.20 on the six nodes.  The library builds it as sum_r d_r p_r, so this
//! is a genuine second route, not a copy.
std::vector<double> primitive_row5(std::vector<double> const& z, double xf) {
  const int n = 5;
  std::vector<double> lp(n + 1, 0.);
  for (int j = 0; j <= n; ++j) {
    double s = 0.;
    for (int i = 0; i <= n; ++i) {
      if (i == j) continue;
      double term = 1. / (z[j] - z[i]);
      for (int l = 0; l <= n; ++l) {
        if (l == j || l == i) continue;
        term *= (xf - z[l]) / (z[j] - z[l]);
      }
      s += term;
    }
    lp[j] = s;
  }
  std::vector<double> w(n, 0.);
  for (int k = 0; k < n; ++k) {
    double acc = 0.;
    for (int j = k + 1; j <= n; ++j) acc += lp[j];
    w[k] = (z[k + 1] - z[k]) * acc;
  }
  return w;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. A uniform face vector reproduces the shipped literals (note Sec. 4.5).
// ---------------------------------------------------------------------------
TEST(X1Coeffs, uniform_matches_shipped_rows) {
  Weno5Interp weno;
  Center5Interp cp5;

  for (double h : {1., 1.e-3, 7.5e3}) {
    auto x1f = faces_uniform(12, h);
    auto w = weno5_x1_coeffs(x1f);
    auto p = cp5_x1_coeffs(x1f);

    auto shipped_m = weno->cm.reshape({45});
    auto shipped_p = weno->cp.reshape({45});

    for (int64_t m = 0; m < w.first.size(0); ++m) {
      auto cm45 = w.first[m].slice(0, 0, 45);
      auto cp45 = w.second[m].slice(0, 0, 45);
      EXPECT_LT((cm45 - shipped_m).abs().max().item<double>(), 1.e-14)
          << "h = " << h << ", row " << m;
      EXPECT_LT((cp45 - shipped_p).abs().max().item<double>(), 1.e-14)
          << "h = " << h << ", row " << m;

      // the three linear weights are the literals .3/.6/.1
      auto d = w.first[m].slice(0, 45, 48);
      EXPECT_NEAR(d[0].item<double>(), .3, 1.e-15);
      EXPECT_NEAR(d[1].item<double>(), .6, 1.e-15);
      EXPECT_NEAR(d[2].item<double>(), .1, 1.e-15);

      EXPECT_LT((p.first[m] - cp5->cm).abs().max().item<double>(), 1.e-15);
      EXPECT_LT((p.second[m] - cp5->cp).abs().max().item<double>(), 1.e-15);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Row sums, in Xi's G1 form, on uniform / geometric / ratio-jump windows.
// ---------------------------------------------------------------------------
TEST(X1Coeffs, row_sums) {
  std::vector<std::pair<char const*, torch::Tensor>> grids = {
      {"uniform", faces_uniform(16)},
      {"geometric q=1.05", faces_geometric(16, 1.05)},
      {"geometric q=0.94080", faces_geometric(16, 0.94080)},
      {"geometric q=1.2", faces_geometric(16, 1.2)},
  };

  // a ratio jump 1:4 in the middle
  {
    std::vector<double> y{0.};
    for (int i = 0; i < 8; ++i) y.push_back(y.back() + 1.);
    for (int i = 0; i < 7; ++i) y.push_back(y.back() + 4.);
    grids.emplace_back("jump 1:4", torch::tensor(y, torch::kFloat64));
  }

  for (auto const& [name, x1f] : grids) {
    auto w = weno5_x1_coeffs(x1f);
    auto p = cp5_x1_coeffs(x1f);
    for (int64_t m = 0; m < w.first.size(0); ++m) {
      for (auto const& t : {w.first, w.second}) {
        for (int r = 0; r < 3; ++r) {
          expect_row_sum(t[m].slice(0, 5 * r, 5 * r + 5), name);
        }
        // the three linear weights are positive and sum to one
        auto d = t[m].slice(0, 45, 48);
        EXPECT_GT(d.min().item<double>(), 0.) << name << " row " << m;
        expect_row_sum(d, name);
      }
      expect_row_sum(p.first[m], name);
      expect_row_sum(p.second[m], name);
    }
  }
}

// ---------------------------------------------------------------------------
// 3. The cp5 row, which the library builds as sum_r d_r p_r (note Sec. 4.2),
//    equals the 6-node primitive-function row derived independently here.
//    That is simultaneously a check of the note's closed form for d_r and of
//    the primitive-function candidates: nothing else makes the two agree.
// ---------------------------------------------------------------------------
TEST(X1Coeffs, cp5_row_matches_an_independent_derivation) {
  double worst = 0.;
  for (double q : {1.0, 1.05, 0.94080, 1.2, 0.8, 1.6}) {
    auto x1f = faces_geometric(16, q);
    auto p = cp5_x1_coeffs(x1f);
    auto f = x1f.accessor<double, 1>();

    for (int64_t m = 0; m < p.first.size(0); ++m) {
      std::vector<double> y(6);
      for (int k = 0; k < 6; ++k) y[k] = f[m + k];
      double h2 = y[3] - y[2];

      // cp: target y3, in the normalised coordinate
      std::vector<double> z(6);
      for (int k = 0; k < 6; ++k) z[k] = (y[k] - y[3]) / h2;
      auto ref_p = primitive_row5(z, 0.);

      // cm: target y2, same construction on the mirrored window
      std::vector<double> ym(6), zm(6);
      for (int k = 0; k < 6; ++k) ym[k] = -y[5 - k];
      double h2m = ym[3] - ym[2];
      for (int k = 0; k < 6; ++k) zm[k] = (ym[k] - ym[3]) / h2m;
      auto t = primitive_row5(zm, 0.);
      std::vector<double> ref_m(5);
      for (int n = 0; n < 5; ++n) ref_m[n] = t[4 - n];

      for (int n = 0; n < 5; ++n) {
        worst = std::max(worst,
                         std::abs(p.first[m][n].item<double>() - ref_m[n]));
        worst = std::max(worst,
                         std::abs(p.second[m][n].item<double>() - ref_p[n]));
        EXPECT_NEAR(p.first[m][n].item<double>(), ref_m[n], 1.e-13)
            << "q = " << q << ", row " << m << ", cm col " << n;
        EXPECT_NEAR(p.second[m][n].item<double>(), ref_p[n], 1.e-13)
            << "q = " << q << ", row " << m << ", cp col " << n;
      }
    }
  }
  std::cout << "  max|sum_r d_r p_r - 6-node primitive row| = " << worst
            << std::endl;
}

// ---------------------------------------------------------------------------
// 4. Exactness on polynomial data, with max|u| scaling (the G1 form again).
//    Each candidate is exact to degree 2, the cp5 / optimal row to degree 4.
//    THIS is the check that pins the face index: the target point is read from
//    the face vector, so a row built for the wrong face fails here.
// ---------------------------------------------------------------------------
TEST(X1Coeffs, polynomial_exactness_and_face_index) {
  auto x1f = faces_geometric(18, 1.1, 0.37);
  auto w = weno5_x1_coeffs(x1f);
  auto p = cp5_x1_coeffs(x1f);
  auto f = x1f.accessor<double, 1>();

  for (int deg = 0; deg <= 4; ++deg) {
    std::vector<double> c(deg + 1, 0.);
    c[deg] = 1.;
    if (deg > 0) c[0] = -0.3;  // a non-trivial constant part

    auto u = cell_avg_poly(x1f, c);
    double umax = u.abs().max().item<double>();

    for (int64_t m = 0; m < w.first.size(0); ++m) {
      auto win = u.slice(0, m, m + 5);

      // cm targets face m+2, cp targets face m+3
      const double tgt[2] = {poly_at(c, f[m + 2]), poly_at(c, f[m + 3])};

      for (int side = 0; side < 2; ++side) {
        auto t = side == 0 ? w.first[m] : w.second[m];
        auto opt = side == 0 ? p.first[m] : p.second[m];

        if (deg <= 2) {
          for (int r = 0; r < 3; ++r) {
            double v = (t.slice(0, 5 * r, 5 * r + 5) * win).sum().item<double>();
            double tol = 64. * kEps * umax *
                         t.slice(0, 5 * r, 5 * r + 5).abs().sum().item<double>();
            EXPECT_LE(std::abs(v - tgt[side]), std::max(tol, 1.e-13 * umax))
                << "deg " << deg << " row " << m << " side " << side << " r "
                << r;
          }
        }
        double vo = (opt * win).sum().item<double>();
        EXPECT_LE(std::abs(vo - tgt[side]), 1.e-12 * std::max(umax, 1.))
            << "cp5 deg " << deg << " row " << m << " side " << side;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 5. beta is dimensionless and stencil independent on linear data.
//    On exactly linear data all three beta_r equal (slope * h_i)^2, which is
//    what makes the shipped absolute eps = 1e-6 keep its meaning (Sec. 4.2).
// ---------------------------------------------------------------------------
TEST(X1Coeffs, beta_is_scale_free) {
  auto x1f = faces_geometric(16, 1.15, 2.5);
  auto w = weno5_x1_coeffs(x1f);
  auto f = x1f.accessor<double, 1>();

  const double slope = 3.25;
  auto u = cell_avg_poly(x1f, {-1.5, slope});

  for (int64_t m = 0; m < w.first.size(0); ++m) {
    auto win = u.slice(0, m, m + 5);
    double h2 = f[m + 3] - f[m + 2];
    double expect = slope * h2 * (slope * h2);

    for (int side = 0; side < 2; ++side) {
      auto t = side == 0 ? w.first[m] : w.second[m];
      for (int r = 0; r < 3; ++r) {
        double a = (t.slice(0, 15 + 10 * r, 20 + 10 * r) * win).sum().item<double>();
        double b = (t.slice(0, 20 + 10 * r, 25 + 10 * r) * win).sum().item<double>();
        double beta = 13. / 12. * a * a + .25 * b * b;
        EXPECT_NEAR(beta, expect, 1.e-10 * expect)
            << "row " << m << " side " << side << " r " << r;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 6. Input refusals.
// ---------------------------------------------------------------------------
TEST(X1Coeffs, refusals) {
  EXPECT_THROW(weno5_x1_coeffs(torch::zeros({5}, torch::kFloat64)),
               c10::Error);  // too few faces
  EXPECT_THROW(weno5_x1_coeffs(torch::zeros({3, 7}, torch::kFloat64)),
               c10::Error);  // not 1-D
  auto bad = faces_uniform(10).clone();
  bad[4] = bad[5];
  EXPECT_THROW(weno5_x1_coeffs(bad), c10::Error);  // not strictly increasing
}

// ---------------------------------------------------------------------------
// 8. Reconstruct::set_x1_faces reaches both interpolators, and the buffers are
//    reachable by their dotted names (what Xinyue's harness uses).
// ---------------------------------------------------------------------------
TEST(X1Coeffs, reconstruct_seam_and_buffer_names) {
  auto opts = ReconstructOptionsImpl::create();
  opts->interp()->type("weno5");
  Reconstruct recon(opts);

  auto x1f = faces_geometric(21, 1.03);
  recon->set_x1_faces(x1f);

  auto named = recon->named_buffers();
  ASSERT_TRUE(named.contains("interp1.cm_x1"));
  ASSERT_TRUE(named.contains("interp1.cp_x1"));
  ASSERT_TRUE(named.contains("interp2.cm_x1"));
  ASSERT_TRUE(named.contains("interp2.cp_x1"));

  EXPECT_EQ(named["interp1.cm_x1"].size(0), x1f.numel() - 5);
  EXPECT_EQ(named["interp1.cm_x1"].size(1), 48);
  EXPECT_EQ(named["interp2.cm_x1"].size(1), 5);

  // the named buffer is the live tensor: a harness can overwrite rows in place
  auto live = named["interp1.cm_x1"];
  live[0][45] = 0.5;
  EXPECT_EQ(
      std::dynamic_pointer_cast<Weno5InterpImpl>(recon->pinterp1)->cm_x1[0][45]
          .item<double>(),
      0.5);
}

// ---------------------------------------------------------------------------
// 9. A type route A does not cover ignores the call (note Sec. 4.7).
// ---------------------------------------------------------------------------
TEST(X1Coeffs, uncovered_types_ignore_faces) {
  for (char const* type : {"dc", "plm", "ppm", "cp3", "weno3"}) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);
    recon->set_x1_faces(faces_geometric(21, 1.03));
    EXPECT_FALSE(recon->pinterp1->has_x1_coeffs()) << type;
  }
}
// ---------------------------------------------------------------------------
// 12. The ghost width the stencils need from global_x1f(extra).
//     Owed to Jiheng (piece 1).  Derived here from the real indices rather
//     than asserted: the test reproduces reconstruct.cpp's il/iu, finds which
//     output element targets the first and the last INTERIOR face, and reports
//     the lowest and highest face each of their windows touches.
// ---------------------------------------------------------------------------
TEST(X1Coeffs, ghost_faces_needed_by_the_stencils) {
  for (int nghost : {3, 4}) {
    for (int nx1 : {16, 64}) {
      const int nc1 = nx1 + 2 * nghost;

      // reconstruct.cpp:86-88 with stencils() == 5
      const int rec_ng = 5 / 2 + 1;  // == 3, independent of the deck's nghost
      const int il = rec_ng;
      const int iu = nc1 - rec_ng;
      const int nout = iu - il + 2;

      auto x1f = faces_geometric(nc1 + 1, 1.01);
      auto w = weno5_x1_coeffs(x1f);
      ASSERT_EQ(w.first.size(0), nout);

      // interior cells are nghost .. nc1-nghost-1, so the interior faces are
      // nghost .. nc1-nghost
      const int f_first = nghost;
      const int f_last = nc1 - nghost;

      // cm (IRT) row m targets face m+2; cp (ILT) row m targets face m+3.
      // Row m's window is cells m..m+4, bounded by faces m..m+5.
      int lo = nc1, hi = 0;
      for (int f : {f_first, f_last}) {
        for (int off : {2, 3}) {
          int m = f - off;
          ASSERT_GE(m, 0) << "face " << f << " off " << off;
          ASSERT_LT(m, nout) << "face " << f << " off " << off;
          lo = std::min(lo, m);
          hi = std::max(hi, m + 5);
        }
      }

      // in global numbering the interior spans faces 0 .. nx1, and block face
      // k is global face k - nghost
      int g_lo = lo - nghost;
      int g_hi = hi - nghost;
      std::cout << "  nghost = " << nghost << " nx1 = " << nx1
                << ": stencils for the first and last interior face span "
                << "global faces " << g_lo << " .. " << g_hi
                << " (interior is 0 .. " << nx1 << "), so they need "
                << -g_lo << " ghost faces below and " << g_hi - nx1
                << " above; extra = max(0, 3 - nghost) = "
                << std::max(0, 3 - nghost) << std::endl;

      // exactly three ghost faces per end, no more, whatever nghost is
      EXPECT_EQ(g_lo, -3);
      EXPECT_EQ(g_hi, nx1 + 3);

      // and today's block array already holds them
      EXPECT_GE(lo, 0);
      EXPECT_LE(hi, x1f.numel() - 1);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
