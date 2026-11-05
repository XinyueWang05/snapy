// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>

// torch
#include <torch/torch.h>

#include "../src/hydro/gravity_work_radial.hpp"

using namespace snap;

// #280 item (3): a stretched x1 does not use the dx^2/12 curvature flux H
// (refused, hydro_forward.cpp); it uses option F, the corrected-PE work of
// SNAP_GRAVITY_WORK_RADIAL_EXACT (gravity_work_radial.hpp), which is fourth
// order there. This pins that order. Column: a dry adiabat (300 km, g 22,
// Ts 600, gamma 1.4, R 3940) and the mass flux F = rho(z) sin(pi z / L). Per
// cell, the face-form work g (F_- + F_+) / 2 plus option F against the exact
// g <F>, at a fixed h_max / h_min so the order is that of a stretched family.

namespace {

constexpr double kL = 3.e5, kG = 22., kTs = 600.;
constexpr double kR = 8.314462618 / 2.112e-3, kCp = 3.5 * kR;
constexpr double kRatio = 5.742;  // h_max / h_min, x1rat 1.058 at nx1 32

double flux(double z) {
  return std::pow((kTs - kG * z / kCp) / kTs, 2.5) * std::sin(M_PI * z / kL);
}

//! max over the cells 1..n-2 of |W - g <F>| / max |g <F>|
double option_f_error(int n) {
  double q = std::pow(kRatio, 1. / (n - 1));
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto h = torch::pow(torch::full({n}, q, opt), torch::arange(n, opt));
  h *= kL / h.sum();
  auto x1f = torch::cat({torch::zeros({1}, opt), h.cumsum(0)});
  auto x1v = 0.5 * (x1f.slice(0, 1) + x1f.slice(0, 0, n));

  // eight-point Gauss-Legendre cell averages
  constexpr double gx[8] = {-0.9602898564975363, -0.7966664774136267,
                            -0.5255324099163290, -0.1834346424956498,
                            0.1834346424956498,  0.5255324099163290,
                            0.7966664774136267,  0.9602898564975363};
  constexpr double gw[8] = {0.1012285362903763, 0.2223810344533745,
                            0.3137066458778873, 0.3626837833783620,
                            0.3626837833783620, 0.3137066458778873,
                            0.2223810344533745, 0.1012285362903763};
  auto xf = x1f.accessor<double, 1>();
  auto face = torch::empty({n + 1}, opt), avg = torch::empty({n}, opt);
  for (int i = 0; i <= n; ++i) face[i] = flux(xf[i]);
  for (int i = 0; i < n; ++i) {
    double c = 0.5 * (xf[i] + xf[i + 1]), r = 0.5 * (xf[i + 1] - xf[i]), s = 0.;
    for (int k = 0; k < 8; ++k) s += 0.5 * gw[k] * flux(c + r * gx[k]);
    avg[i] = s;
  }

  // dt = 1: the face form and option F as hydro_forward.cpp books them
  double grav1 = -kG;
  auto div = (face.slice(0, 1) - face.slice(0, 0, n)) / h;
  auto work = grav1 * 0.5 * (face.slice(0, 1) + face.slice(0, 0, n)) +
              corrected_pe_work(-div, x1f, x1v, 0, n, grav1, X1Measure::plain);
  auto exact = grav1 * avg;
  auto err = (work - exact).slice(0, 1, n - 1).abs().max().item<double>();
  return err / exact.abs().max().item<double>();
}

}  // namespace

TEST(StretchedX1OptionF, fourth_order_on_a_stretched_family) {
  double e64 = option_f_error(64), e128 = option_f_error(128);
  std::printf(
      "option F on h_max/h_min = %.3f: nx1 64 %.4e, nx1 128 %.4e, "
      "order %.3f\n",
      kRatio, e64, e128, std::log2(e64 / e128));
  EXPECT_LE(e64, 2.e-6);
  EXPECT_GE(e64 / e128, std::pow(2., 3.8));
}
