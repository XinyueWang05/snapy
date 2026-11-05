// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../src/hydro/x1_mass_covariance.hpp"

using namespace snap;

// #280 item (5): the x1 derivatives of SNAP_X1_MASS_COVARIANCE on a keyed
// stretched x1 (cells/x1rat 1.058, 300 km, nx1 32 and 64). Exact for a
// quadratic, and the wall rho_z of a dry adiabat (g 22, Ts 600, gamma 1.4,
// R 3940) within the bounds of the non-uniform three-point weights in the
// items (3)-(5) note: inner wall 2.30e-5 / 4.54e-7, outer wall 1.13e-2 /
// 8.69e-3 relative (the shipped equal-spacing weights miss by -2.8e-2 and
// +3.6e-2).

namespace {

std::string card(int nx1, bool stretched) {
  return std::string(
             "reference-state: {Tref: 300., Pref: 1.e5}\n"
             "species:\n"
             "  - name: dry\n"
             "    composition: {H: 1.5, He: 0.15}\n"
             "    cv_R: 2.5\n"
             "geometry:\n"
             "  type: cartesian\n"
             "  bounds: {x1min: 0., x1max: 3.e5, x2min: 0., x2max: 1., "
             "x3min: 0., x3max: 1.}\n"
             "  cells: {nx1: ") +
         std::to_string(nx1) + ", nx2: 1, nx3: 1, nghost: 3" +
         (stretched ? ", x1rat: 1.058" : "") +
         "}\n"
         "dynamics:\n"
         "  equation-of-state: {type: ideal-gas}\n"
         "  reconstruct:\n"
         "    vertical: {type: weno5, scale: false, shock: false}\n"
         "    horizontal: {type: weno5, scale: false, shock: false}\n"
         "  riemann-solver: {type: lmars}\n"
         "forcing:\n"
         "  const-gravity: {grav1: -22.}\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: reflecting\n"
         "    x1-outer: reflecting\n";
}

std::shared_ptr<MeshBlockImpl> make_block(int nx1, bool stretched) {
  char fname[] = "/tmp/test-x1-mass-covariance-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << card(nx1, stretched);
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(torch::kCPU, torch::kFloat64);
  return block;
}

// the dry adiabat of the note: rho ~ T^(cv/R), T = Ts - g z / cp
constexpr double kG = 22., kTs = 600., kR = 8.314462618 / 2.112e-3;
constexpr double kCp = 3.5 * kR;
torch::Tensor adiabat(torch::Tensor const& z) {
  return torch::pow((kTs - kG * z / kCp) / kTs, 2.5);
}
torch::Tensor adiabat_slope(torch::Tensor const& z) {
  return 2.5 * adiabat(z) / (kTs - kG * z / kCp) * (-kG / kCp);
}

struct WallErrors {
  double inner, outer;
};

//! relative error of the wall rho_z of the adiabat at the first and last
//! owned cell
WallErrors wall_errors(int nx1) {
  auto block = make_block(nx1, true);
  auto c = block->pcoord;
  auto x1v = c->x1v;
  int is = c->il(), iu = c->iu();
  auto rho = adiabat(x1v).view({1, 1, -1});
  auto exact = adiabat_slope(x1v);
  double in = (x1_cov_one_sided(rho, x1v, is, 1, true).item<double>() /
                   exact[is].item<double>() -
               1.);
  double out = (x1_cov_one_sided(rho, x1v, iu, -1, true).item<double>() /
                    exact[iu].item<double>() -
                1.);
  return {in, out};
}

}  // namespace

// The non-uniform three-point weights are exact for a quadratic: at both
// walls and at every interior cell.
TEST(X1MassCovariance, stretched_slopes_of_a_quadratic_are_exact) {
  for (int nx1 : {32, 64}) {
    auto block = make_block(nx1, true);
    auto c = block->pcoord;
    ASSERT_TRUE(c->options->stretched_x1());
    auto x1v = c->x1v;
    int is = c->il(), iu = c->iu();
    auto z = x1v / 1.e5;  // order one, so the tolerance is relative
    auto rho = (2. - 0.7 * z + 0.3 * z * z).view({1, 1, -1});
    auto exact = (-0.7 + 0.6 * z) / 1.e5;
    auto scale = exact.abs().max().item<double>();

    double in = (x1_cov_one_sided(rho, x1v, is, 1, true) - exact[is])
                    .abs()
                    .item<double>();
    double out = (x1_cov_one_sided(rho, x1v, iu, -1, true) - exact[iu])
                     .abs()
                     .item<double>();
    auto d = x1_cov_d1_centred(rho, x1v, true).view({-1});
    int n1 = x1v.size(0);
    double mid = (d.slice(0, 1, n1 - 1) - exact.slice(0, 1, n1 - 1))
                     .abs()
                     .max()
                     .item<double>();
    std::printf(
        "nx1 %d quadratic: |error| / max|rho_z| inner %.2e outer %.2e "
        "interior %.2e\n",
        nx1, in / scale, out / scale, mid / scale);
    EXPECT_LT(in / scale, 1.e-12) << "nx1 " << nx1;
    EXPECT_LT(out / scale, 1.e-12) << "nx1 " << nx1;
    EXPECT_LT(mid / scale, 1.e-12) << "nx1 " << nx1;
  }
}

// The wall rho_z of the adiabat, against the note's non-uniform numbers.
TEST(X1MassCovariance, stretched_wall_slope_of_the_adiabat) {
  auto e32 = wall_errors(32), e64 = wall_errors(64);
  std::printf(
      "adiabat wall rho_z relative error: inner %+.3e / %+.3e, outer %+.3e "
      "/ %+.3e (nx1 32 / 64); order inner %.2f outer %.2f\n",
      e32.inner, e64.inner, e32.outer, e64.outer,
      std::log2(std::abs(e32.inner / e64.inner)),
      std::log2(std::abs(e32.outer / e64.outer)));
  EXPECT_LT(std::abs(e32.inner), 1.e-4);
  EXPECT_LT(std::abs(e64.inner), 1.e-5);
  EXPECT_LT(std::abs(e32.outer), 1.2e-2);
  EXPECT_LT(std::abs(e64.outer), 9.e-3);
}

// ctest test_x1_mass_covariance_switch runs this with
// SNAP_X1_MASS_COVARIANCE=1: a stretched deck builds and steps a resting
// column with the switch on, and so does its uniform twin.
TEST(X1MassCovariance, stretched_deck_runs_with_the_switch) {
  if (!HydroImpl::x1_mass_covariance())
    GTEST_SKIP() << "SNAP_X1_MASS_COVARIANCE is not set";
  for (bool stretched : {true, false}) {
    auto block = make_block(32, stretched);
    auto c = block->pcoord;
    auto w = torch::zeros({block->phydro->peos->nvar(), c->options->nc3(),
                           c->options->nc2(), c->options->nc1()},
                          torch::kFloat64);
    w[IDN].copy_(torch::exp(-c->x1v / 1.07e5));
    w[IPR].copy_(22. * 1.07e5 * torch::exp(-c->x1v / 1.07e5));
    auto u = block->phydro->peos->compute("W->U", {w});
    Variables vars;
    vars["hydro_w"] = torch::empty_like(w);
    EXPECT_NO_THROW(block->phydro->forward(1.e-3, u, vars))
        << "stretched " << stretched;
  }
}
