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

#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../src/hydro/hydro_dispatch.hpp"

using namespace snap;

// #280 stage 1, the reference on a keyed stretched x1 and what it refuses.
// Every refusal is shown twice: the stretched card throws with its message,
// and the same card on a uniform grid does not.

namespace {

// a 24-cell column, 2 km tall, cells growing upward by 1.058 (or uniform)
std::string card(bool stretched, std::string const& forcing = "}",
                 std::string const& inner = "reflecting",
                 std::string const& outer = "reflecting",
                 std::string const& geometry = "cartesian",
                 std::string const& recon = "weno5") {
  return std::string(
             "reference-state: {Tref: 300., Pref: 1.e5}\n"
             "species:\n"
             "  - name: dry\n"
             "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
             "    cv_R: 2.5\n"
             "geometry:\n"
             "  type: ") +
         geometry +
         "\n"
         "  bounds: {x1min: 1000., x1max: 3000., x2min: 0., x2max: 1., "
         "x3min: 0., x3max: 1.}\n"
         "  cells: {nx1: 24, nx2: 1, nx3: 1, nghost: 3" +
         (stretched ? ", x1rat: 1.058" : "") +
         "}\n"
         "dynamics:\n"
         "  equation-of-state: {type: ideal-gas}\n"
         "  reconstruct:\n"
         "    vertical: {type: " +
         recon +
         ", scale: false, shock: false}\n"
         "    horizontal: {type: weno5, scale: false, shock: false}\n"
         "  riemann-solver: {type: lmars}\n"
         "forcing:\n"
         "  const-gravity: {grav1: -9.8" +
         forcing +
         "\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: " +
         inner +
         "\n"
         "    x1-outer: " +
         outer + "\n";
}

std::string write_temp(std::string const& text) {
  char fname[] = "/tmp/test-stretched-x1-ref-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << text;
  return fname;
}

//! build the block from the card and take one forward step of a resting
//! isothermal column
void build_and_step(std::string const& text) {
  auto fname = write_temp(text);
  std::shared_ptr<MeshBlockImpl> block;
  try {
    auto options = MeshBlockOptionsImpl::from_yaml(fname);
    std::remove(fname.c_str());
    block = std::make_shared<MeshBlockImpl>(options);
  } catch (...) {
    std::remove(fname.c_str());
    throw;
  }
  block->to(torch::kCPU, torch::kFloat64);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  auto z = coord->x1v - 1000.;
  w[IDN].copy_(torch::exp(-z / 8.e3));
  w[IPR].copy_(8.e3 * 9.8 * torch::exp(-z / 8.e3));
  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  block->phydro->forward(1.e-3, u, vars);
}

//! the card throws with `message` when stretched and not when uniform
void refuses(std::string const& message, std::string const& forcing = "}",
             std::string const& inner = "reflecting",
             std::string const& outer = "reflecting",
             std::string const& geometry = "cartesian") {
  bool threw = false;
  try {
    build_and_step(card(true, forcing, inner, outer, geometry));
  } catch (c10::Error const& e) {
    threw = true;
    EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
        << "expected \"" << message << "\", got:\n"
        << e.what();
  }
  EXPECT_TRUE(threw) << "a stretched x1 did not refuse: " << message;
  if (geometry == "cartesian") {
    EXPECT_NO_THROW(
        build_and_step(card(false, forcing, inner, outer, geometry)));
  } else {  // the uniform twin need not be a runnable deck; it must not refuse
    try {
      build_and_step(card(false, forcing, inner, outer, geometry));
    } catch (c10::Error const& e) {
      EXPECT_EQ(std::string(e.what()).find(message), std::string::npos)
          << e.what();
    }
  }
}

}  // namespace

TEST(StretchedX1Ref, uniform_and_stretched_reflecting_decks_run) {
  EXPECT_NO_THROW(build_and_step(card(false)));
  EXPECT_NO_THROW(build_and_step(card(true)));
}

TEST(StretchedX1Ref, refuses_curvature_correction) {
  // gravity-work face-wallc leaves the radial-exact work off, so the dx^2/12
  // curvature correction of weno5 runs
  refuses("the dx^2/12 x1 curvature correction of cp3/cp5/weno5 is not "
          "supported on a stretched x1",
          ", gravity-work: face-wallc}");
}

TEST(StretchedX1Ref, refuses_diffusion) {
  refuses("[Diffusion] not supported on a stretched x1",
          "}\n  diffusion: {nu_iso: 1.e-3}");
}

TEST(StretchedX1Ref, refuses_bot_heat) {
  refuses("[BotHeat] not supported on a stretched x1",
          "}\n  bot-heat: {flux: 1., depth: 1}");
}

TEST(StretchedX1Ref, refuses_top_cool) {
  refuses("[TopCool] not supported on a stretched x1",
          "}\n  top-cool: {flux: -1., depth: 1}");
}

TEST(StretchedX1Ref, refuses_periodic_x1) {
  refuses("needs reflecting x1 boundaries", "}", "periodic", "periodic");
}

TEST(StretchedX1Ref, refuses_outflow_x1_outer) {
  refuses("the x1-outer boundary is not reflecting", "}", "reflecting",
          "outflow");
}

TEST(StretchedX1Ref, refuses_spherical_polar) {
  refuses("is supported only on a cartesian geometry", "}", "reflecting",
          "reflecting", "spherical-polar");
}

TEST(StretchedX1Ref, refuses_gnomonic_equiangle) {
  refuses("is supported only on a cartesian geometry", "}", "reflecting",
          "reflecting", "gnomonic-equiangle");
}

// ctest test_stretched_x1_ref_wb_ref4 runs this with SNAP_WB_REF4=1, and
// test_stretched_x1_ref_x1_centroid with SNAP_X1_CENTROID_EXACT=1
TEST(StretchedX1Ref, refuses_wb_ref4_switch) {
  if (!HydroImpl::wb_ref4()) GTEST_SKIP() << "SNAP_WB_REF4 is not set";
  refuses("is not supported on a stretched x1 (geometry/x1f or cells/x1rat");
}

namespace {

//! an isothermal column of `n` cells over `height`, cells growing upward by
//! `q`, cell-centred values; the grid is what balance_column is handed
std::pair<torch::Tensor, torch::Tensor> column(int n, double q,
                                               double height) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto k = torch::arange(n + 1, opt);
  auto faces = height * torch::expm1(k * std::log(q)) / std::expm1(n * std::log(q));
  auto dx1f = faces.slice(0, 1) - faces.slice(0, 0, n);
  auto zc = 0.5 * (faces.slice(0, 1) + faces.slice(0, 0, n));
  auto w = torch::zeros({IPR + 1, 1, 1, n}, opt);
  double H = 2.e4, g = 22.;
  w[IDN][0][0].copy_(1.e7 / (g * H) * torch::exp(-zc / H));
  w[IPR][0][0].copy_(1.e7 * torch::exp(-zc / H));
  return {w, dx1f};
}

}  // namespace

// RED/GREEN for the log1p log-mean (#280). A fine stretched column (nx1 =
// 128, cells 1.058x apart, 3e5 m over a 2e4 m scale height) is balanced by
// the log-mean dp / ln(lo / hi) only to its round-off floor eps (p / dp)^2,
// about 1e-10 here, so at rtol 1e-13 the projection throws; dp / log1p(dp /
// hi) floors at eps p / dp and converges.
TEST(StretchedX1Ref, keyed_column_balances_to_1e13) {
  auto [w, dx1f] = column(128, 1.058, 3.e5);
  auto [wb, err, sweeps] =
      balance_column(w, dx1f, 22., /*wall_clamp=*/true, /*rtol=*/1.e-13,
                     /*max_iter=*/400, "cartesian", /*stretched=*/true);
  std::printf("keyed nx1 = 128: residual %.3e after %d sweeps\n", err, sweeps);
  EXPECT_LT(err, 1.e-13);
}

// the same column without the key takes the plain log-mean: the two cell
// pressures differ only by the round-off log1p removes
TEST(StretchedX1Ref, keyed_and_unkeyed_log_means_agree_to_round_off) {
  auto [w, dx1f] = column(32, 1.058, 3.e5);
  auto sizes = w.sizes().slice(1).vec();
  auto run = [&](int x1_grid) {
    auto lo = torch::empty(sizes, w.options()), hi = torch::empty_like(lo);
    auto pref = torch::empty_like(lo), dsf = torch::empty_like(lo),
         dref = torch::empty_like(lo);
    at::native::call_hydro_ref_x1(torch::kCPU, w, dx1f, torch::Tensor(), lo,
                                  hi, pref, dsf, dref, 31, 22., x1_grid, true,
                                  true, true);
    return pref;
  };
  auto keyed = run(kX1RefStretched), plain = run(kX1RefNonUniform);
  double rel = ((keyed - plain).abs() / plain).max().item<double>();
  std::printf("keyed vs plain log-mean, max relative difference %.3e\n", rel);
  EXPECT_LT(rel, 1.e-12);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
