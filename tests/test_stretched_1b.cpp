// #280 stage 1b on a keyed stretched x1: one exact-answer RED/GREEN per item
#include <gtest/gtest.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../src/hydro/gravity_work_radial.hpp"
#include "../src/hydro/x1_mass_covariance.hpp"

using namespace snap;

namespace {
std::shared_ptr<MeshBlockImpl> block(
    bool stretched) {  // 1000..3000 m, 24 cells
  char f[] = "/tmp/stretched-1b-XXXXXX";
  close(mkstemp(f));
  std::ofstream(f) << "reference-state: {Tref: 300., Pref: 1.e5}\n"
                      "species: [{name: dry, composition: {O: 0.42, N: 1.56, "
                      "Ar: 0.01}, cv_R: 2.5}]\n"
                      "geometry: {type: cartesian, bounds: {x1min: 1000., "
                      "x1max: 3000., x2min: 0., "
                      "x2max: 1., x3min: 0., x3max: 1.}, cells: {nx1: 24, nx2: "
                      "1, nx3: 1, nghost: 3"
                   << (stretched ? ", x1rat: 1.1" : "")
                   << "}}\ndynamics: {equation-of-state: {type: ideal-gas}, "
                      "reconstruct: {vertical: "
                      "{type: weno5}, horizontal: {type: weno5}}, "
                      "riemann-solver: {type: lmars}}\n"
                      "forcing: {const-gravity: {grav1: "
                      "-9.8}}\nboundary-condition: {external: "
                      "{x1-inner: reflecting, x1-outer: reflecting}}\n";
  auto b = std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(f));
  std::remove(f);
  b->to(torch::kCPU, torch::kFloat64);
  return b;
}
torch::Tensor state(MeshBlockImpl& b) {  // rest, rho 1, p 1e5
  auto w = torch::zeros({b.phydro->peos->nvar(), 1, 1, b.pcoord->x1v.size(0)},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  return w;
}
torch::Tensor inner(MeshBlockImpl& b, torch::Tensor v) {
  return v.narrow(-1, b.pcoord->il(), 24);
}
double dev(torch::Tensor a, double x) {
  return (a - x).abs().max().item<double>();
}
}  // namespace

// (1) vx = a x1 (dynamic nu): du[IVX] = 0, du[IPR] = dt nu (4/3) a^2; linear T
// and rho (kinematic kappa): du[IPR] = dt kappa cv T' rho'. RED: the stretched
// refusal / 2*dx1f spacing.
TEST(Stretched1b, diffusion_exact_on_linear_fields) {
  auto b = block(true);
  auto x = b->pcoord->x1v.view({1, 1, -1});
  auto w = state(*b);
  w[IVX].copy_(1.e-3 * x);
  Diffusion visc(DiffusionOptionsImpl::from_yaml(
                     YAML::Load("diffusion: {nu_iso: 2., dynamic: true}")),
                 b->phydro.get());
  auto du = visc->forward(torch::zeros_like(w), w,
                          b->phydro->peos->compute("W->T", {w}), 0.5);
  EXPECT_LT(dev(inner(*b, du[IVX]), 0.), 1.e-12);
  EXPECT_LT(dev(inner(*b, du[IPR]), 0.5 * 2. * 4. / 3. * 1.e-6), 1.e-16);
  w = state(*b);
  w[IDN].copy_(1. - 1.e-4 * (x - 1000.));
  auto temp = 300. - 2.e-3 * (x - 1000.);
  auto cv =
      b->phydro->peos->specific_heat_cv(w, temp).flatten()[0].item<double>();
  Diffusion cond(
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {kappa_iso: 3.}")),
      b->phydro.get());
  du = cond->forward(torch::zeros_like(w), w, temp, 0.5);
  EXPECT_LT(dev(inner(*b, du[IPR]), 0.5 * 3. * cv * 2.e-7) / (3. * cv * 1.e-7),
            1.e-8);
}

// (2) a layer of thickness D deposits flux*dt in the column on any grid; an
// integer depth is refused on a stretched x1. RED: thickness is an unknown key.
TEST(Stretched1b, heat_layer_deposits_flux_dt) {
  for (bool stretched : {true, false}) {
    auto b = block(stretched);
    auto w = state(*b);
    auto t = b->phydro->peos->compute("W->T", {w});
    auto h = inner(*b, b->pcoord->dx1f);
    BotHeat bot(BotHeatOptionsImpl::from_yaml(
                    YAML::Load("bot-heat: {flux: 100., thickness: 230.}")),
                b->phydro.get());
    TopCool top(TopCoolOptionsImpl::from_yaml(
                    YAML::Load("top-cool: {flux: -40., thickness: 7.}")),
                b->phydro.get());
    EXPECT_NEAR(
        (inner(*b, bot->forward(torch::zeros_like(w), w, t, 0.25)[IPR][0][0]) *
         h)
            .sum()
            .item<double>(),
        25., 1.e-12);
    EXPECT_NEAR(
        (inner(*b, top->forward(torch::zeros_like(w), w, t, 0.25)[IPR][0][0]) *
         h)
            .sum()
            .item<double>(),
        -10., 1.e-12);
  }
  EXPECT_THROW(BotHeat(BotHeatOptionsImpl::from_yaml(
                           YAML::Load("bot-heat: {flux: 1., depth: 2}")),
                       block(true)->phydro.get()),
               c10::Error);
}

// (3) a stretched x1 uses option F (H stays refused): fourth order, <= 2e-6 at
// nx1 64
TEST(Stretched1b, option_f_fourth_order) {
  auto err = [](int n) {
    auto h = torch::pow(
        torch::full({n}, std::pow(5.742, 1. / (n - 1)), torch::kFloat64),
        torch::arange(n, torch::kFloat64));
    h *= 3.e5 / h.sum();
    auto xf = torch::cat({torch::zeros({1}, torch::kFloat64), h.cumsum(0)});
    auto F = [](torch::Tensor z) {
      return torch::pow(1. - 22. * z / (3.5 * 3937. * 600.), 2.5) *
             torch::sin(M_PI * z / 3.e5);
    };
    auto g = torch::tensor({-0.861136311594053, -0.339981043584856,
                            0.339981043584856, 0.861136311594053},
                           torch::kFloat64);
    auto gw = torch::tensor({0.347854845137454, 0.652145154862546,
                             0.652145154862546, 0.347854845137454},
                            torch::kFloat64);
    auto c = 0.5 * (xf.slice(0, 1) + xf.slice(0, 0, n)), r = 0.5 * h;
    auto avg = (0.5 * gw * F(c.unsqueeze(1) + r.unsqueeze(1) * g)).sum(1);
    auto ff = F(xf);
    auto work = -22. * 0.5 * (ff.slice(0, 1) + ff.slice(0, 0, n)) +
                corrected_pe_work(-(ff.slice(0, 1) - ff.slice(0, 0, n)) / h, xf,
                                  c, 0, n, -22., X1Measure::plain);
    return ((work + 22. * avg).slice(0, 1, n - 1).abs().max() /
            (22. * avg).abs().max())
        .item<double>();
  };
  EXPECT_LE(err(64), 2.e-6);
  EXPECT_GE(err(64) / err(128), std::pow(2., 3.8));
}

// (5) the mass-covariance slopes are exact for a quadratic on a stretched x1.
// RED: (-3, 4, -1)/2h
TEST(Stretched1b, covariance_slopes_exact_for_quadratic) {
  auto b = block(true);
  auto x = b->pcoord->x1v / 1000.;
  auto rho = (2. - 0.7 * x + 0.3 * x * x).view({1, 1, -1}),
       exact = (-0.7 + 0.6 * x) / 1000.;
  int is = b->pcoord->il(), iu = b->pcoord->iu();
  EXPECT_LT(dev(x1_cov_one_sided(rho, b->pcoord->x1v, is, 1, true),
                exact[is].item<double>()),
            1.e-15);
  EXPECT_LT(dev(x1_cov_one_sided(rho, b->pcoord->x1v, iu, -1, true),
                exact[iu].item<double>()),
            1.e-15);
  EXPECT_LT(dev(x1_cov_d1_centred(rho, b->pcoord->x1v, true)
                        .view({-1})
                        .narrow(0, 1, 28) -
                    exact.narrow(0, 1, 28),
                0.),
            1.e-15);
}
