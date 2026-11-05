// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// POSIX
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

#include "cuda_test_gate.hpp"

using namespace snap;

// #280 stage 1, items (1)-(2) on a stretched x1: the diffusion operator and
// the BotHeat/TopCool layer. Every field below is linear (or a product of
// linear factors), so the exact answer is known cell by cell and holds on any
// grid; the tolerances are roundoff. The conductive rest state uses a minimal
// in-test setup: the rest-state helper of the oracle tests does not exist yet.

namespace {

// a 1000 to 3000 column of 24 cells growing upward by 1.1 (or uniform)
std::string card(bool stretched, int nx2) {
  return std::string(
             "reference-state: {Tref: 300., Pref: 1.e5}\n"
             "species:\n"
             "  - name: dry\n"
             "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
             "    cv_R: 2.5\n"
             "geometry:\n"
             "  type: cartesian\n"
             "  bounds: {x1min: 1000., x1max: 3000., x2min: 0., x2max: 4., "
             "x3min: 0., x3max: 1.}\n"
             "  cells: {nx1: 24, nx2: ") +
         std::to_string(nx2) + ", nx3: 1, nghost: 3" +
         (stretched ? ", x1rat: 1.1" : "") +
         "}\n"
         "dynamics:\n"
         "  equation-of-state: {type: ideal-gas}\n"
         "  reconstruct:\n"
         "    vertical: {type: weno5, scale: false, shock: false}\n"
         "    horizontal: {type: weno5, scale: false, shock: false}\n"
         "  riemann-solver: {type: lmars}\n"
         "forcing:\n"
         "  const-gravity: {grav1: -9.8}\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: reflecting\n"
         "    x1-outer: reflecting\n";
}

std::string write_temp(std::string const& text) {
  char fname[] = "/tmp/test-stretched-diffusion-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << text;
  return fname;
}

std::shared_ptr<MeshBlockImpl> make_block(bool stretched, int nx2,
                                          torch::Device device) {
  auto fname = write_temp(card(stretched, nx2));
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname.c_str());
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, torch::kFloat64);
  return block;
}

std::vector<torch::Device> devices() {
  std::vector<torch::Device> out = {torch::kCPU};
  if (snapy_cuda_test_enabled()) out.emplace_back(torch::kCUDA, 0);
  return out;
}

//! x1, x2 at the cell centres, ghosts included, broadcast to (nc3, nc2, nc1)
std::pair<torch::Tensor, torch::Tensor> centres(MeshBlockImpl const& block) {
  auto c = block.pcoord;
  auto shape = std::vector<int64_t>{c->options->nc3(), c->options->nc2(),
                                    c->options->nc1()};
  return {c->x1v.view({1, 1, -1}).expand(shape),
          c->x2v.view({1, -1, 1}).expand(shape)};
}

//! a resting isothermal state, ghosts included, at the block's options
torch::Tensor make_state(MeshBlockImpl const& block) {
  auto c = block.pcoord;
  auto w = torch::zeros({block.phydro->peos->nvar(), c->options->nc3(),
                         c->options->nc2(), c->options->nc1()},
                        c->x1v.options());
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  return w;
}

//! the active cells of a (nvar, nc3, nc2, nc1) or (nc3, nc2, nc1) tensor
torch::Tensor interior(MeshBlockImpl& block, torch::Tensor const& v) {
  auto part = PartOptions().exterior(false);
  if (v.dim() == 3) part.ndim(3);
  return v.index(block.part({0, 0, 0}, part));
}

double max_abs(torch::Tensor const& v) { return v.abs().max().item<double>(); }

DiffusionOptions diffusion(std::string const& yaml) {
  return DiffusionOptionsImpl::from_yaml(YAML::Load(yaml));
}

}  // namespace

// ---- item (1), RED/GREEN test 1: viscosity of a linear velocity ----------

// vx = a x1: the dilatation is a in every cell and the x1 stress 4a/3 at every
// face, so the momentum flux is uniform (du[IVX] = 0) and the energy flux
// -nu (4a/3) a x1 has divergence -(4/3) nu a^2. Needs the centred x1
// derivative over x1v[i+1] - x1v[i-1] and a distance-weighted face velocity.
TEST(StretchedDiffusion, viscosity_of_a_linear_velocity_is_exact) {
  double const a = 1.e-3, nu = 2., dt = 0.5;
  for (auto device : devices()) {
    auto block = make_block(true, 1, device);
    ASSERT_TRUE(block->pcoord->options->stretched_x1());
    Diffusion diff(diffusion("diffusion: {nu_iso: 2., dynamic: true}"),
                   block->phydro.get());
    auto w = make_state(*block);
    auto [x1, x2] = centres(*block);
    w[IVX].copy_(a * x1);
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = diff->forward(torch::zeros_like(w), w, temp, dt);

    double heat = dt * nu * (4. / 3.) * a * a;
    EXPECT_LT(max_abs(interior(*block, du[IVX])), 1.e-12) << device;
    EXPECT_LT(max_abs(interior(*block, du[IPR]) - heat), 1.e-10 * heat)
        << device;
  }
}

// vx = a x1, vy = b x1 x2: the x1 derivative of vy enters the x2-face shear
// stress through the centred x1 derivative, and the dilatation a + b x1 varies
// in x1. du[IVX] = dt nu b / 3 and du[IVY] = 0 in every cell.
TEST(StretchedDiffusion, viscous_cross_terms_of_a_linear_shear_are_exact) {
  double const a = 1.e-3, b = 2.e-4, nu = 2., dt = 0.5;
  for (auto device : devices()) {
    auto block = make_block(true, 4, device);
    Diffusion diff(diffusion("diffusion: {nu_iso: 2., dynamic: true}"),
                   block->phydro.get());
    auto w = make_state(*block);
    auto [x1, x2] = centres(*block);
    w[IVX].copy_(a * x1);
    w[IVY].copy_(b * x1 * x2);
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = diff->forward(torch::zeros_like(w), w, temp, dt);

    double expected = dt * nu * b / 3.;
    EXPECT_LT(max_abs(interior(*block, du[IVX]) - expected), 1.e-10 * expected)
        << device;
    EXPECT_LT(max_abs(interior(*block, du[IVY])), 1.e-13) << device;
  }
}

// ---- item (1), RED/GREEN test 2: conduction through a linear density -----

// T = T0 + G x1 and rho = r0 + r1 x1 (kinematic kappa): the face conductivity
// kappa cv rho(x1f) is exact only when the face density is distance-weighted;
// then du[IPR] = dt kappa cv G r1 in every cell, the wall cells included.
TEST(StretchedDiffusion, conduction_through_a_linear_density_is_exact) {
  double const kappa = 3., G = -2.e-3, r1 = -1.e-4, dt = 0.5;
  for (auto device : devices()) {
    auto block = make_block(true, 1, device);
    Diffusion diff(diffusion("diffusion: {kappa_iso: 3.}"),
                   block->phydro.get());
    auto w = make_state(*block);
    auto [x1, x2] = centres(*block);
    w[IDN].copy_(1. + r1 * (x1 - 1000.));
    auto temp = 300. + G * (x1 - 1000.);
    auto cv = block->phydro->peos->specific_heat_cv(w, temp);
    ASSERT_LT(max_abs(cv - cv.flatten()[0]), 1.e-12 * max_abs(cv));
    auto du = diff->forward(torch::zeros_like(w), w, temp, dt);

    double expected = dt * kappa * cv.flatten()[0].item<double>() * G * r1;
    // roundoff: T ~ 300 differenced over ~20 m, times kappa rho cv ~ 2e3
    EXPECT_LT(max_abs(interior(*block, du[IPR]) - expected),
              1.e-8 * std::abs(expected))
        << device;
  }
}

// The conductive rest state: a dynamic kappa, the wall temperatures fixed and
// T linear between them over a stratified density. Every face carries the
// same flux, so du = 0 to roundoff on any grid.
TEST(StretchedDiffusion, conductive_rest_state_between_fixed_walls) {
  double const ti = 320., to = 260.;
  for (auto device : devices()) {
    auto block = make_block(true, 1, device);
    Diffusion diff(
        diffusion("diffusion: {kappa_iso: 3., dynamic: true, "
                  "temperature_inner: 320., temperature_outer: 260.}"),
        block->phydro.get());
    auto w = make_state(*block);
    auto [x1, x2] = centres(*block);
    w[IDN].copy_(torch::exp(-(x1 - 1000.) / 8.e3));
    auto temp = ti + (to - ti) * (x1 - 1000.) / 2000.;
    auto du = diff->forward(torch::zeros_like(w), w, temp, 1.);
    // a flux of kappa (ti - to) / 2000 = 0.09 per face
    EXPECT_LT(max_abs(interior(*block, du)), 1.e-12) << device;
  }
}

// ---- item (2), RED/GREEN test 1: the energy integral of the layer --------

// thickness: a layer of D in x1 length units. Whatever the grid, the column
// gains exactly flux * dt: sum_i du_i dx1f_i = flux * dt.
TEST(StretchedHeatLayer, energy_integral_is_the_flux) {
  double const dt = 0.25;
  for (auto device : devices()) {
    for (bool stretched : {true, false}) {
      auto block = make_block(stretched, 1, device);
      auto c = block->pcoord;
      int il = c->il(), iu = c->iu();
      auto w = make_state(*block);
      auto temp = block->phydro->peos->compute("W->T", {w});
      auto width = c->dx1f.slice(0, il, iu + 1);
      for (double d : {230., 7., 2000.}) {
        BotHeat bot(BotHeatOptionsImpl::from_yaml(
                        YAML::Load("bot-heat: {flux: 100., thickness: " +
                                   std::to_string(d) + "}")),
                    block->phydro.get());
        TopCool top(TopCoolOptionsImpl::from_yaml(
                        YAML::Load("top-cool: {flux: -40., thickness: " +
                                   std::to_string(d) + "}")),
                    block->phydro.get());
        auto du = bot->forward(torch::zeros_like(w), w, temp, dt);
        auto energy = (du[IPR][0][0].slice(0, il, iu + 1) * width).sum();
        EXPECT_NEAR(energy.item<double>(), 100. * dt, 1.e-12 * 100. * dt)
            << device << " stretched " << stretched << " D " << d;
        du = top->forward(torch::zeros_like(w), w, temp, dt);
        energy = (du[IPR][0][0].slice(0, il, iu + 1) * width).sum();
        EXPECT_NEAR(energy.item<double>(), -40. * dt, 1.e-12 * 40. * dt)
            << device << " stretched " << stretched << " D " << d;
      }
    }
  }
}

// ---- item (2), RED/GREEN test 2: the profile of the layer ----------------

// Each cell takes flux / D times the fraction of its width inside the layer:
// a cell inside takes flux / D, the cell the layer edge cuts its share, cells
// beyond nothing; a layer thinner than the first cell sits in that cell.
TEST(StretchedHeatLayer, each_cell_takes_its_overlap) {
  double const dt = 0.25, flux = 100.;
  for (auto device : devices()) {
    auto block = make_block(true, 1, device);
    auto c = block->pcoord;
    int il = c->il(), iu = c->iu();
    auto w = make_state(*block);
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto xf = c->x1f.cpu();
    double x1min = xf[il].item<double>(), x1max = xf[iu + 1].item<double>();
    ASSERT_DOUBLE_EQ(x1min, 1000.);
    ASSERT_DOUBLE_EQ(x1max, 3000.);
    ASSERT_LT(xf[il + 1].item<double>() - x1min, 230.);  // a partial cell

    for (double d : {230., 7.}) {
      BotHeat bot(
          BotHeatOptionsImpl::from_yaml(YAML::Load(
              "bot-heat: {flux: 100., thickness: " + std::to_string(d) + "}")),
          block->phydro.get());
      TopCool top(
          TopCoolOptionsImpl::from_yaml(YAML::Load(
              "top-cool: {flux: -100., thickness: " + std::to_string(d) + "}")),
          block->phydro.get());
      auto bot_du = bot->forward(torch::zeros_like(w), w, temp, dt).cpu();
      auto top_du = top->forward(torch::zeros_like(w), w, temp, dt).cpu();
      int cut_bot = 0, cut_top = 0;
      for (int i = il; i <= iu; ++i) {
        double lo = xf[i].item<double>(), hi = xf[i + 1].item<double>();
        double in_bot = std::max(0., std::min(hi, x1min + d) - lo);
        double in_top = std::max(0., hi - std::max(lo, x1max - d));
        cut_bot += in_bot > 0. && in_bot < hi - lo;
        cut_top += in_top > 0. && in_top < hi - lo;
        double q = dt * flux / d;
        EXPECT_NEAR(bot_du[IPR][0][0][i].item<double>(), q * in_bot / (hi - lo),
                    1.e-12 * q)
            << device << " D " << d << " cell " << i;
        EXPECT_NEAR(top_du[IPR][0][0][i].item<double>(),
                    -q * in_top / (hi - lo), 1.e-12 * q)
            << device << " D " << d << " cell " << i;
      }
      EXPECT_EQ(cut_bot, 1) << "D " << d;
      EXPECT_EQ(cut_top, 1) << "D " << d;
      // only the energy row moves
      bot_du[IPR].zero_();
      top_du[IPR].zero_();
      EXPECT_EQ(max_abs(bot_du), 0.);
      EXPECT_EQ(max_abs(top_du), 0.);
    }
  }
}

// The layer is placed in the global x1: with x1 split into two blocks, each
// block takes the overlap of its own cells, and the column gains flux * dt.
TEST(StretchedHeatLayer, a_layer_spans_two_x1_blocks) {
  double const dt = 0.25;
  auto fname = write_temp(
      "distribute: {layout: cubed, nb1: 2, blocks_per_process: 2, backend: "
      "gloo}\n" +
      card(true, 1));
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(fname));
  std::remove(fname.c_str());
  ASSERT_EQ(mesh->blocks.size(), 2);
  double bot_total = 0., top_total = 0.;
  for (auto block : mesh->blocks) {
    auto c = block->pcoord;
    int il = c->il(), iu = c->iu();
    auto w = make_state(*block);
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto width = c->dx1f.slice(0, il, iu + 1);
    BotHeat bot(BotHeatOptionsImpl::from_yaml(
                    YAML::Load("bot-heat: {flux: 100., thickness: 1600.}")),
                block->phydro.get());
    TopCool top(TopCoolOptionsImpl::from_yaml(
                    YAML::Load("top-cool: {flux: -100., thickness: 1600.}")),
                block->phydro.get());
    auto du = bot->forward(torch::zeros_like(w), w, temp, dt);
    double bot_e =
        (du[IPR][0][0].slice(0, il, iu + 1) * width).sum().item<double>();
    du = top->forward(torch::zeros_like(w), w, temp, dt);
    double top_e =
        (du[IPR][0][0].slice(0, il, iu + 1) * width).sum().item<double>();
    // each block holds part of each layer
    EXPECT_GT(bot_e, 0.);
    EXPECT_LT(bot_e, 100. * dt);
    EXPECT_LT(top_e, 0.);
    EXPECT_GT(top_e, -100. * dt);
    bot_total += bot_e;
    top_total += top_e;
  }
  EXPECT_NEAR(bot_total, 100. * dt, 1.e-12 * 100. * dt);
  EXPECT_NEAR(top_total, -100. * dt, 1.e-12 * 100. * dt);
}

// An integer depth counts cells, so on a stretched x1 it is no fixed layer:
// refused there, with thickness named. A thickness beyond the column and a
// card giving both keys are refused on any grid.
TEST(StretchedHeatLayer, refusals) {
  auto stretched = make_block(true, 1, torch::kCPU);
  auto uniform = make_block(false, 1, torch::kCPU);
  auto refuses = [](auto make, std::string const& message) {
    try {
      make();
      ADD_FAILURE() << "did not refuse: " << message;
    } catch (c10::Error const& e) {
      EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
          << e.what();
    }
  };
  for (auto key :
       {"bot-heat: {flux: 1., depth: 2}", "top-cool: {flux: -1., depth: 2}"}) {
    bool bot = std::string(key).find("bot") == 0;
    auto build = [&](std::shared_ptr<MeshBlockImpl> const& block) {
      if (bot)
        BotHeat(BotHeatOptionsImpl::from_yaml(YAML::Load(key)),
                block->phydro.get());
      else
        TopCool(TopCoolOptionsImpl::from_yaml(YAML::Load(key)),
                block->phydro.get());
    };
    refuses([&] { build(stretched); }, "give thickness");
    EXPECT_NO_THROW(build(uniform));
  }
  refuses(
      [&] {
        BotHeat(BotHeatOptionsImpl::from_yaml(
                    YAML::Load("bot-heat: {flux: 1., thickness: 2000.5}")),
                uniform->phydro.get());
      },
      "thickness exceeds the x1 column");
  refuses(
      [&] {
        TopCoolOptionsImpl::from_yaml(
            YAML::Load("top-cool: {flux: -1., depth: 2, thickness: 30.}"));
      },
      "not both");
  refuses(
      [&] {
        BotHeatOptionsImpl::from_yaml(
            YAML::Load("bot-heat: {flux: 1., thickness: 0.}"));
      },
      "thickness must be");
}
