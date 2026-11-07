// Two processes, a stretched-x1 spherical-polar deck split into two x2 blocks
// with one block per process. The pole refusal reads the mesh's x2 bounds, so
// a deck touching a pole refuses on BOTH ranks, not only on the rank whose
// block holds the pole (the other one would go on and wait for it, #280). A
// pole-free deck split the same way runs.
//
// Run one case per process (tests/CMakeLists.txt), e.g.
// torchrun --no-python --nproc-per-node=2 test_sph_pole_split_mp.<build> \
//   --gtest_filter=*refuses_the_inner_pole*

// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

// torch
#include <torch/torch.h>
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

std::string deck(double x2min, double x2max) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5
geometry:
  type: spherical-polar
  bounds: {x1min: 5., x1max: 7., x2min: %.17g, x2max: %.17g, x3min: 0., x3max: 0.1}
  cells: {nx1: 16, nx2: 4, nx3: 1, nghost: 3, x1rat: 1.05}
dynamics:
  equation-of-state: {type: ideal-gas}
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: lmars}
integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)",
                x2min, x2max);
  return buf;
}

// two x2 blocks, one per process
Mesh make_split(double x2min, double x2max) {
  char fname[] = "/tmp/sph-pole-split-mp-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << deck(x2min, x2max);
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  block_opts->layout()->type() = "cubed";
  block_opts->layout()->px(2);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(torch::kCPU, torch::kFloat64);
  return mesh;
}

int env_rank() {
  char const* r = std::getenv("RANK");
  return r ? std::atoi(r) : 0;
}

// every rank refuses, with the pole-free message
void refuses_on_every_rank(double x2min, double x2max) {
  int rank = env_rank();
  std::string what;
  try {
    make_split(x2min, x2max);
  } catch (c10::Error const& e) {
    what = e.what_without_backtrace();
  }
  std::cout << "rank " << rank << ": "
            << (what.empty() ? "ran" : what.substr(0, what.find('\n')))
            << std::endl;
  EXPECT_NE(what.find("pole-free spherical-polar"), std::string::npos)
      << "rank " << rank << " did not refuse x2 = [" << x2min << ", " << x2max
      << "]";
}

}  // namespace

TEST(SphPoleSplitMp, pole_free_x2_split_runs) {
  auto mesh = make_split(M_PI / 2 - 0.2, M_PI / 2 + 0.2);
  auto layout = mesh->blocks.front()->get_layout();
  ASSERT_TRUE(layout->has_process_group() &&
              layout->options->process_world_size() == 2 &&
              mesh->blocks.size() == 1)
      << "the two x2 halves must be in two processes";
  auto coord = mesh->blocks.front()->pcoord;
  auto w = torch::zeros(
      {mesh->blocks.front()->phydro->peos->nvar(), coord->options->nc3(),
       coord->options->nc2(), coord->options->nc1()},
      coord->x1v.options());
  w[IDN].fill_(1.);
  w[IPR].fill_(1.);
  MeshVariables vars(1);
  vars[0]["hydro_w"] = w;
  mesh->initialize(vars);
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int n = 0; n < 3; ++n)
    for (int stage = 0; stage < nstage; ++stage)
      mesh->forward(vars, 1.e-3, stage);
  auto u = vars[0].at("hydro_u");
  EXPECT_TRUE(torch::isfinite(u).all().item<bool>())
      << "rank " << layout->options->process_rank();
}

TEST(SphPoleSplitMp, refuses_the_inner_pole_on_both_ranks) {
  refuses_on_every_rank(0., 0.4);
}

TEST(SphPoleSplitMp, refuses_the_outer_pole_on_both_ranks) {
  refuses_on_every_rank(M_PI - 0.4, M_PI);
}
