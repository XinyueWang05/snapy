// C/C++
#include <cstdio>
#include <fstream>

// POSIX
#include <unistd.h>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

// A cp5 deck that leaves nghost at its default of 1, while the five-point
// stencil reads 3 ghosts: it must refuse, or get a linear field's first face.
TEST(Cp5Nghost, refuses_or_reconstructs_the_first_x1_face) {
  char fname[] = "/tmp/test-cp5-nghost-XXXXXX";
  close(mkstemp(fname));
  std::ofstream(fname) << R"(
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, cv_R: 2.5}
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 16., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 16, nx2: 1, nx3: 1}
dynamics:
  equation-of-state: {type: ideal-gas}
  reconstruct:
    vertical: {type: cp5, scale: false, shock: false}
    horizontal: {type: cp5, scale: false, shock: false}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: periodic,
             x2-outer: periodic, x3-inner: periodic, x3-outer: periodic}
)";
  std::shared_ptr<MeshBlockImpl> block;
  try {
    block =
        std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(fname));
  } catch (c10::Error const&) {
    std::remove(fname);
    return;  // refused at setup
  }
  std::remove(fname);

  // w = 2 + x1, so every x1 face holds 2 + x1f; the first one is 2
  int ng = block->pcoord->options->nghost();
  auto w = (2. + block->pcoord->x1v).view({1, 1, 1, -1}).repeat({5, 1, 1, 1});
  auto wlr = block->phydro->precon1->forward(w, 3);
  EXPECT_DOUBLE_EQ(wlr[ILT][IDN][0][0][ng].item<double>(), 2.);
  EXPECT_DOUBLE_EQ(wlr[IRT][IDN][0][0][ng].item<double>(), 2.);
}
