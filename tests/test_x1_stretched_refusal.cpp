//! \file test_x1_stretched_refusal.cpp
//! #280 review F3: a stretched x1 is REFUSED for the five interpolators route
//! A does not cover (dc, plm, ppm, cp3, weno3).  Before this, `set_x1_coeffs`
//! was a silent no-op on them, so a deck with `cells/x1rat` or `geometry/x1f`
//! ran uniform-grid arithmetic on a non-uniform grid and said nothing.
//!
//! The refusal is installed where the rows are installed -- at configure time,
//! in `Reconstruct::set_x1_faces` -> `Interp::set_x1_coeffs` -- so a deck that
//! cannot be reconstructed fails when the block is built, not per call.
//!
//! Four things are pinned here: the five refusals at module level, the same
//! five at deck level (the Mesh does not build), weno5 and cp5 still
//! installing and running on a stretched x1, and every uniform deck -- all
//! seven types -- building exactly as before.

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

// POSIX
#include <unistd.h>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>

using namespace snap;

namespace {

constexpr int kNx1 = 16, kNghost = 3;
constexpr double kX1min = 0., kX1max = 16.;

// x1 is the last dimension of (nvar, nc3, nc2, nc1)
enum { DIM1 = 3 };

//! the five types route A does not cover (note Sec. 4.7)
std::vector<std::string> const kRefused = {"dc", "plm", "ppm", "cp3", "weno3"};
//! the two it does
std::vector<std::string> const kCovered = {"weno5", "cp5"};

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

//! A minimal cartesian deck; `type` is the x1 ("vertical") reconstruction and
//! `cells_extra` carries the #280 grid key when the grid is stretched.
std::string deck(std::string const& type, std::string const& cells_extra) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: %g, x1max: %g, x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 1, nx3: 1, nghost: %d%s}
dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
  reconstruct:
    vertical: {type: %s, scale: false, shock: false}
    horizontal: {type: %s, scale: false, shock: false}
  riemann-solver:
    type: lmars
integration:
  type: rk3
  cfl: 0.4
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic
)",
                kX1min, kX1max, kNx1, kNghost, cells_extra.c_str(),
                type.c_str(), type.c_str());
  return buf;
}

Mesh make_mesh(std::string const& config) {
  char fname[] = "/tmp/test-x1-stretched-refusal-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);

  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);
  return Mesh(mesh_opts);
}

//! `x1rat` spelling of the grid key
std::string const kStretched = ", x1rat: 1.1";
std::string const kUniform = "";

//! What `f` threw, as a c10::Error, or the empty string if it returned.
std::string refusal_of(std::function<void()> const& f) {
  try {
    f();
  } catch (c10::Error const& e) {
    return e.what();
  }
  return "";
}

//! `type` was refused, by route A's own check, naming itself and the pair.
::testing::AssertionResult refused(std::string const& what,
                                   std::string const& type,
                                   std::string const& msg) {
  if (msg.empty()) {
    return ::testing::AssertionFailure()
           << what << " with x1 reconstruction '" << type
           << "' was accepted: it would run uniform-grid arithmetic on a "
              "stretched x1 and say nothing";
  }
  if (msg.find(type) == std::string::npos ||
      msg.find("weno5") == std::string::npos) {
    return ::testing::AssertionFailure()
           << what << " with '" << type
           << "' was refused, but not by route A's own check (the message "
              "must name the type and the covered pair): "
           << msg;
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Module level: the five uncovered types refuse a face vector.
// ---------------------------------------------------------------------------
TEST(StretchedX1Refusal, the_five_uncovered_types_refuse_a_face_vector) {
  auto x1f = faces_geometric(21, 1.03);

  for (auto const& type : kRefused) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);

    EXPECT_TRUE(refused("a face vector", type,
                        refusal_of([&] { recon->set_x1_faces(x1f); })));

    EXPECT_FALSE(recon->pinterp1->has_x1_coeffs()) << type;
  }
}

// ---------------------------------------------------------------------------
// 2. Module level: weno5 and cp5 install the rows and run on them.
// ---------------------------------------------------------------------------
TEST(StretchedX1Refusal, weno5_and_cp5_still_take_a_stretched_x1) {
  auto x1f = faces_geometric(21, 1.03);

  for (auto const& type : kCovered) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);

    ASSERT_NO_THROW(recon->set_x1_faces(x1f)) << type;
    EXPECT_TRUE(recon->pinterp1->has_x1_coeffs()) << type;
    EXPECT_TRUE(recon->pinterp2->has_x1_coeffs()) << type;

    // and the rows are actually walked: 20 cells for 21 faces
    auto w = torch::rand({5, 1, 1, 20}, torch::kFloat64) + 1.;
    auto wlr = recon->forward(w, DIM1);
    EXPECT_TRUE(torch::isfinite(wlr).all().item<bool>()) << type;
  }
}

// ---------------------------------------------------------------------------
// 3. Deck level: a stretched deck with an uncovered type does not build.
//    This is the point of F3 -- the refusal is at configure time.
// ---------------------------------------------------------------------------
TEST(StretchedX1Refusal, a_stretched_deck_with_an_uncovered_type_is_refused) {
  for (auto const& type : kRefused) {
    EXPECT_TRUE(refused("a stretched x1rat deck", type, refusal_of([&] {
                          make_mesh(deck(type, kStretched));
                        })));
  }
}

// ---------------------------------------------------------------------------
// 4. Deck level: a stretched weno5/cp5 deck builds and holds the rows.
// ---------------------------------------------------------------------------
TEST(StretchedX1Refusal, a_stretched_deck_with_a_covered_type_still_builds) {
  for (auto const& type : kCovered) {
    Mesh mesh = nullptr;
    ASSERT_NO_THROW(mesh = make_mesh(deck(type, kStretched))) << type;
    auto pmb = mesh->blocks.front();
    EXPECT_TRUE(pmb->phydro->precon1->pinterp1->has_x1_coeffs()) << type;
    EXPECT_TRUE(pmb->phydro->precon1->pinterp2->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon23->pinterp1->has_x1_coeffs()) << type;
  }
}

// ---------------------------------------------------------------------------
// 5. The uniform path is untouched: all seven types still build, and none of
//    them holds a table.
// ---------------------------------------------------------------------------
TEST(StretchedX1Refusal, every_uniform_deck_is_unaffected) {
  std::vector<std::string> all = kRefused;
  all.insert(all.end(), kCovered.begin(), kCovered.end());

  for (auto const& type : all) {
    Mesh mesh = nullptr;
    ASSERT_NO_THROW(mesh = make_mesh(deck(type, kUniform)))
        << "uniform " << type << " deck no longer builds";
    auto pmb = mesh->blocks.front();
    EXPECT_FALSE(pmb->phydro->precon1->pinterp1->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon1->pinterp2->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon23->pinterp1->has_x1_coeffs()) << type;
  }
}
