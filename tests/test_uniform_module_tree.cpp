//! \file test_uniform_module_tree.cpp
//! #280 review item 2: a deck with no stretched x1 must carry the SAME module
//! tree it carried before route A existed.
//!
//! Route A's two tables and the passive scalar's own x1 reconstruction are of
//! no use to a uniform deck, and they are not free: `named_buffers(recurse)`
//! is copied wholesale into the user-forcing input dictionary
//! (src/mesh/meshblock.cpp, `stage_forcing_variables`), so every buffer added
//! here becomes a key every user forcing sees, and an extra `Reconstruct`
//! child changes `named_modules()` and every state dict built from it.
//!
//! What this file pins down, for two uniform decks:
//!   * no module in the tree carries a `cm_x1` / `cp_x1` buffer;
//!   * the scalar module has no `recon1` child, and the module it hands DIM1
//!     IS the one it hands DIM2/DIM3, as it was before #280;
//!   * the forcing-input dictionary of `meshblock.cpp` therefore holds no
//!     route A key.  It is read from the REAL call site: a scripted stage
//!     forcing that raises with the keys it was given.
//! and, on a stretched twin of the first deck, that all three DO appear -- the
//! lazy registration must still install, or the whole of route A is dead.
//!
//! With `SNAPY_TREE_DUMP=<dir>` in the environment each deck also writes its
//! sorted buffer names, module names and forcing-dictionary keys to that
//! directory.  The file uses no API newer than 2d9fa419, so the same binary
//! can be built in a tree at the base commit and the dumps diffed.

// C/C++
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// POSIX
#include <unistd.h>

// external
#include <gtest/gtest.h>

// torch
#include <torch/script.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/scalar/scalar.hpp>

// Route A does not exist at 2d9fa419, and neither do `stretched_x1()` and
// `has_x1_coeffs()`.  Everything that reads the uniform decks is outside this
// switch, so the same file builds in a tree at the base commit and writes the
// dumps the uniform lists are compared against.
#if defined(__has_include)
#if __has_include(<snap/recon/x1_coeffs.hpp>)
#define SNAPY_HAS_ROUTE_A 1
#endif
#endif

using namespace snap;

namespace {

//! a cartesian deck with `ntracer` passive tracers and no stretched x1 unless
//! `cells_extra` says so
std::string deck(char const* interp, int ntracer, int nx1,
                 std::string const& cells_extra = "") {
  std::string names = "[";
  for (int i = 0; i < ntracer; ++i) {
    names += (i ? ", tracer" : "tracer") + std::to_string(i);
  }
  names += "]";

  char buf[4096];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 16., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 8, nx3: 1, nghost: 3%s}
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
scalar:
  nvar: %d
  names: %s
  reconstruct: {type: %s, scale: false, shock: false}
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
                nx1, cells_extra.c_str(), interp, interp, ntracer,
                names.c_str(), interp);
  return buf;
}

std::shared_ptr<MeshBlockImpl> make_block(std::string const& config) {
  char fname[] = "/tmp/test-uniform-module-tree-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  return std::make_shared<MeshBlockImpl>(options);
}

std::vector<std::string> buffer_names(MeshBlockImpl const& block) {
  std::vector<std::string> names;
  for (auto const& item : block.named_buffers(/*recurse=*/true)) {
    names.push_back(item.key());
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> module_names(MeshBlockImpl& block) {
  std::vector<std::string> names;
  for (auto const& item : block.named_modules()) {
    names.push_back(item.key());
  }
  std::sort(names.begin(), names.end());
  return names;
}

//! A stage forcing that refuses to run and reports the keys it was handed.
/*!
 * The input dictionary is built at `stage_forcing_variables` in
 * meshblock.cpp; reading it from a forcing is the only way to see exactly
 * what a user sees, rather than a reconstruction of it.
 */
std::filesystem::path key_reporting_forcing() {
  auto path = std::filesystem::temp_directory_path() / "snapy_280_keys.pt";
  torch::jit::Module module("snapy_280_keys");
  module.define(
      "def forward(self, variables: Dict[str, Tensor], dt: float, stage: int) "
      "-> Dict[str, Tensor]:\n"
      "  msg = \"FORCINGKEYS<\" + \"|\".join(sorted(list(variables.keys()))) "
      "+ \">\"\n"
      // `raise` needs a name the C++ resolver does not have; an assertion
      // that can never hold carries the message just as well
      "  assert dt < -1.0, msg\n"
      "  return {\"hydro_du\": torch.zeros_like(variables[\"hydro_u\"])}\n");
  module.save(path.string());
  return path;
}

//! the keys of the forcing-input dictionary of one advance
std::vector<std::string> forcing_keys(std::shared_ptr<MeshBlockImpl> block) {
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.);
  vars["hydro_w"][IPR].fill_(3.);
  if (block->pscalar->nvar() > 0) {
    // `initialize` seeds scalar_s from this (meshblock.cpp,
    // finalize_initialization)
    vars["scalar_r"] = torch::full({block->pscalar->nvar(), nc3, nc2, nc1},
                                   1.e-3, torch::dtype(torch::kFloat64));
  }
  block->initialize(vars);

  auto forcing = key_reporting_forcing();
  block->set_user_stage_forcings({forcing.string()});

  std::string msg;
  try {
    block->advance_local(vars, 0., 0);
  } catch (std::exception const& exc) {
    msg = exc.what();
  }
  std::filesystem::remove(forcing);

  // the message echoes the source line before the evaluated text, and the
  // source contains the marker too: take the LAST one
  auto lo = msg.rfind("FORCINGKEYS<");
  EXPECT_NE(lo, std::string::npos)
      << "the stage forcing never ran, so the input dictionary was never "
         "built; message was: "
      << msg;
  if (lo == std::string::npos) return {};
  lo += std::string("FORCINGKEYS<").size();
  auto hi = msg.find('>', lo);
  EXPECT_NE(hi, std::string::npos);

  std::vector<std::string> keys;
  std::string field = msg.substr(lo, hi - lo);
  size_t at = 0;
  while (at <= field.size()) {
    auto bar = field.find('|', at);
    if (bar == std::string::npos) bar = field.size();
    if (bar > at) keys.push_back(field.substr(at, bar - at));
    at = bar + 1;
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

void dump(std::string const& tag, char const* what,
          std::vector<std::string> const& names) {
  char const* dir = std::getenv("SNAPY_TREE_DUMP");
  if (dir == nullptr) return;
  std::ofstream os(std::string(dir) + "/" + tag + "." + what);
  for (auto const& n : names) os << n << "\n";
}

bool ends_with(std::string const& s, char const* tail) {
  std::string t(tail);
  return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

//! every route A buffer anywhere in the tree
std::vector<std::string> route_a_buffers(std::vector<std::string> const& names) {
  std::vector<std::string> hits;
  for (auto const& n : names) {
    if (ends_with(n, "cm_x1") || ends_with(n, "cp_x1")) hits.push_back(n);
  }
  return hits;
}

std::string joined(std::vector<std::string> const& v) {
  std::string s;
  for (auto const& e : v) s += (s.empty() ? "" : ", ") + e;
  return s;
}

bool has_child(torch::nn::Module& m, char const* name) {
  for (auto const& item : m.named_children()) {
    if (item.key() == name) return true;
  }
  return false;
}

struct Case {
  char const* tag;
  char const* interp;
  int ntracer;
  int nx1;
};

constexpr Case kUniform[] = {{"uniform_weno5", "weno5", 1, 16},
                             {"uniform_cp5", "cp5", 2, 20}};

}  // namespace

//! The measured lists, written out for a diff against the same binary built
//! at 2d9fa419, and checked for the two things route A could have added.
TEST(UniformModuleTree, a_uniform_deck_gains_no_route_a_buffer) {
  for (auto const& c : kUniform) {
    auto block = make_block(deck(c.interp, c.ntracer, c.nx1));
#ifdef SNAPY_HAS_ROUTE_A
    ASSERT_FALSE(block->pcoord->options->stretched_x1())
        << c.tag << " is supposed to be a uniform deck";
#endif

    auto buffers = buffer_names(*block);
    auto modules = module_names(*block);
    auto keys = forcing_keys(block);

    dump(c.tag, "buffers", buffers);
    dump(c.tag, "modules", modules);
    dump(c.tag, "forcingkeys", keys);

    // the dictionary is the buffers plus the step's own variables, so it
    // cannot be smaller than the buffer list; this catches a parse that
    // silently read the wrong thing
    EXPECT_TRUE(std::includes(keys.begin(), keys.end(), buffers.begin(),
                              buffers.end()))
        << c.tag << ": the forcing dictionary does not contain every buffer";

    EXPECT_TRUE(route_a_buffers(buffers).empty())
        << c.tag << ": a uniform deck carries route A buffers: "
        << joined(route_a_buffers(buffers));
    EXPECT_TRUE(route_a_buffers(keys).empty())
        << c.tag << ": a user forcing on a uniform deck is handed route A "
                    "keys: "
        << joined(route_a_buffers(keys));
  }
}

//! Before #280 the scalar module reconstructed x1, x2 and x3 through one
//! `Reconstruct`.  A uniform deck has no reason to hold two.
TEST(UniformModuleTree, a_uniform_deck_gains_no_scalar_recon1) {
  for (auto const& c : kUniform) {
    auto block = make_block(deck(c.interp, c.ntracer, c.nx1));
    ASSERT_GT(block->pscalar->nvar(), 0) << c.tag;
    EXPECT_FALSE(has_child(*block->pscalar, "recon1"))
        << c.tag
        << ": the scalar module carries a second Reconstruct that a uniform "
           "deck never needs";
  }
}

#ifdef SNAPY_HAS_ROUTE_A

//! The lazy path must still install, or route A never runs at all.
TEST(UniformModuleTree, a_stretched_deck_still_installs_everything) {
  auto block = make_block(deck("weno5", 1, 16, ", x1rat: 1.05"));
  ASSERT_TRUE(block->pcoord->options->stretched_x1());

  auto buffers = buffer_names(*block);
  auto hits = route_a_buffers(buffers);
  // hydro recon1 (interp1 + interp2) and the scalar's own, two tables each
  EXPECT_EQ(hits.size(), 8u) << "installed route A buffers: " << joined(hits);
  EXPECT_TRUE(has_child(*block->pscalar, "recon1"))
      << "a stretched deck's tracers have no x1 reconstruction of their own";

  dump("stretched_weno5", "buffers", buffers);
  dump("stretched_weno5", "modules", module_names(*block));
}

//! A clone must come out with the same buffers it went in with: `Cloneable`
//! compares the counts after `reset()`, so a table registered outside
//! `reset()` has to be re-registered there.
TEST(UniformModuleTree, clone_keeps_the_installed_tables) {
  auto block = make_block(deck("weno5", 1, 16, ", x1rat: 1.05"));
  auto copy = std::dynamic_pointer_cast<MeshBlockImpl>(block->clone());
  ASSERT_NE(copy, nullptr);

  auto before = route_a_buffers(buffer_names(*block));
  auto after = route_a_buffers(buffer_names(*copy));
  EXPECT_EQ(before, after);

  for (auto const& p : {copy->phydro->precon1->pinterp1,
                        copy->phydro->precon1->pinterp2}) {
    EXPECT_TRUE(p->has_x1_coeffs())
        << p->options->type() << " lost its table in the clone";
  }
}

#endif  // SNAPY_HAS_ROUTE_A

//! and a uniform deck clones with no route A buffer either
TEST(UniformModuleTree, clone_of_a_uniform_deck_stays_empty) {
  auto block = make_block(deck("weno5", 1, 16));
  auto copy = std::dynamic_pointer_cast<MeshBlockImpl>(block->clone());
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(route_a_buffers(buffer_names(*copy)).empty());
  EXPECT_EQ(buffer_names(*block), buffer_names(*copy));
}
