#pragma once

// C/C++
#include <array>
#include <functional>
#include <map>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include "bc.hpp"
#include "bc_func.hpp"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct CoordinateOptionsImpl;
using CoordinateOptions = std::shared_ptr<CoordinateOptionsImpl>;

class MeshBlockImpl;

struct InternalBoundaryOptionsImpl {
  static constexpr int MAXRUN = 4;

  static std::shared_ptr<InternalBoundaryOptionsImpl> create() {
    return std::make_shared<InternalBoundaryOptionsImpl>();
  }
  static std::shared_ptr<InternalBoundaryOptionsImpl> from_yaml(
      std::string const &filename);
  static std::shared_ptr<InternalBoundaryOptionsImpl> from_yaml(
      const YAML::Node &node);

  InternalBoundaryOptionsImpl() = default;
  std::shared_ptr<InternalBoundaryOptionsImpl> clone() const {
    return std::make_shared<InternalBoundaryOptionsImpl>(*this);
  }
  void report(std::ostream &os) const {
    os << "-- internal boundary options --\n";
    os << "* MAXRUN = " << MAXRUN << "\n"
       << "* max_iter = " << max_iter() << "\n"
       << "* solid_density = " << solid_density() << "\n"
       << "* solid_pressure = " << solid_pressure() << "\n";
  }

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, solid_density) = 1.e3;
  ADD_ARG(double, solid_pressure) = 1.e9;
};
using InternalBoundaryOptions = std::shared_ptr<InternalBoundaryOptionsImpl>;

class InternalBoundaryImpl : public torch::nn::Cloneable<InternalBoundaryImpl> {
 public:
  //! Create and register a InternalBoundary module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `InternalBoundary` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `InternalBoundary` module
   */
  static std::shared_ptr<InternalBoundaryImpl> create(
      InternalBoundaryOptions const &opts, torch::nn::Module *p,
      std::string const &name = "ib");

  //! options with which this `InternalBoundary` was constructed
  InternalBoundaryOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const *pmb = nullptr;

  //! Constructor to initialize the layers
  InternalBoundaryImpl() : options(InternalBoundaryOptionsImpl::create()) {}
  explicit InternalBoundaryImpl(InternalBoundaryOptions const &options,
                                torch::nn::Module *p = nullptr);
  void reset() override;

  //! Mark the solid cells
  /*!
   * \param w       primitive states
   * \param solid   internal solid boundary in [0, 1]
   */
  void mark_prim_solid_(torch::Tensor w, torch::Tensor solid) const;

  //! Mark the solid cells
  /*!
   * \param u       conserved states
   * \param solid   internal solid boundary in [0, 1]
   */
  void fill_cons_solid_(torch::Tensor u, torch::Tensor solid,
                        torch::Tensor fill) const;

  //! Rectify the solid cells
  /*!
   * \param solid_in internal solid boundary in [0, 1]
   * \param total_num_flips total number of flips
   * \param bfuncs boundary functions
   * \return rectified internal solid boundary
   */
  torch::Tensor rectify_solid(torch::Tensor solid_in, int &total_num_flips,
                              std::vector<bcfunc_t> const &bfuncs = {}) const;

  //! Revise the left/right states
  /*!
   * \param wlr primitive left/right states
   * \param solid internal solid boundary in [0, 1]
   * \return revised primitive left/right states
   */
  torch::Tensor forward(torch::Tensor wlr, int dim, torch::Tensor solid) const;

  //! x2/x3 solid walls: one wall frame per fluid run with a solid end
  struct Mirror {
    std::vector<std::array<int, 2>> span;
    std::vector<torch::Tensor> lines, src, sign;
    torch::Tensor cov_keep;
  };
  //! the walls of other["solid"] (nullptr without one)
  Mirror const *mirror_index(std::map<std::string, torch::Tensor> const &other,
                             int dim) const;
  //! recon of v, each wall frame's faces from recon of its frame
  torch::Tensor reconstruct(
      torch::Tensor v, Mirror const &m, int dim, int normal,
      std::function<torch::Tensor(torch::Tensor const &)> const &recon) const;

  //! x1 wall at each column's first fluid cell, grouped by depth
  struct X1Group {
    torch::Tensor depth;              // [nc3, nc2] int64, CPU
    std::vector<int> depths;          // distinct, ascending
    std::vector<torch::Tensor> cols;  // flat column indices per depth
    bool uniform() const { return depths.size() == 1 && depths[0] == 0; }
  };
  //! per-column (cell) and per-face (x2, x3) groups; rest: other solids
  struct X1Wall {
    X1Group cell, face2, face3;
    torch::Tensor rest;
  };
  //! the x1 wall of other["solid"], built once per mask (nullptr without one)
  X1Wall const *x1_wall(
      std::map<std::string, torch::Tensor> const &other) const;
  //! fn on each depth's columns, x1 narrowed so il is the surface
  torch::Tensor per_column(
      X1Group const &g, torch::Tensor const &t,
      std::function<torch::Tensor(torch::Tensor const &, int)> const &fn,
      bool zero_pad = false) const;
  //! reflecting-wall ghosts below each column's surface
  torch::Tensor x1_wall_fill(X1Group const &g, torch::Tensor const &v,
                             int normal = -1) const;

 private:
  mutable std::shared_ptr<X1Wall> x1wall_;
  mutable torch::Tensor x1wall_src_;  // held: its storage is not reused
  mutable int64_t x1wall_ver_ = -1;
  mutable std::map<int, Mirror> mirrors_;
  mutable torch::Tensor mirror_src_;  // held, as x1wall_src_
  mutable int64_t mirror_ver_ = -1;
};
TORCH_MODULE(InternalBoundary);

}  // namespace snap

#undef ADD_ARG
