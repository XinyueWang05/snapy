#pragma once

// C/C++
#include <map>
#include <tuple>

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

  //! Mirror ghosts of the solid cells along w-dimension `dim` (3 = x1): each
  //! solid cell within `reach` of a fluid face takes the reflection of the
  //! fluid across that face (gather index `idx`, odd reflection count `odd`).
  //! x1 (bottom_only) mirrors the bottom-attached run only, from above, and
  //! `surf` holds its depth per column; elsewhere solid runs thinner than
  //! 2 reach get two copies (alternate runs seen from below) and `pick` holds
  //! the copy each cell's stencil reads.
  struct Mirror {
    torch::Tensor idx, odd, pick, surf;  // idx/odd: [copies, nc3, nc2, nc1]
    int copies = 1;
  };
  Mirror const &mirror_index(torch::Tensor solid, int dim, int reach,
                             bool bottom_only) const;
  //! v [..., nc3, nc2, nc1] gathered from copy `copy`; row `normal` of a
  //! 4-d v (the normal velocity) changes sign under an odd reflection
  torch::Tensor mirror(torch::Tensor v, Mirror const &m, int copy, int dim,
                       int normal = -1) const;
  //! face states (ILT, IRT stacked first) from two copies: each side from
  //! the copy its source cell reads (ILT from cell i-1, IRT from cell i)
  torch::Tensor pick_lr(torch::Tensor lr0, torch::Tensor lr1, Mirror const &m,
                        int dim) const;

 private:
  mutable std::map<std::tuple<int, int, bool>, Mirror> mirrors_;
  mutable void const *mirror_src_ = nullptr;
  mutable int64_t mirror_ver_ = -1;
};
TORCH_MODULE(InternalBoundary);

}  // namespace snap

#undef ADD_ARG
