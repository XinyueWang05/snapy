#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/gravity_work_radial.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct ImplicitOptionsImpl {
  static std::shared_ptr<ImplicitOptionsImpl> create() {
    return std::make_shared<ImplicitOptionsImpl>();
  }
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(const YAML::Node& node);

  ImplicitOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- implicit hydro options --\n";
    os << "* type = " << type() << "\n"
       << "* scheme = " << scheme() << "\n"
       << "* advection_cfl = " << advection_cfl() << "\n"
       << "* shear_cfl = " << shear_cfl() << "\n"
       << "* integrator = " << integrator() << "\n"
       << "* integrator_iterations = " << integrator_iterations() << "\n"
       << "* integrator_tolerance = " << integrator_tolerance() << "\n"
       << "* integrator_auto = " << integrator_auto() << "\n";
  }

  int size() const {
    if ((scheme() >> 3) & 1) {  // full
      return 5;
    } else {
      return 3;
    }
  }

  std::string type() const;
  ADD_ARG(int, scheme) = 0;
  ADD_ARG(double, advection_cfl) = 1.0;
  ADD_ARG(double, shear_cfl) = 0.0;
  //! "rk": the implicit correction inside each stage of the RK integrator.
  //! "hevi-ark2": vic-full only; MeshBlock advances the step with the
  //! ARK2 HEVI tableau (explicit horizontal, linearly implicit vertical)
  ADD_ARG(std::string, integrator) = "rk";
  //! the integrator was chosen without an implicit-integrator key: vic-full
  //! with integration/type rk3 runs hevi-ark2 (implicit-integrator: rk opts
  //! out and restores the rk path)
  ADD_ARG(bool, integrator_auto) = false;
  //! hevi-ark2: relative tolerance of the stage solves; < 0: key absent.
  //! With a tolerance > 0 the stage iterates until every conserved row's
  //! correction satisfies max|dD| <= tol max|D| + kHeviAtolEps eps scale
  //! (scale: the row's max |R|; sqrt(rho E) for the momenta)
  //! (at least kHeviMinSolves, the measured fixed-solve floor, at most
  //! hevi_max_solves(); past the minimum it also stops once the correction
  //! stops contracting); 0 switches the stop off and runs hevi_iterations()
  //! fixed solves
  ADD_ARG(double, integrator_tolerance) = -1.;
  //! hevi-ark2: solves per implicit stage. 1: one linearised solve about
  //! the predictor; n > 1: n - 1 further corrections towards the stage's
  //! implicit solution, D <- D + (I + g dt J)^-1 (g dt V(R + D) - D).
  //! 0 (default): the fixed count is kHeviDefaultIterations. On a rest column
  //! from the discretely balanced state, 1 solve grows at vertical acoustic
  //! Courant >= 1.65 and 3 solves at >= 197; 4 stay at round-off up to 250.
  //! 5: the smallest fixed count that matches converged ARS at dt0 on all 22
  //! linear-model arms (docs/user_guide/configuration.rst, Limits). The
  //! iteration does not converge monotonically in the count
 public:
  static constexpr int kHeviDefaultIterations = 5;
  static constexpr int kHeviMaxIterations = 8;
  static constexpr int kHeviMinSolves = 4;
  static constexpr double kHeviAtolEps = 1.e3;
  //! fixed solves per stage when the tolerance stop is off
  int hevi_iterations() const {
    return integrator_iterations() > 0 ? integrator_iterations()
                                       : kHeviDefaultIterations;
  }
  //! the tolerance in use (0: fixed hevi_iterations() solves). The stop is
  //! on only with implicit-integrator-tolerance > 0: by default the stages
  //! run the measured kHeviDefaultIterations fixed solves. Fewer than 4 grow
  //! on a balanced rest column at vertical acoustic Courant >= 65.6 (cubed
  //! tall column, face-work operator box); 5 and more redo on a 45-level,
  //! 40-scale-height column at dt 1500 s, as the rk path does (Limits)
  double hevi_tolerance() const {
    return integrator_tolerance() >= 0. ? integrator_tolerance() : 0.;
  }
  //! solves per stage at most (tolerance stop) or exactly (fixed)
  int hevi_max_solves() const {
    if (hevi_tolerance() > 0.)
      return integrator_iterations() > 0 ? integrator_iterations()
                                         : kHeviMaxIterations;
    return hevi_iterations();
  }
  ADD_ARG(int, integrator_iterations) = 0;
};
using ImplicitOptions = std::shared_ptr<ImplicitOptionsImpl>;

class HydroImpl;

class ImplicitHydroImpl : public torch::nn::Cloneable<ImplicitHydroImpl> {
 public:
  //! Create and register a ImplicitHydro module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `ImplicitHydro` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `ImplicitHydro` module
   */
  static std::shared_ptr<ImplicitHydroImpl> create(
      ImplicitOptions const& opts, torch::nn::Module* p,
      std::string const& name = "icorr");

  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! non-owning pointer to parent
  HydroImpl const* phydro = nullptr;

  //! Constructor to initialize the layer
  ImplicitHydroImpl() : options(ImplicitOptionsImpl::create()) {}
  explicit ImplicitHydroImpl(ImplicitOptions const& options,
                             torch::nn::Module* p = nullptr);
  void reset() override;

  //! corrector for the implicit hydro
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor gamma,
                        double dt);
  torch::Tensor forward_masked(torch::Tensor du, torch::Tensor w,
                               torch::Tensor gamma, double dt,
                               torch::Tensor solid);

  torch::Tensor correction() const { return _corr; }
  torch::Tensor mass_correction() const { return _mass_corr; }

  //! running max relative break of the VIC telescoping identity (clamp meter)
  torch::Tensor clamp_residual() const { return _clamp_residual; }

  //! 1 if the dry-gas clamp emptied a cell since the last reset, else 0
  torch::Tensor dry_clamp_step() const { return _dry_clamp_step; }
  void reset_dry_clamp_step() { _dry_clamp_step.zero_(); }

  //! Latched until the next step, including later RK stages. A failure
  //! found after the solve is latched on the device; this reads it back.
  bool solve_failed() const {
    return _solve_failed || (_failed_dev.defined() && _failed_dev.item<bool>());
  }
  void reset_solve_failure() {
    _solve_failed = false;
    if (_failed_dev.defined()) _failed_dev.zero_();
  }

  //! solve_failed() as a 0-d tensor (1 or 0) with `opts`, without a host read
  torch::Tensor solve_failed_tensor(torch::TensorOptions const& opts) const {
    auto flag = torch::full({}, _solve_failed ? 1. : 0., opts);
    return _failed_dev.defined() ? torch::maximum(flag, _failed_dev.to(opts))
                                 : flag;
  }

 private:
  void ensure_workspace(torch::Tensor const& w);
  bool _solve_failed = false;
  torch::Tensor _failed_dev;  // 0-d bool: a failure found after the solve

  torch::Tensor _a, _b, _c, _delta, _du0, _corr, _mass_corr, _clamp_residual,
      _dry_clamp_step;

  //! SNAP_GRAVITY_WORK_RADIAL_EXACT: the grid-only stencil rows and work
  //! factors, built once per grid (#291)
  X1GridKey _rx_key;
  torch::Tensor _rx_lo, _rx_mid, _rx_hi;
  CorrectedPEFactors _rx_work;

  // grid-only factors of forward_masked, rebuilt only
  // when a coordinate face tensor, the device, dtype, shape or grav1 changes
  struct GeoKey {
    c10::TensorImpl const *f1 = nullptr, *f2 = nullptr, *f3 = nullptr;
    int64_t v1 = -1, v2 = -1, v3 = -1;
    torch::Device dev = torch::kCPU;
    c10::ScalarType st = torch::kDouble;
    std::vector<int64_t> shape;
    double grav1 = 0.;
    bool operator==(GeoKey const& o) const {
      return f1 == o.f1 && f2 == o.f2 && f3 == o.f3 && v1 == o.v1 &&
             v2 == o.v2 && v3 == o.v3 && dev == o.dev && st == o.st &&
             shape == o.shape && grav1 == o.grav1;
    }
  };
  GeoKey _g_key;
  torch::Tensor _g_area, _g_volume, _g_work_lo, _g_work_hi, _g_mask0, _g_phi,
      _g_dp_lo, _g_dp_hi;
};
TORCH_MODULE(ImplicitHydro);

}  // namespace snap

#undef ADD_ARG
