#pragma once

#include <torch/torch.h>

#include <utility>

#include "hydro_dispatch.hpp"

namespace snap {

//! Outputs of the existing well-balanced reference kernel. The caller owns
//! seam exchange and the optional wb_ref4 correction, in that order.
struct HydroReferenceX1 {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;

  explicit HydroReferenceX1(torch::Tensor const& w) {
    auto shape = w.sizes().slice(1);
    psf_lo = torch::empty(shape, w.options());
    psf_hi = torch::empty(shape, w.options());
    pref = torch::empty(shape, w.options());
    dsf = torch::empty(shape, w.options());
    dref = torch::empty(shape, w.options());
  }
};

//! Shared grid setup and evaluation for runtime hydro and balance_column.
//! Explicit stretched grids always use quadrature. The legacy runtime
//! nonuniform branch is retained; initialization opts into stretched rows for
//! unequal widths. These are different operators, not accuracy settings.
class HydroReferenceX1Operator {
 public:
  HydroReferenceX1Operator(torch::Tensor const& dx1f, int iu, bool stretched,
                           bool phys_in, bool phys_out, bool wall_clamp,
                           X1RefGrid unequal = kX1RefNonUniform)
      : dx1f_(dx1f),
        version_(version(dx1f)),
        dtype_(dx1f.scalar_type()),
        device_(dx1f.device()),
        iu_(iu),
        stretched_(stretched),
        phys_in_(phys_in),
        phys_out_(phys_out),
        wall_clamp_(wall_clamp) {
    auto d = dx1f.cpu().to(torch::kFloat64);
    grid_ = stretched ? kX1RefStretched
                      : ((d.max() - d.min()).item<double>() <
                                 1.e-10 * d.mean().item<double>()
                             ? kX1RefUniform
                             : unequal);
    if (grid_ == kX1RefStretched) {
      rows_ =
          hydro_ref_x1_stretched_rows(dx1f, iu, phys_in, phys_out, wall_clamp);
    }
  }

  bool matches(torch::Tensor const& dx1f, int iu, bool stretched, bool phys_in,
               bool phys_out, bool wall_clamp) const {
    // Module::to preserves identity and version but changes dtype/device.
    return dx1f_.is_same(dx1f) && version_ == version(dx1f) &&
           dtype_ == dx1f.scalar_type() && device_ == dx1f.device() &&
           iu_ == iu && stretched_ == stretched && phys_in_ == phys_in &&
           phys_out_ == phys_out && wall_clamp_ == wall_clamp;
  }

  int grid() const { return grid_; }

  //! Linear pressure scan with an explicit upper face anchor at cell iu.
  //! rho is mixture density (or its increment), [nc3,nc2,nc1]; anchor has
  //! shape [nc3,nc2,1]. Signed values and zero are retained without floors.
  //! This does not linearize pref/dref, wall continuation, or nonlinear guards.
  //! Caller owns anchor relay and any background cache. Output dtype/device
  //! follow rho; geometry and anchor are converted before scanning. Returns
  //! {lower, upper} face pressure for each cell, including ghosts.
  std::pair<torch::Tensor, torch::Tensor> scan_pressure(
      torch::Tensor const& rho, double grav,
      torch::Tensor const& anchor) const {
    TORCH_CHECK(rho.dim() == 3 && rho.numel() > 0 && dx1f_.dim() == 1 &&
                    rho.size(-1) == dx1f_.numel() && iu_ >= 0 &&
                    iu_ < rho.size(-1),
                "hydro pressure scan: density and grid shapes disagree");
    TORCH_CHECK(rho.scalar_type() == torch::kFloat32 ||
                    rho.scalar_type() == torch::kFloat64,
                "hydro pressure scan: density must be float32 or float64");
    TORCH_CHECK(
        anchor.defined() && anchor.dim() == 3 &&
            anchor.size(0) == rho.size(0) && anchor.size(1) == rho.size(1) &&
            anchor.size(2) == 1,
        "hydro pressure scan: explicit anchor must have shape [nc3,nc2,1]");
    auto lo = torch::empty(rho.sizes(), rho.options());
    auto hi = torch::empty_like(lo);
    at::native::call_hydro_pressure_scan_x1(
        rho.device().type(), rho.contiguous(),
        dx1f_.to(rho.options()).contiguous(),
        anchor.to(rho.options()).contiguous(), lo, hi, iu_, grav);
    return {lo, hi};
  }

  void evaluate(torch::Tensor const& w, double grav,
                torch::Tensor const& anchor, HydroReferenceX1& out) const {
    TORCH_CHECK(w.dim() == 4 && w.size(-1) == dx1f_.numel(),
                "hydro reference: state and grid shapes disagree");
    auto dx = dx1f_.to(w.options()).contiguous();
    auto rows =
        rows_.defined() ? rows_.to(w.options()).contiguous() : torch::Tensor();
    auto a = anchor.defined() ? anchor.to(w.options()).contiguous()
                              : torch::Tensor();
    at::native::call_hydro_ref_x1(w.device().type(), w.contiguous(), dx, a,
                                  out.psf_lo, out.psf_hi, out.pref, out.dsf,
                                  out.dref, iu_, grav, grid_, phys_in_,
                                  phys_out_, wall_clamp_, rows);
  }

 private:
  static int64_t version(torch::Tensor const& t) {
    return t.is_inference() ? 0 : t._version();
  }
  torch::Tensor dx1f_, rows_;
  int64_t version_;
  c10::ScalarType dtype_;
  torch::Device device_;
  int iu_, grid_;
  bool stretched_, phys_in_, phys_out_, wall_clamp_;
};

}  // namespace snap
