// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

// snap
#include "hydro_dispatch.hpp"
#include "hydro_ref_x1_impl.h"

namespace snap {

//! hydro_ref_x1_cpu on blocks with a solid: each column from its surface.
//! A unit of its own, so the no-solid kernel compiles (and inlines) as before.
void hydro_ref_x1_cpu_surface(
    torch::Tensor const& w, torch::Tensor const& dx1f,
    torch::Tensor const& anchor, torch::Tensor const& psf_lo,
    torch::Tensor const& psf_hi, torch::Tensor const& pref,
    torch::Tensor const& dsf, torch::Tensor const& dref, int iu, double grav,
    int x1_grid, bool phys_in, bool phys_out, bool wall_clamp,
    torch::Tensor const& rows, torch::Tensor const& surf) {
  auto sp = surf.data_ptr<int>();
  int ncolumns = w.size(1) * w.size(2);
  int nc1 = w.size(3);
  // A trailing pair dimension selects finite-increment reference arithmetic.
  if (w.dim() == 5) {
    TORCH_CHECK(w.scalar_type() == torch::kFloat64 && w.size(4) == 2,
                "split reference requires packed FP64 pairs");
    auto ptr = [](torch::Tensor const& t) -> SRS* {
      return t.defined() ? reinterpret_cast<SRS*>(t.data_ptr<double>())
                         : nullptr;
    };
    at::parallel_for(0, ncolumns, 0, [&](int64_t begin, int64_t end) {
      for (int column = begin; column < end; ++column)
        hydro_ref_x1_impl<SRS, true>(ptr(w), ptr(dx1f), ptr(anchor), ptr(psf_lo),
                          ptr(psf_hi), ptr(pref), ptr(dsf), ptr(dref), column,
                          ncolumns, nc1, iu, SRS(grav), x1_grid, phys_in,
                          phys_out, wall_clamp, ptr(rows), sp);
    });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "hydro_ref_x1_cpu", [&] {
    auto anchor_ptr = anchor.defined() ? anchor.data_ptr<scalar_t>() : nullptr;
    auto rows_ptr = rows.defined() ? rows.data_ptr<scalar_t>() : nullptr;
    at::parallel_for(0, ncolumns, 0, [&](int64_t begin, int64_t end) {
      for (int64_t column = begin; column < end; ++column) {
        hydro_ref_x1_impl<scalar_t, true>(w.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(),
                          anchor_ptr, psf_lo.data_ptr<scalar_t>(),
                          psf_hi.data_ptr<scalar_t>(),
                          pref.data_ptr<scalar_t>(), dsf.data_ptr<scalar_t>(),
                          dref.data_ptr<scalar_t>(), static_cast<int>(column),
                          ncolumns, nc1, iu, scalar_t(grav), x1_grid, phys_in,
                          phys_out, wall_clamp, rows_ptr, sp);
      }
    });
  });
}

}  // namespace snap
