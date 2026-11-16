// C/C++
#include <optional>
#include <sstream>

// torch
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_hydro(py::module& m) {
  auto pyHydroOptions =
      py::class_<snap::HydroOptionsImpl, snap::HydroOptions>(m, "HydroOptions");

  pyHydroOptions.def(py::init<>())
      .def_static("from_yaml", &snap::HydroOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const snap::HydroOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("HydroOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::HydroOptionsImpl, verbose)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x1)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x2)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x3)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, wb_wall_clamp)
      .ADD_OPTION(std::string, snap::HydroOptionsImpl, wb_wall_reference)
      .ADD_OPTION(int, snap::HydroOptionsImpl, conversion_order)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, flux_covariance)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, x1_mass_covariance)
      .ADD_OPTION(std::string, snap::HydroOptionsImpl, conserve_axial_momentum)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, conserve_axial_momentum_closure)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, perturbation_flux)
      .ADD_OPTION(std::string, snap::HydroOptionsImpl, precision)
      .ADD_OPTION(snap::ConstGravityOptions, snap::HydroOptionsImpl, grav)
      .ADD_OPTION(snap::CoriolisOptions, snap::HydroOptionsImpl, coriolis)
      .ADD_OPTION(snap::DiffusionOptions, snap::HydroOptionsImpl, diffusion)
      .ADD_OPTION(snap::EquationOfStateOptions, snap::HydroOptionsImpl, eos)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon1)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon23)
      .ADD_OPTION(snap::RiemannSolverOptions, snap::HydroOptionsImpl, riemann)
      .ADD_OPTION(snap::ImplicitOptions, snap::HydroOptionsImpl, icorr);

  ADD_SNAP_MODULE(Hydro, HydroOptions)
      .def(py::init<snap::HydroOptions, torch::nn::Module*>(),
           py::arg("options"), py::arg("block") = nullptr)
      .def("max_time_step", &snap::HydroImpl::max_time_step);

  m.def(
      "balance_column",
      [](torch::Tensor const& w, torch::Tensor const& dx1f, double grav,
         bool wall_clamp, double rtol, int max_iter,
         std::string const& geometry, bool stretched, int conversion_order,
         snap::EquationOfStateImpl* eos, std::optional<bool> upward,
         std::optional<std::string> reference, std::optional<bool> solid) {
        // the reference: upward= if given, else the deck's (reference=,
        // solid=) resolved as the solver does, else main's (legacy default)
        TORCH_CHECK(!(upward && reference),
                    "balance_column: pass upward= or reference=, not both");
        TORCH_CHECK(!reference || *reference != "auto" || solid,
                    "balance_column: reference='auto' needs solid= (whether "
                    "the deck has an immersed solid)");
        bool up = upward      ? *upward
                  : reference ? snap::upward_wall_reference(
                                    *reference, solid.value_or(false))
                              : false;
        return snap::balance_column(w, dx1f, grav, wall_clamp, rtol, max_iter,
                                    geometry, stretched, conversion_order, eos,
                                    up);
      },
      py::arg("w"), py::arg("dx1f"), py::arg("grav"),
      py::arg("wall_clamp") = true, py::arg("rtol") = 1.e-10,
      py::arg("max_iter") = 120, py::arg("geometry") = "",
      py::arg("stretched") = false, py::arg("conversion_order") = 2,
      py::arg("eos") = nullptr, py::arg("upward") = py::none(),
      py::arg("reference") = py::none(), py::arg("solid") = py::none(),
      R"(Project a ghost-free x1 column onto the well-balanced scheme's own
discrete hydrostatic balance, holding p/rho -- the temperature -- fixed per
cell. Returns (w, residual, sweeps); the residual is max|p'-C|/(rho*g*dz), the
acceleration the balanced column can still feel in units of g. `w` is
(nvar, nc3, nc2, nx1) with NO ghost cells: hand it the whole column and the p
and rho it returns do not depend on how that column is later split over blocks,
because the cell pressure reference is bitwise the same either way. Raises if it
does not converge. `wall_clamp` must be true and must match
`dynamics/wb-wall-clamp` in the card that will run the result, so a card that
turns the clamp off cannot use this primitive. See
src/hydro/balance_column.hpp for the caller's two obligations (physical x1
boundaries; >= 5 x1 cells per block). Under SNAP_X1_CENTROID_EXACT pass
geometry='cartesian'; any other geometry is refused. On a stretched x1 grid
key (#280) pass stretched=True (the coordinate's stretched_x1()), so the
column is classified as the solver classifies its blocks. conversion_order=4
balances the state the key-4 solver applies the reference to, W(u - L u) +
L W(u); it needs `eos` (block.module("hydro.eos")) and returns passes.
The x1 wall reference: upward=True/False selects it explicitly; otherwise
reference= (the deck's dynamics/wb-wall-reference: auto, upward or main) and
solid= (whether the deck has an immersed solid) resolve it as the solver does;
with neither it is main's. A solid deck must be balanced with reference='auto',
solid=True.)");
}
