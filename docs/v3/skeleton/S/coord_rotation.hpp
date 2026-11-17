#pragma once
// V3 lane S skeleton: declarations only, in no build. Revision 1.1.
// Pin chengcli/snapy 9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410; lines at the
// pin. Untested claims: "hypothesis". S-1 (settled): coord owns ONE rotation
// and the rotation data; lane A owns the Omega and mass-flux consumers.
#include <array>
#include <torch/torch.h>

namespace snap {  // R's namespace (flux_register.hpp:32 at 3490d7e1)
class CoordinateImpl;

//! declared rotation axis. Only z is supported; anything else is refused at
//! construction with a message (FOR CHENG item, kept in lane A). The axis is
//! hard-coded z today (gnomonic_equiangle.cpp:22-35).
struct RotationAxis {
  std::array<double, 3> unit = {0., 0., 1.};
  void check_supported() const;  // throws unless unit == z
};

//! the one rotation. Replaces the spherical frame matrices
//! (cubed_sphere_utils.cpp:146-193, 259-334; cubed_sphere_layout.cpp:132-191),
//! Coriolis' own frame (coriolis.cpp:128-142) and diffusion's calls
//! (diffusion_cubed_sphere.cpp:193, 371). The Cartesian route needs no atan2
//! (cubed_sphere_utils.cpp:189 is the H16 source).
class PanelRotation {
 public:
  //! panel contravariant <-> global Cartesian (cubed_sphere_utils.cpp:194-243)
  void contra_to_cart_(torch::Tensor vel, int panel, torch::Tensor alpha,
                       torch::Tensor beta) const;
  void cart_to_contra_(torch::Tensor vel, int panel, torch::Tensor alpha,
                       torch::Tensor beta) const;
  //! a covariant face vector of panel `from` into the covariant basis of
  //! panel `to` at the same seam face: lower o cart_to_contra(to) o
  //! contra_to_cart(from) o raise. Used by the seam transport (row 16r).
  void panel_to_panel_rows_(torch::Tensor rows, int from, int to, int dim,
                            torch::Tensor alpha, torch::Tensor beta) const;
  //! the face rotation the solver and the register use today
  //! (gnomonic_equiangle.cpp:365-484), unchanged in V3
  void prim2local_(torch::Tensor w, int dim) const;
  void flux2global_(torch::Tensor flux, int dim) const;
  //! face unit normal in the post frame, for boundary.prescribed (S.rst s4)
  torch::Tensor face_normal(int dim) const;
};

//! the rotation data coord owns (gnomonic_equiangle.hpp, AAM block): the
//! unit-radius rotation field k, its face averages, cos^2 lat and the
//! rate-to-rows projection (gnomonic_equiangle.cpp:485-603, 679-720)
struct AxialRotationData {
  RotationAxis axis;
  torch::Tensor k_cell, k_face2, k_face3;  // contravariant, unit radius
  torch::Tensor cos2_lat;
  //! rate -> rows (axial_rate_to_rows_), read by lane A
  void rate_to_rows_(torch::Tensor row2, torch::Tensor row3,
                     torch::Tensor rate) const;
};

//! per-panel derived geometry built once (owner G in the 919936da designs):
//! seam arcs from one function of the unordered endpoint pair (already at
//! the pin, gnomonic_equiangle.cpp:114-123)
struct PanelGeometry {
  torch::Tensor seam_arc2, seam_arc3;
  static torch::Tensor arc(torch::Tensor const& e, torch::Tensor const& a,
                           torch::Tensor const& b);
};

}  // namespace snap
