#pragma once
// V3 lane S skeleton: declarations only, in no build. Revision 1.1.
// The two-sided register entry for every shared face (S.rst section 4) and
// the wall/solid faces S supplies for R's "boundary.prescribed". Names are
// R's at happysky19/snapy 3490d7e1 (flux_register.hpp): FluxRegister,
// RegisterPhase::reconcile, SeamRule {mean, owner_copy} (:46),
// replace(dim, faces, peer, rule) (:77-78), set_prescribed(dim, mask,
// value) (:61-62); R itself calls apply_prescribed(dim) (:79).
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace snap {
class FluxRegister;
class Exchange;
class PanelRotation;
enum class SeamRule : uint8_t;  // R's (flux_register.hpp:46)

//! who holds the canonical value of a shared face
enum class SeamOwner : uint8_t {
  both_mean,   // every x1 seam, process or same-process: SeamRule::mean
               // (pin: process seams only, hydro_forward.cpp:887-888)
  lower_rank,  // panel seam: SeamRule::owner_copy, momentum rotated (16r)
  self,        // intra-panel x2/x3 edge: same bits on both sides, no replace
};

//! the declared gate of one entry kind
struct SeamGate {
  bool bitwise;        // true: compared with ==
  double rel_tol;      // used only when !bitwise (panel momentum: 1e-14)
  std::string reason;  // e.g. "two panels' covariant bases, one rotation"
};

struct SeamEntry {
  int dim;              // 1 = x1, 2 = x2, 3 = x3 (R's numbering)
  SeamOwner owner;
  SeamGate gate;
  std::vector<int> rotated_rows;  // momentum rows; empty when none
};

//! the entries S declares (S.rst section 4 table)
std::vector<SeamEntry> const& seam_entries();

//! phase reconcile: per dim, S gathers the shared faces and the peer entry
//! (Exchange::seam_slab: board for a local peer, process group otherwise),
//! rotates the momentum rows of a panel peer with coord's rotation, and
//! calls R's FluxRegister::replace(dim, faces, peer, rule)
void reconcile_seams(FluxRegister& reg, Exchange& ex,
                     PanelRotation const& rot);

//! wall and immersed-solid faces for R's "boundary.prescribed". A face is
//! prescribed if it is a reflecting wall or either neighbour cell is solid
//! (internal_boundary.cpp:74-76). Rows set: mass, species, energy, scalars,
//! value 0. Momentum rows are not set: they keep the mirror-Riemann flux
//! (free-slip, internal_boundary.cpp:78-80, 84-91). normal is coord's face
//! unit normal in the post frame, supplied for EF's tangential option only.
struct PrescribedFaces {
  torch::Tensor mask[3];    // bool [nvar, faces...]: row prescribed on face
  torch::Tensor value[3];   // [nvar, faces...]: 0 on every set row
  torch::Tensor normal[3];  // [3, faces...], unused by the default rule
  torch::Tensor solid_cells;  // EF: column breaks; out of theta and census
  static PrescribedFaces build(torch::Tensor const& solid, int nvar,
                               std::vector<int> const& zero_rows);
};

//! S's only call into R for boundaries: set_prescribed(dim, mask, value)
//! per active dim, redone when the solid field or wall set changes
void supply_prescribed(FluxRegister& reg, PrescribedFaces const& faces);

}  // namespace snap
