#pragma once
// V3 lane S skeleton: declarations only, in no build. Revision 1.
// The two-sided register entry for x1 seams, x2/x3 block edges and panel
// seams (S.rst section 4), and the solid-face mask S supplies. Names are
// R's (flux_register.hpp at 543ff84f): FluxRegister, RegisterPhase,
// replace(dim, name, seam), add(dim, name, rows, increment).
#include <string>
#include <vector>
#include <torch/torch.h>

namespace snap {
class FluxRegister;
class Exchange;
class PanelRotation;

//! who holds the canonical value of a shared face
enum class SeamOwner : uint8_t {
  both_mean,    // x1 process seam: R's rule today (hydro_forward.cpp:937, 942)
  lower_rank,   // panel seam: the block with the smaller global rank
  self,         // intra-panel x2/x3 edge: both sides compute the same bits
};

//! the declared gate of one entry kind
struct SeamGate {
  bool bitwise;        // true: compared with ==
  double rel_tol;      // used only when !bitwise
  std::string reason;  // e.g. "two panels' covariant bases, one rotation"
};

struct SeamEntry {
  int dim;              // 1, 2, 3 (R's dim)
  SeamOwner owner;
  SeamGate gate;
  std::vector<int> rotated_rows;  // momentum rows; empty when none
};

//! the entries S declares (section 4 table)
std::vector<SeamEntry> const& seam_entries();

//! phase reconcile: S moves the slab (Exchange::seam_slab), rotates the
//! momentum rows with coord's rotation, and calls R's
//! FluxRegister::replace(dim, name, seam) for the total and kept parts
void reconcile_seams(FluxRegister& reg, Exchange& ex,
                     PanelRotation const& rot, std::string const& name);

//! solid faces: S's coord/bc supplies the mask; the register applies
//! the zero as one named boundary increment ("bc.solid_face") through
//! FluxRegister::add. Rows zeroed: mass, species, energy, scalars; the
//! normal momentum keeps the mirror-Riemann flux (internal_boundary.cpp:70-95).
//! Whether R or S makes the add() call is open (S.rst section 11).
struct SolidFaceMask {
  torch::Tensor face[3];  // bool, true where either neighbour is solid
  std::vector<int> zero_rows;
  static SolidFaceMask build(torch::Tensor const& solid, int nvar);
};

void post_solid_faces(FluxRegister& reg, SolidFaceMask const& mask);

}  // namespace snap
