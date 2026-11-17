#pragma once
// V3 lane S skeleton: declarations only, in no build. Revision 1.
// Pin chengcli/snapy 9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410. S owns the
// transport of every within-stage exchange (S.rst section 3 table); the
// rules for registered fluxes are R's (flux_register.hpp:79-80 at 543ff84f).
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
enum class RegisterPhase : uint8_t;  // R's (flux_register.hpp:43-51)

//! what a block face is; replaces three wall predicates (hydro.cpp:296-301,
//! is_physical_boundary, bc_func.cpp:231-250) with one lookup
enum class FaceKind : uint8_t {
  interior,  // same-panel neighbour block
  seam_x1,   // x1 process or thread seam (pz > 1)
  panel,     // cubed-sphere panel seam, rotated
  periodic,  // seam to self (layout wrap, or the bc copy on slab x1)
  wall,      // reflecting / solid
  outflow,   // radiating
  custom,    // user callback
};

//! corners: diagonal neighbour on slab and cubed layouts; synthesized
//! 0.5*(edge+edge) only at cubed-sphere panel corners (layout.cpp:681-721)
enum class CornerRule : uint8_t { none, diagonal, synthesized, x1_full };

enum class Frame : uint8_t { raw, rotated_cell, rotated_face };

//! one row of the declared stage schedule (S.rst section 3.3)
struct ExchangeRow {
  int id;                 // row number in S.rst
  std::string name;       // e.g. "reconcile.x1"
  RegisterPhase phase;    // R's phase where the row is a register phase
  Frame frame;
  CornerRule corners;
  bool interpolate;       // cubed-sphere ghost interpolation (delta form)
  int rounds;             // blocking rounds with every option on
};

//! the declared schedule of one stage; X reads it to place segment cuts
std::vector<ExchangeRow> const& stage_schedule();

//! one transport for same-process boards and the process group (today two
//! paths: layout.cpp:312-500, 895-940 and 556-653)
class Channel {
 public:
  virtual ~Channel() = default;
  virtual void post(int block, torch::Tensor const& t, int tag) = 0;
  virtual torch::Tensor take(int block, torch::Tensor const& like,
                             int tag) = 0;
  virtual void wait_all() = 0;
};

//! S's transport API. Replaces the hand-written relays in hydro
//! (hydro.cpp:344-377, 727-776; hydro_forward.cpp:882-946).
class Exchange {
 public:
  explicit Exchange(MeshBlockImpl const* pmb);
  FaceKind kind(int face) const;  // face 0..5 as bfuncs()
  //! ghost fill of cell fields (rows 1-3, 17, 26, 27)
  void ghosts(std::vector<torch::Tensor> fields, int type, bool interpolate);
  //! m x1 rows over every x2/x3 column, packed in one message (rows 4, 5,
  //! 8, 10, 13-16, 19, 22-25; merges M4, M5)
  void x1_rows(std::vector<torch::Tensor> fields, int m, bool faces, int tag);
  //! the peer's face slab for R's replace (rows 9, 16r): returns, per
  //! shared face, the peer's values in the receiver's basis
  torch::Tensor seam_slab(int dim, torch::Tensor const& slab, int tag);
  //! x1 column gather (row 28) and the reference anchor scan (rows 6-7)
  torch::Tensor gather_x1(torch::Tensor const& piece);
  torch::Tensor anchor_scan(torch::Tensor const& local_drop);  // M3, non-bitwise
};

}  // namespace snap
