#pragma once
// V3 lane I skeleton (declarations only; not part of any build).
// Design: docs/v3/design/I.rst. Pin 9e5de2d4. Replaces src/implicit/
// implicit_hydro.hpp.

#include <torch/nn/cloneable.h>
#include <yaml-cpp/yaml.h>

#include <snap/add_arg.h>

namespace snap {

class HydroImpl;
class GravityWork;      // gravity_work.hpp
class X1FaceRegister;   // lane R; the posting calls are an open question (I.rst s11)

//! The two VIC matrices; the value is the block size. Replaces the bit test
//! (scheme >> 3) & 1 of ImplicitOptionsImpl (implicit_hydro.hpp:37).
enum class VicMatrix : int { partial = 3, full = 5 };

//! Bits of the 0-d int32 device status of one block's solve. Written by the
//! kernels; read on the host only by check_redo, once per step (I.rst s8).
enum VicStatusBit : int32_t {
  kVicInputNonfinite = 1,   //!< a column's du, w or gamma is not finite
  kVicSingular = 2,         //!< ludcmp found a singular block
  kVicResultNonfinite = 4,  //!< a nonfinite solution or redistribution
  kVicFp64Retried = 8,      //!< the column predicate ran the FP64 stage
  kVicFp64Failed = 16       //!< the FP64 retry failed too: redo the step
};

struct VerticalImplicitOptionsImpl {
  //! integration/implicit-scheme: 1 -> partial, 9 -> full; 0 or absent ->
  //! nullptr (no module). Other keys as at the pin.
  static std::shared_ptr<VerticalImplicitOptionsImpl> from_yaml(
      YAML::Node const& integration);

  ADD_ARG(VicMatrix, matrix) = VicMatrix::full;
  ADD_ARG(std::string, precision) = "auto";  //!< auto | float32 | float64
  ADD_ARG(bool, fallback_fp64) = true;       //!< per-column, on device
  ADD_ARG(double, advection_cfl) = 1.0;
  ADD_ARG(double, shear_cfl) = 0.0;
};
using VerticalImplicitOptions = std::shared_ptr<VerticalImplicitOptionsImpl>;

//! Fluid segments of each column of a block, rebuilt only when the solid
//! mask's version changes (from #369's X1Wall depth). One segment per column
//! (the fluid above a solid floor, up to the top wall).
struct ColumnSegments {
  torch::Tensor lo;          //!< [nc3, nc2] int: first fluid cell
  torch::Tensor hi;          //!< [nc3, nc2] int: one past the last fluid cell
  torch::Tensor solid_floor; //!< [nc3, nc2] bool: lo is a solid surface, not
                             //!< the physical x1 wall
  int64_t mask_version = -1;

  //! no solid: lo = 0, hi = nx1 everywhere
  static ColumnSegments whole(int nx1, torch::TensorOptions const& opts);
  //! from the block's solid mask and its X1Wall depth (#369)
  static ColumnSegments from_solid(torch::Tensor const& solid,
                                   HydroImpl const& hydro);
};

//! What one solve posts (I.rst s3). Face transfers are x1 faces [is, ie],
//! area- and dt-integrated (kg per stage, or momentum per stage), in the
//! frame of the explicit x1 fluxes (after de-projection).
struct VicIncrement {
  torch::Tensor mass_dry;    //!< [nc3, nc2, nc1+1] clamped dry transfer
  torch::Tensor mass_species;//!< [ny, nc3, nc2, nc1+1] clamped transfers
  torch::Tensor mass_total;  //!< [nc3, nc2, nc1+1] dry + species: G reads it
  torch::Tensor mass_clamped;//!< [nc3, nc2, nc1+1] requested - moved (R7)
  torch::Tensor momentum_t;  //!< [2, nc3, nc2, nc1+1] partial matrix only
  torch::Tensor cell;        //!< [nvar, nc3, nc2, nc1] increments not in face
                             //!< form: x1 momentum, energy (gravity work
                             //!< excluded), full-matrix tangential momenta
};

class VerticalImplicitImpl
    : public torch::nn::Cloneable<VerticalImplicitImpl> {
 public:
  static std::shared_ptr<VerticalImplicitImpl> create(
      VerticalImplicitOptions const& opts, torch::nn::Module* p,
      std::string const& name = "icorr");

  VerticalImplicitOptions options;
  HydroImpl const* phydro = nullptr;  //!< non-owning parent
  VerticalImplicitImpl();
  explicit VerticalImplicitImpl(VerticalImplicitOptions const& options,
                                torch::nn::Module* p = nullptr);
  void reset() override;

  //! Solve one stage. du: the explicit stage increment, G[M_exp] already in
  //! its energy row (s4); dt: the stage-weighted operator dt
  //! (hydro_forward.cpp:1568-1583, unchanged). Posts the increment to reg,
  //! then books the gravity-work remainder G[M_imp] - R dq into du (s4).
  //! No host read: a failure only sets status() bits.
  void forward(torch::Tensor du, torch::Tensor w, torch::Tensor gamma,
               double dt, GravityWork const& gw, X1FaceRegister& reg,
               torch::Tensor const& solid = {});

  //! 0-d int32, VicStatusBit; check_redo reads it with the other redo marks
  torch::Tensor status() const;
  void reset_status();
  //! meters as at the pin (implicit_hydro.hpp:94-99), accumulation dtype
  torch::Tensor clamp_residual() const;
  torch::Tensor dry_clamp_step() const;
  torch::Tensor fp64_fallbacks() const;

  //! on a redo only (the host already synchronised): log the failed columns
  void report_failure(std::ostream& os) const;

 private:
  //! scratch a, b, c, delta, z0, z1, du0: unregistered, allocated once per
  //! (shape, dtype, device) so captured graphs see stable addresses
  void ensure_scratch(torch::Tensor const& w);

  VicIncrement _inc;
  ColumnSegments _seg;
  torch::Tensor _a, _b, _c, _delta, _z0, _z1, _du0, _status, _meters[3];
};
TORCH_MODULE(VerticalImplicit);

}  // namespace snap
