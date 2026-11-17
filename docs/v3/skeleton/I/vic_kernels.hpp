#pragma once
// V3 lane I skeleton (declarations only; not part of any build). I.rst s6, s8:
// one definition of the column kernels for CPU, CUDA and MPS (OnHost). Replaces
// implicit_dispatch.hpp, vic_assemble_{partial,full}_impl.h,
// vic_gravity_work_impl.h and the stub bodies of implicit_dispatch.cpp/.cu.

#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <configure.h>

namespace snap {

//! scalar arguments of every stage; no flag bits in dir (I.rst s6)
struct VicParams {
  double dt = 0.;
  double grav = 0.;   //!< grav1, the full gravity (implicit_hydro.cpp:326)
  int dir = 0;        //!< 0: x1
  int ny = 0;         //!< species rows
  int n = 5;          //!< block size, VicMatrix
  int k = 0;          //!< 2: Woodbury wall third points, 0: none
  bool fp64_retry = false;  //!< the predicated second launch (s8)
};

//! one column's operands, unpacked once from the iterator (13-15 operands
//! were unpacked separately in each of 9 CPU and 9 CUDA stubs at the pin)
template <typename T>
struct VicColumn {
  T *du, *mass, *w, *gamma, *area, *vol;
  T *face_lo, *face_hi;      //!< GravityWorkRows of this column
  void *a, *b, *c, *delta, *z0, *z1;
  int lo, hi;                //!< ColumnSegments
  bool solid_floor;
  int32_t* status;           //!< VicStatusBit, block-wide (atomic OR)
  int64_t stride1, stride2;
};

//! Cell i's blocks a, b, c: the 5x5 |A| and flux Jacobians once, the energy
//! row from GravityWorkRows, restricted to {rho, m1, E} for N == 3 with the
//! tangential-momentum fold (vic_assemble_partial_impl.h:159-199).
template <typename T, int N>
DISPATCH_MACRO void vic_assemble_cell(VicColumn<T> const& col, int i,
                                      VicParams const& p);

//! the K = 2 correction y -= Z (I + V^T Z)^{-1} V^T y, per column, after
//! the back-substitution (was host tensor code, implicit_hydro.cpp:446-472)
template <typename T, int N>
DISPATCH_MACRO void vic_woodbury_column(VicColumn<T> const& col, T k0,
                                        T k1);

//! input screen of one column; sets kVicInputNonfinite, returns false then
template <typename T>
DISPATCH_MACRO bool vic_column_finite(VicColumn<T> const& col,
                                      VicParams const& p);

enum class VicStage { assemble, solve, redistribute };
//! the CPU launcher: iter.for_each over columns (assembly: over cells)
struct CpuColumns {
  template <class F>
  static void run(at::TensorIterator& iter, F&& per_column);
};

//! the CUDA launcher: gpu_kernel, 64 threads for K = 2 (#358), 128 else.
//! Declared in vic_kernels.cuh, which only implicit_dispatch.cu includes.
struct CudaColumns;

//! one stage for one launcher: unpacks VicColumn, then calls the per-column
//! templates with N and K chosen from p. Instantiated in
//! implicit_dispatch.cpp (CpuColumns) and implicit_dispatch.cu (CudaColumns).
template <class Launch>
void vic_stage(at::TensorIterator& iter, VicParams const& p, VicStage s);

//! the dry donor sweep of the tracer transfer (meshblock.cpp:786-806); no
//! host read: an always-run sweep on CPU, device-predicated on CUDA
void donor_cap(at::Tensor& P, at::Tensor const& held, int is, int ie);

}  // namespace snap

namespace at::native {

using vic_stage_fn = void (*)(at::TensorIterator&, snap::VicParams const&,
                              snap::VicStage);
using donor_cap_fn = void (*)(at::Tensor&, at::Tensor const&, int, int);

//! one stub for the three stages (the pin had 9: assemble, solve and
//! Woodbury solve, redistribute, each partial and full, plus donor_cap)
DECLARE_DISPATCH(vic_stage_fn, vic_stage_stub);
DECLARE_DISPATCH(donor_cap_fn, donor_cap_stub);

}  // namespace at::native
