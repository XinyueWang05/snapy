#pragma once

// C/C++
#include <cmath>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/math/ludcmp.h>
#include <snap/math/luminv.h>
#include <snap/snap.h>

#include "vic_solve_failure.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]

namespace snap {

//! Block-Thomas solve of one column whose two wall rows carry one extra
//! entry each, outside the block tridiagonal. The corrected-PE work's
//! one-sided wall slope couples the energy row (N-1) of the bottom wall cell
//! il to the mass (0) of cell il+2, entry e_lo = -wlo/dt, and the energy row
//! of the top wall cell iu to the mass of cell iu-2, entry e_hi = -whi/dt
//! (wlo, whi: g1 sigma^2 times the 3-point wall stencil's third-cell weight).
//! The border is carried through the ordinary forward and backward pass, with
//! no second factorization:
//!  * bottom: the LU of a_il also gives h = a_il^-1 e_(N-1), and
//!    x_il = Y_il - G_il x_il+1 - e_lo h x_il+2(0); row il+1 therefore sees
//!    c_il+1 - e_lo (b_il+1 h) e_0^T, and the back substitution adds
//!    -e_lo h x_il+2(0) at il;
//!  * top: x_iu-2 = Y_iu-2 - G_iu-2 x_iu-1 (both stored by the sweep), so row
//!    iu takes b_iu - e_hi e_(N-1) G_iu-2.row(0) and rhs - e_hi Y_iu-2(0).
//! It also books the explicit mass's third-cell work into DU(IPR), the
//! right-hand side of the same 3-point wall slope. A solid wall cell carries
//! no corrected-PE work, so its entry is dropped. Requires iu - il >= 3.
//! Returns false (column set to NaN) like ForwardSweep.
template <typename T, int N>
bool DISPATCH_MACRO BorderedSweep(Eigen::Matrix<T, N, N>* a,
                                  Eigen::Matrix<T, N, N>* b,
                                  Eigen::Matrix<T, N, N>* c,
                                  Eigen::Matrix<T, N, 1>* delta, T* du,
                                  T const* solid, double dt, int il, int iu,
                                  int dir, int ny, int stride1, int stride2,
                                  bool rhs_in_delta, double wlo, double whi) {
  if (solid[il * stride2] != 0) wlo = 0.;
  if (solid[iu * stride2] != 0) whi = 0.;
  auto mass = [&](int i) {
    T m = DU(IDN, i);
    for (int n = 0; n < ny; ++n) m += DU(ICY + n, i);
    return m;
  };
  DU(IPR, il) -= wlo * mass(il + 2);
  DU(IPR, iu) -= whi * mass(iu - 2);
  T const e_lo = -wlo / dt, e_hi = -whi / dt;

  Eigen::Matrix<T, N, 1> rhs, h;
  auto load = [&](int i) {
    rhs(0) = mass(i) / dt;
    rhs(1) = DU(IVX + dir, i) / dt;
    if constexpr (N == 3) {
      rhs(2) = DU(IPR, i) / dt;
    } else {
      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
      rhs(4) = DU(IPR, i) / dt;
    }
    if (rhs_in_delta)
      for (int row = 0; row < N; ++row)
        if (delta[i](row) != 0) rhs(row) -= delta[i](row);
  };
  auto finite = [&](int i) {
    for (int row = 0; row < N; ++row) {
      if (!std::isfinite(delta[i](row))) return false;
      for (int col = 0; col < N; ++col)
        if (!std::isfinite(a[i](row, col))) return false;
    }
    return true;
  };

  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A;

  // first row: the pivot's LU also gives h = a_il^-1 e_(N-1)
  load(il);
  if constexpr (N > 4) {
    Eigen::Matrix<T, N, N + 2, Eigen::RowMajor> solved;
    A = a[il];
    for (int n = 0; n < N; ++n) indx[n] = n;
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    solved.template leftCols<N>() = c[il];
    solved.col(N) = rhs;
    solved.col(N + 1).setZero();
    solved(N - 1, N + 1) = 1;
    lubksb(A, indx, solved);
    a[il] = solved.template leftCols<N>();
    delta[il] = solved.col(N);
    h = solved.col(N + 1);
  } else {
    A = a[il];
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    a[il] = a[il].inverse().eval();
    h = a[il].col(N - 1);
    delta[il] = a[il] * rhs;
    a[il] = a[il] * c[il];
  }
  if (!finite(il)) return vic_fail_column(delta, il, iu);
  for (int row = 0; row < N; ++row)
    if (!std::isfinite(h(row))) return vic_fail_column(delta, il, iu);
  // bottom border into row il+1's coupling to cell il+2 (mass column)
  c[il + 1].col(0) -= e_lo * (b[il + 1] * h);

  for (int i = il + 1; i <= iu; ++i) {
    load(i);
    if (i == iu) {  // top border, from the stored G, Y of iu-2
      b[iu].row(N - 1) -= e_hi * a[iu - 2].row(0);
      rhs(N - 1) -= e_hi * delta[iu - 2](0);
    }
    a[i] -= b[i] * a[i - 1];
    if constexpr (N > 4) {
      Eigen::Matrix<T, N, N + 1, Eigen::RowMajor> solved;
      A = a[i];
      for (int n = 0; n < N; ++n) indx[n] = n;
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      solved.template leftCols<N>() = c[i];
      solved.col(N) = rhs - b[i] * delta[i - 1];
      lubksb(A, indx, solved);
      a[i] = solved.template leftCols<N>();
      delta[i] = solved.col(N);
    } else {
      A = a[i];
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      a[i] = a[i].inverse().eval();
      delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
      a[i] = a[i] * c[i];
    }
    if (!finite(i)) return vic_fail_column(delta, il, iu);
  }

  // back substitution (as vic_backward_substitute) plus the bottom border
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];
  delta[il] -= (e_lo * delta[il + 2](0)) * h;
  return true;
}

}  // namespace snap

#undef DU
