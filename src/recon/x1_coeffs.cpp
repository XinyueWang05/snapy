// C/C++
#include <cmath>

// snap
#include "x1_coeffs.hpp"

namespace snap {
namespace {

//! Shu (1998) eq. 2.20 primitive-function weights at the target point 0.
/*!
 * For the `n` cells bounded by the `n+1` nodes z[0..n],
 *   w_k = (z_{k+1} - z_k) * sum_{j>k} L_j'(0),
 * with L_j the Lagrange basis on those nodes.  sum_k w_k == 1 identically.
 *
 * \param[in]  z  n+1 nodes, in the normalised coordinate of the caller
 * \param[in]  n  number of cells, 3 (one WENO5 candidate) or 5 (cp5)
 * \param[out] w  n weights
 */
void primitive_row(double const* z, int n, double* w) {
  double lp[6];
  for (int j = 0; j <= n; ++j) {
    double s = 0.;
    for (int i = 0; i <= n; ++i) {
      if (i == j) continue;
      double term = 1. / (z[j] - z[i]);
      for (int l = 0; l <= n; ++l) {
        if (l == j || l == i) continue;
        term *= (0. - z[l]) / (z[j] - z[l]);
      }
      s += term;
    }
    lp[j] = s;
  }

  for (int k = 0; k < n; ++k) {
    double acc = 0.;
    for (int j = k + 1; j <= n; ++j) acc += lp[j];
    w[k] = (z[k + 1] - z[k]) * acc;
  }
}

//! Rows 1 and 2 of V^{-1} for the quadratic q = a0 + a1 z + a2 z^2 whose cell
//! averages over the three cells bounded by z[0..3] are the given ones.
/*!
 * V[k] = {1, (p+q)/2, (p^2+pq+q^2)/3} for the cell [p, q] = [z[k], z[k+1]].
 * \param[out] r1  a1 = r1 . ubar
 * \param[out] r2  a2 = r2 . ubar
 */
void quad_inv_rows(double const* z, double* r1, double* r2) {
  double V[3][3];
  for (int k = 0; k < 3; ++k) {
    double p = z[k], q = z[k + 1];
    V[k][0] = 1.;
    V[k][1] = .5 * (p + q);
    V[k][2] = (p * p + p * q + q * q) / 3.;
  }

  // cofactor inverse; the matrix is 3x3 and well conditioned in the
  // normalised coordinate (|z| <= 5 for every admissible window)
  double cof[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      cof[i][j] = V[i1][j1] * V[i2][j2] - V[i1][j2] * V[i2][j1];
    }
  }
  double det = V[0][0] * cof[0][0] + V[0][1] * cof[0][1] + V[0][2] * cof[0][2];
  TORCH_CHECK(det != 0., "[x1_coeffs] degenerate WENO5 stencil");

  // inv[i][j] = cof[j][i] / det
  for (int j = 0; j < 3; ++j) {
    r1[j] = cof[j][1] / det;
    r2[j] = cof[j][2] / det;
  }
}

//! One 48-wide WENO5 row in the "p-type" orientation.
/*!
 * `y` are the six faces y0 < ... < y5 of the 5-cell window.  The target face
 * is y3, the UPPER face of the central cell I2 = [y2, y3].  Everything is
 * evaluated in zeta = (x - y3)/h2, in which the target sits at zeta = 0 and
 * the central cell has unit width, so the rows are scale free.
 */
void ptype_row(double const* y, double* row) {
  double h2 = y[3] - y[2];
  double z[6];
  for (int k = 0; k < 6; ++k) z[k] = (y[k] - y[3]) / h2;

  // stencil r uses the three window cells starting at k0[r]; the order is the
  // one the shipped rows use, largest linear weight first
  const int k0[3] = {2, 1, 0};
  // sign of the second smoothness row per stencil.  beta is invariant under
  // it; this is the sign convention of the shipped literals, so that a uniform
  // window reproduces src/recon/weno5.cpp:11-23 entry for entry.
  const double bsign[3] = {1., 1., -1.};

  for (int i = 0; i < 45; ++i) row[i] = 0.;

  for (int r = 0; r < 3; ++r) {
    double const* nodes = z + k0[r];

    double w[3];
    primitive_row(nodes, 3, w);

    double r1[3], r2[3];
    quad_inv_rows(nodes, r1, r2);

    for (int n = 0; n < 3; ++n) {
      int col = k0[r] + n;
      row[r * 5 + col] = w[n];
      row[(3 + 2 * r) * 5 + col] = 2. * r2[n];
      row[(4 + 2 * r) * 5 + col] = bsign[r] * (2. * r2[n] - 2. * r1[n]);
    }
  }

  // closed-form optimal weights (note Sec. 4.2); every factor is a positive
  // length, so d_r > 0 on any admissible grid
  double a = -z[0], b = -z[1], c = z[4], e = z[5];
  row[45] = a * b / ((a + e) * (b + e));
  row[46] = a * e * (a + b + c + e) / ((a + e) * (b + e) * (a + c));
  row[47] = c * e / ((a + c) * (a + e));
}

//! The two WENO5 rows of one window: cm targets y2, cp targets y3.
/*!
 * cm is the mirror of the p-type build: reflecting the window maps the target
 * y2 onto the p-type target, and flipping the cell order of each 5-entry row
 * brings it back.  Built this way the two sides are each other's mirror by
 * construction, which is the uniform-grid identity cp = cm.flip.
 */
void window_weno5(double const* y, double* cm, double* cp) {
  ptype_row(y, cp);

  double ym[6];
  for (int k = 0; k < 6; ++k) ym[k] = -y[5 - k];

  double tmp[kWeno5NuCoeff];
  ptype_row(ym, tmp);
  for (int r = 0; r < 9; ++r) {
    for (int n = 0; n < 5; ++n) cm[r * 5 + n] = tmp[r * 5 + (4 - n)];
  }
  cm[45] = tmp[45];
  cm[46] = tmp[46];
  cm[47] = tmp[47];
}

//! Combine one WENO5 row's candidates with its linear weights.
/*!
 * Note Sec. 4.2: "cp5 (`pinterp2`) gets the 5-entry row sum d_r p_r per face
 * and side."  That identity is also what makes WENO5 fifth order, so the cp5
 * row is BUILT this way rather than from an independent 6-node primitive
 * function: the two agree to round-off (checked in tests/test_x1_coeffs.cpp)
 * and this form loses fewer digits on a sharp width jump, where the 6-node
 * form cancels harder.
 */
void combine_optimal(double const* row48, double* out5) {
  for (int n = 0; n < 5; ++n) {
    out5[n] = row48[45] * row48[n] + row48[46] * row48[5 + n] +
              row48[47] * row48[10 + n];
  }
}

//! The two cp5 rows of one window: the degree-4 reconstruction at y2 and y3.
void window_cp5(double const* y, double* cm, double* cp) {
  double rm[kWeno5NuCoeff], rp[kWeno5NuCoeff];
  window_weno5(y, rm, rp);
  combine_optimal(rm, cm);
  combine_optimal(rp, cp);
}

//! Shared input validation; returns a contiguous CPU float64 copy of the faces.
torch::Tensor check_faces(torch::Tensor const& x1f) {
  TORCH_CHECK(x1f.defined(), "[x1_coeffs] face tensor is undefined");
  TORCH_CHECK(x1f.dim() == 1, "[x1_coeffs] expected a 1-D face vector, got ",
              x1f.dim(), " dimensions");
  TORCH_CHECK(x1f.numel() >= 6,
              "[x1_coeffs] a WENO5/cp5 window needs at least 6 faces, got ",
              x1f.numel());

  auto f = x1f.to(torch::kCPU, torch::kFloat64).contiguous();
  TORCH_CHECK(torch::isfinite(f).all().item<bool>(),
              "[x1_coeffs] face vector has a non-finite entry");

  auto d = f.slice(0, 1) - f.slice(0, 0, f.numel() - 1);
  TORCH_CHECK(d.min().item<double>() > 0.,
              "[x1_coeffs] face vector must be strictly increasing");
  return f;
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f) {
  auto f = check_faces(x1f);
  int64_t nout = f.numel() - 5;

  auto cm = torch::empty({nout, kWeno5NuCoeff}, torch::kFloat64);
  auto cp = torch::empty({nout, kWeno5NuCoeff}, torch::kFloat64);

  auto const* y = f.data_ptr<double>();
  auto* pm = cm.data_ptr<double>();
  auto* pp = cp.data_ptr<double>();
  for (int64_t m = 0; m < nout; ++m) {
    window_weno5(y + m, pm + m * kWeno5NuCoeff, pp + m * kWeno5NuCoeff);
  }
  return {cm, cp};
}

std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(
    torch::Tensor const& x1f) {
  auto f = check_faces(x1f);
  int64_t nout = f.numel() - 5;

  auto cm = torch::empty({nout, kPoly5NuCoeff}, torch::kFloat64);
  auto cp = torch::empty({nout, kPoly5NuCoeff}, torch::kFloat64);

  auto const* y = f.data_ptr<double>();
  auto* pm = cm.data_ptr<double>();
  auto* pp = cp.data_ptr<double>();
  for (int64_t m = 0; m < nout; ++m) {
    window_cp5(y + m, pm + m * kPoly5NuCoeff, pp + m * kPoly5NuCoeff);
  }
  return {cm, cp};
}

torch::Tensor const& x1_coeffs_for(torch::Tensor const& src,
                                   torch::Tensor& cache,
                                   torch::Tensor const& w) {
  if (src.scalar_type() == w.scalar_type() && src.device() == w.device()) {
    return src;
  }
  if (!cache.defined() || cache.scalar_type() != w.scalar_type() ||
      cache.device() != w.device() || cache.sizes() != src.sizes()) {
    cache = src.to(w.device(), w.scalar_type()).contiguous();
  }
  return cache;
}

torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim) {
  TORCH_CHECK(c.dim() == 2 && c.is_contiguous(),
              "[x1_coeffs] row view needs a contiguous 2-D table");
  TORCH_CHECK(dim >= 0 && dim < out.dim(), "[x1_coeffs] bad dim ", dim);
  TORCH_CHECK(c.size(0) == out.size(dim), "[x1_coeffs] table has ", c.size(0),
              " rows but the output has ", out.size(dim), " faces along dim ",
              dim);

  auto sizes = out.sizes().vec();
  std::vector<int64_t> strides(sizes.size(), 0);
  strides[dim] = c.size(1);
  return c.as_strided(sizes, strides, c.storage_offset());
}

}  // namespace snap
