// C/C++
#include <cmath>
#include <vector>

// gtest
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro_dispatch.hpp>

// snapy#280 stage 1: the two reference changes a stretched x1 makes to the
// smoothed rho/p, pinned on ONE stretched column against values this file
// derives itself.  Nothing here is compared with a stored kernel output.
//
//   (1) THE FACE VALUE.  dsf = psf_lo * rf, and on a stretched x1 rf is the
//       cell-centred ratio carried to where the face ACTUALLY IS, not to the
//       midpoint of the two cell indices.  The face x1f[i] sits between the
//       centres x1v[i-1] and x1v[i] at the fraction
//           t = (x1f[i] - x1v[i-1]) / (x1v[i] - x1v[i-1]),
//       which is h[i-1] / (h[i-1] + h[i]) and is 1/2 only on a uniform grid.
//       The shipped uniform path used t = 1/2 everywhere.
//
//   (2) THE WALL CONTINUATION.  dref = pref * rs, and at a clamped wall the
//       two ghosts of the 5-point stencil are continued from the wall cell and
//       its neighbour.  On a stretched x1 the continuation is driven by the
//       REAL DISTANCE to the ghost centre, in units of the mean of the wall
//       cell and its neighbour, not by the integer ghost index.  Ghost m past
//       the wall mirrors the width of the owned cell m-1 from the wall, so
//       both distances are fixed by the width list alone.
//
// Both expectations are closed form here because the column is built so that
// rho/p is EXACTLY LINEAR IN THE CELL INDEX.  The 5-point binomial smoother
// 1-4-6-4-1 / 16 has weights that sum to one and first moment zero about its
// centre, so on such a profile it returns the centre value exactly, and the
// smoothing drops out of every expectation below without this file having to
// restate it.  rf and the wall ghosts are then the only things left to pin --
// which is precisely the pair of kernel changes under test.
//
// Each check also asserts that the value the SHIPPED UNIFORM RULE would give
// is outside the tolerance, so the test cannot pass if the stretched branch is
// removed.

namespace {

constexpr int kNc1 = 20;
constexpr int kIu = 16;          // il = nc1 - 1 - iu = 3, owned cells [3, 16]
constexpr int kIl = kNc1 - 1 - kIu;

// rho/p of cell j: exactly linear in the index, and falling, so the LOWER
// wall takes the continuation's linear branch and the UPPER wall its
// ln-linear one. Both branches are therefore exercised.
constexpr double kA = 1.0, kB = -0.03;
//! the profile at a real index, so a ghost at a fractional distance from the
//! wall has a value too
double rop_at(double j) { return kA + kB * j; }
double rop_of(int j) { return rop_at(j); }

constexpr double kP0 = 100.0;    // constant pressure; rho carries the profile
constexpr double kGrav = 1.0;

//! a stretched column: geometric widths at ratio q
std::vector<double> widths(double q, double h0 = 0.08) {
  std::vector<double> h(kNc1);
  h[0] = h0;
  for (int i = 1; i < kNc1; ++i) h[i] = h[i - 1] * q;
  return h;
}

//! cell centres from the widths, with x1f[0] = 0
std::vector<double> centres(std::vector<double> const& h) {
  std::vector<double> x1v(kNc1);
  double face = 0.;
  for (int i = 0; i < kNc1; ++i) {
    x1v[i] = face + 0.5 * h[i];
    face += h[i];
  }
  return x1v;
}

//! the lower face of cell i
double face_of(std::vector<double> const& h, int i) {
  double f = 0.;
  for (int k = 0; k < i; ++k) f += h[k];
  return f;
}

struct Out {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;
};

Out run(std::vector<double> const& h) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto w = torch::zeros({snap::IPR + 1, 1, 1, kNc1}, opts);
  for (int j = 0; j < kNc1; ++j) {
    w[snap::IPR][0][0][j] = kP0;
    w[snap::IDN][0][0][j] = kP0 * rop_of(j);
  }
  auto dx1f = torch::from_blob(const_cast<double*>(h.data()), {kNc1}, opts)
                  .clone();

  auto sizes = w.sizes().slice(1).vec();
  Out o{torch::empty(sizes, opts), torch::empty(sizes, opts),
        torch::empty(sizes, opts), torch::empty(sizes, opts),
        torch::empty(sizes, opts)};

  auto rows = snap::hydro_ref_x1_stretched_rows(dx1f, kIu, /*phys_in=*/true,
                                                /*phys_out=*/true,
                                                /*wall_clamp=*/true);
  at::native::call_hydro_ref_x1(torch::kCPU, w, dx1f, torch::Tensor(), o.psf_lo,
                                o.psf_hi, o.pref, o.dsf, o.dref, kIu, kGrav,
                                snap::kX1RefStretched, /*phys_in=*/true,
                                /*phys_out=*/true, /*wall_clamp=*/true, rows);
  return o;
}

double at1(torch::Tensor const& t, int i) {
  return t[0][0][i].item<double>();
}

//! ghost k past the wall cell jw, in units of the mean of the wall cell and
//! its neighbour. dir = -1 below a lower wall, +1 above an upper wall. Built
//! here from the mirrored widths directly: ghost m carries the width of the
//! owned cell m-1 from the wall.
double steps_to_ghost(std::vector<double> const& h, int jw, int dir, int k) {
  double d = 0.5 * h[jw];            // wall cell centre to the wall itself
  for (int m = 1; m <= k; ++m) {
    double hm = h[jw - dir * (m - 1)];  // the mirrored width of ghost m
    d += (m < k) ? hm : 0.5 * hm;       // whole cells, then half of ghost k
  }
  return d / (0.5 * (h[jw] + h[jw - dir]));
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE FACE VALUE is the profile carried to the face position.
// ---------------------------------------------------------------------------
TEST(HydroRefX1RfWall, face_rho_over_p_is_interpolated_to_the_face_position) {
  auto h = widths(1.2);
  auto x1v = centres(h);
  auto o = run(h);

  int tested = 0;
  // cells whose own 5-point stencil, and that of the cell below, lie wholly
  // inside the owned range: the smoother returns the centre value exactly
  // there and no wall ghost enters.
  for (int i = kIl + 3; i <= kIu - 2; ++i) {
    double xf = face_of(h, i);
    double t = (xf - x1v[i - 1]) / (x1v[i] - x1v[i - 1]);
    ASSERT_GT(t, 0.) << "i = " << i;
    ASSERT_LT(t, 1.) << "i = " << i;

    double want = rop_of(i - 1) + t * (rop_of(i) - rop_of(i - 1));
    double got = at1(o.dsf, i) / at1(o.psf_lo, i);
    EXPECT_NEAR(got, want, 1.e-12 * std::abs(want))
        << "face " << i << ": t = " << t;

    // the shipped uniform rule, t = 1/2, is a different number here, by more
    // than the tolerance: this check bites
    double midpoint = 0.5 * (rop_of(i - 1) + rop_of(i));
    EXPECT_GT(std::abs(want - midpoint), 1.e-6)
        << "face " << i << " does not separate the two rules";
    ++tested;
  }
  EXPECT_GE(tested, 8);

  // on a uniform column the face sits at the midpoint, so the stretched rule
  // and the shipped one agree
  auto hu = widths(1.0);
  auto xu = centres(hu);
  auto ou = run(hu);
  for (int i = kIl + 3; i <= kIu - 2; ++i) {
    double xf = face_of(hu, i);
    double t = (xf - xu[i - 1]) / (xu[i] - xu[i - 1]);
    EXPECT_NEAR(t, 0.5, 1.e-14) << "i = " << i;
    double want = 0.5 * (rop_of(i - 1) + rop_of(i));
    EXPECT_NEAR(at1(ou.dsf, i) / at1(ou.psf_lo, i), want,
                1.e-12 * std::abs(want))
        << "uniform face " << i;
  }
}

// ---------------------------------------------------------------------------
// 2. THE WALL CONTINUATION runs on the real distance to the ghost centre.
// ---------------------------------------------------------------------------
TEST(HydroRefX1RfWall, the_lower_wall_cell_is_continued_by_distance) {
  auto h = widths(1.2);
  auto o = run(h);

  // the profile falls with index, so at the LOWER wall the neighbour is below
  // the wall cell and the continuation takes its linear branch,
  //   ghost at kk steps  ->  r0 + kk (r0 - r1) = rop(il - kk)
  ASSERT_LT(rop_of(kIl + 1), rop_of(kIl));
  double kk1 = steps_to_ghost(h, kIl, -1, 1);
  double kk2 = steps_to_ghost(h, kIl, -1, 2);
  EXPECT_GT(kk1, 0.);
  EXPECT_GT(kk2, kk1);

  // the smoother over [il-2 .. il+2] with the two ghosts at fractional
  // indices il - kk2 and il - kk1
  double want = (rop_at(kIl - kk2) + 4. * rop_at(kIl - kk1) +
                 6. * rop_of(kIl) + 4. * rop_of(kIl + 1) + rop_of(kIl + 2)) /
                16.;
  double got = at1(o.dref, kIl) / at1(o.pref, kIl);
  EXPECT_NEAR(got, want, 1.e-12 * std::abs(want));

  // the shipped uniform rule puts the ghosts at the integer indices il-1 and
  // il-2, where the binomial reproduces the centre value exactly
  double shipped = rop_of(kIl);
  EXPECT_GT(std::abs(want - shipped), 1.e-6)
      << "the lower wall cell does not separate the two rules";
}

TEST(HydroRefX1RfWall, the_upper_wall_cell_is_continued_by_distance) {
  auto h = widths(1.2);
  auto o = run(h);

  // at the UPPER wall the neighbour is ABOVE the wall cell in rho/p, so the
  // continuation takes its ln-linear branch, r0 (r0/r1)^kk
  double r0 = rop_of(kIu), r1 = rop_of(kIu - 1);
  ASSERT_GT(r1, r0);
  double kk1 = steps_to_ghost(h, kIu, +1, 1);
  double kk2 = steps_to_ghost(h, kIu, +1, 2);
  EXPECT_GT(kk1, 0.);
  EXPECT_GT(kk2, kk1);

  auto ghost = [&](double kk) { return r0 * std::pow(r0 / r1, kk); };
  double want = (rop_of(kIu - 2) + 4. * rop_of(kIu - 1) + 6. * r0 +
                 4. * ghost(kk1) + ghost(kk2)) /
                16.;
  double got = at1(o.dref, kIu) / at1(o.pref, kIu);
  EXPECT_NEAR(got, want, 1.e-12 * std::abs(want));

  // the shipped uniform rule uses the integer ghost indices 1 and 2
  double shipped = (rop_of(kIu - 2) + 4. * rop_of(kIu - 1) + 6. * r0 +
                    4. * ghost(1.) + ghost(2.)) /
                   16.;
  EXPECT_GT(std::abs(want - shipped), 1.e-6)
      << "the upper wall cell does not separate the two rules";
}

// on a uniform column the distances are exactly the integer step counts, so
// the stretched continuation reduces to the shipped one
TEST(HydroRefX1RfWall, on_a_uniform_column_the_step_count_is_the_ghost_index) {
  auto h = widths(1.0);
  for (int k : {1, 2, 3}) {
    EXPECT_NEAR(steps_to_ghost(h, kIl, -1, k), k, 1.e-14) << "lower k = " << k;
    EXPECT_NEAR(steps_to_ghost(h, kIu, +1, k), k, 1.e-14) << "upper k = " << k;
  }
}
