// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <cmath>
#include <limits>
#include <vector>

// snap
#include <snap/implicit/flux_decomposition_impl.h>
#include <snap/implicit/vic_redistribute_impl.h>

#include <snap/implicit/implicit_hydro.hpp>

namespace {

template <typename T>
double column_integral(std::vector<T> const& u, std::vector<T> const& vol,
                       int var, int stride1, int il, int iu) {
  double sum = 0;
  for (int i = il; i <= iu; ++i) sum += vol[i] * u[var * stride1 + i];
  return sum;
}

}  // namespace

TEST(vic_flux_decomposition, species_relabeling_preserves_mixture_density) {
  constexpr int ncell = 2;
  constexpr int stride1 = ncell;
  constexpr int stride2 = 1;
  constexpr int interface = 1;

  std::vector<double> one_species((snap::ICY + 1) * ncell, 0.);
  std::vector<double> two_species((snap::ICY + 2) * ncell, 0.);

  for (int i = 0; i < ncell; ++i) {
    double density = 2. + i;
    double velocity = 10. + i;
    double pressure = 1.e5 + 100. * i;

    one_species[snap::IDN * stride1 + i] = density;
    two_species[snap::IDN * stride1 + i] = density;
    for (int n = snap::IVX; n <= snap::IVZ; ++n) {
      one_species[n * stride1 + i] = velocity + n;
      two_species[n * stride1 + i] = velocity + n;
    }
    one_species[snap::IPR * stride1 + i] = pressure;
    two_species[snap::IPR * stride1 + i] = pressure;

    one_species[snap::ICY * stride1 + i] = 0.4;
    two_species[snap::ICY * stride1 + i] = 0.1;
    two_species[(snap::ICY + 1) * stride1 + i] = 0.3;
  }

  double one_left[5], one_right[5], two_left[5], two_right[5];
  snap::CopyPrimitives(one_left, one_right, one_species.data(), interface,
                       stride1, stride2, 1);
  snap::CopyPrimitives(two_left, two_right, two_species.data(), interface,
                       stride1, stride2, 2);

  for (int n = snap::IDN; n <= snap::IPR; ++n) {
    EXPECT_DOUBLE_EQ(one_left[n], two_left[n]);
    EXPECT_DOUBLE_EQ(one_right[n], two_right[n]);
  }
  EXPECT_DOUBLE_EQ(one_left[snap::IDN], 2.);
  EXPECT_DOUBLE_EQ(one_right[snap::IDN], 3.);
}

TEST(vic_redistribution, conserves_constituent_column_tendencies) {
  constexpr int nlayer = 4;
  constexpr int ny = 3;
  constexpr int nhydro = snap::ICY + ny;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;
  constexpr int il = 0;
  constexpr int iu = nlayer - 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol = {1.0, 2.0, 1.5, 0.5};
  std::vector<double> corr = {0.4, -0.2, 0.1, -0.3};
  std::vector<double> rho = {3.0, 2.5, 4.0, 1.8};

  std::vector<Eigen::Matrix<double, 3, 3>> a(nlayer);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);
  std::vector<double> mass_fix(nhydro * nlayer, 0.);
  for (int i = il; i <= iu; ++i) {
    double vapor_frac = 0.10 + 0.02 * i;
    double cloud_frac = 0.05 + 0.01 * i;
    double tracer_frac = 0.02 + 0.005 * i;
    double dry_frac = 1. - vapor_frac - cloud_frac - tracer_frac;
    double explicit_total = 1.5 + 0.2 * i;

    du[snap::IDN * stride1 + i] = explicit_total * dry_frac;
    du[snap::IVX * stride1 + i] = 10.0 + i;
    du[snap::IPR * stride1 + i] = 20.0 + i;
    du[snap::ICY * stride1 + i] = explicit_total * vapor_frac;
    du[(snap::ICY + 1) * stride1 + i] = explicit_total * cloud_frac;
    du[(snap::ICY + 2) * stride1 + i] = explicit_total * tracer_frac;

    w[snap::ICY * stride1 + i] = vapor_frac;
    w[(snap::ICY + 1) * stride1 + i] = cloud_frac;
    w[(snap::ICY + 2) * stride1 + i] = tracer_frac;
    w[snap::IDN * stride1 + i] = rho[i];

    a[i].setZero();
    delta[i] << explicit_total + corr[i], 100.0 + i, 200.0 + i;
  }

  auto original = du;
  snap::vic_backward_substitute<double, 3>(a.data(), delta.data(), il, iu);
  snap::vic_constituent_column<double, 3>(du.data(), w.data(), mass_fix.data(),
                                          delta.data(), vol.data(), nlayer, 0,
                                          ny, stride1, stride2);
  for (int i = il; i <= iu; ++i) {
    snap::vic_redistribute_cell<double, 3>(
        du.data(), mass_fix.data(), delta.data(), i, 0, ny, stride1, stride2);
  }

  int constituent_vars[] = {snap::IDN, snap::ICY, snap::ICY + 1, snap::ICY + 2};
  for (int var : constituent_vars) {
    EXPECT_NEAR(column_integral(du, vol, var, stride1, il, iu),
                column_integral(original, vol, var, stride1, il, iu), 1.e-12);
  }

  std::vector<double> expected_face_mass = {0., -0.4, 0., -0.15};
  for (int i = il; i <= iu; ++i) {
    EXPECT_NEAR(mass_fix[snap::IVX * stride1 + i], expected_face_mass[i],
                1.e-12);

    double constituent_total = du[snap::IDN * stride1 + i];
    for (int n = 0; n < ny; ++n) {
      constituent_total += du[(snap::ICY + n) * stride1 + i];
      double final_mass = (rho[i] * w[(snap::ICY + n) * stride1 + i] +
                           du[(snap::ICY + n) * stride1 + i]) *
                          vol[i];
      EXPECT_GE(final_mass, -1.e-12);
    }
    EXPECT_NEAR(constituent_total, delta[i](0), 1.e-12);
  }

  // face rows: dry moved mass for the tracers, total moved mass for gravity
  double dry = 0., total = 0.;
  for (int i = il + 1; i <= iu; ++i) {
    dry -= vol[i - 1] * mass_fix[snap::IDN * stride1 + i - 1];
    total -= vol[i - 1] * mass_fix[snap::IDN * stride1 + i - 1];
    for (int n = 0; n < ny; ++n) {
      total -= vol[i - 1] * mass_fix[(snap::ICY + n) * stride1 + i - 1];
    }
    EXPECT_NEAR(mass_fix[snap::IVY * stride1 + i], dry, 1.e-12);
    EXPECT_NEAR(mass_fix[snap::IVZ * stride1 + i], total, 1.e-12);
  }
  EXPECT_GT(std::abs(mass_fix[snap::IVZ * stride1 + 1] -
                     mass_fix[snap::IVY * stride1 + 1]),
            1.e-3);
}

TEST(vic_redistribution, dry_only_transport_is_conservative_and_clamped) {
  constexpr int nlayer = 2;
  constexpr int ny = 0;
  constexpr int nhydro = snap::ICY;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol(nlayer, 1.);
  std::vector<double> mass_fix(nhydro * nlayer, 0.);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);

  w[snap::IDN * stride1] = 1.;
  w[snap::IDN * stride1 + 1] = 1.;
  du[snap::IDN * stride1] = -0.75;
  delta[0] << -2.75, 0., 0.;
  delta[1] << 2., 0., 0.;

  auto original = du;
  snap::vic_constituent_column<double, 3>(du.data(), w.data(), mass_fix.data(),
                                          delta.data(), vol.data(), nlayer, 0,
                                          ny, stride1, stride2);
  for (int i = 0; i < nlayer; ++i) {
    snap::vic_redistribute_cell<double, 3>(
        du.data(), mass_fix.data(), delta.data(), i, 0, ny, stride1, stride2);
  }

  EXPECT_NEAR(mass_fix[snap::IVX * stride1 + 1], 2., 1.e-12);
  EXPECT_NEAR(mass_fix[snap::IDN * stride1], -0.25, 1.e-12);
  EXPECT_NEAR(mass_fix[snap::IDN * stride1 + 1], 0.25, 1.e-12);
  EXPECT_EQ(mass_fix[snap::IPR * stride1], 1.);
  EXPECT_NEAR(mass_fix[snap::IVZ * stride1 + 1], 0.25, 1.e-12);
  EXPECT_EQ(mass_fix[snap::IPR * stride1 + 1], 0.);
  EXPECT_NEAR(column_integral(du, vol, snap::IDN, stride1, 0, 1),
              column_integral(original, vol, snap::IDN, stride1, 0, 1), 1.e-12);
  EXPECT_NEAR(w[snap::IDN * stride1] + du[snap::IDN * stride1], 0., 1.e-12);
}

// The pass-3a clamp acts on the DRY transfer, the marks were gated on the sign
// of the TOTAL-mass transfer; a donor mass fraction above one splits the two.
TEST(vic_redistribution,
     a_negative_dry_fraction_still_marks_the_clamped_donor) {
  constexpr int nlayer = 2;
  constexpr int ny = 1;
  constexpr int nhydro = snap::ICY + ny;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol(nlayer, 1.);
  std::vector<double> mass_fix(nhydro * nlayer, 0.);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);

  w[snap::IDN * stride1 + 0] = 1.;
  w[snap::IDN * stride1 + 1] = 1.;
  w[snap::ICY * stride1 + 0] = 1.5;  // donor dry fraction = -0.5
  w[snap::ICY * stride1 + 1] = 0.9;  // dry availability above it = 0.1
  delta[0] << -2., 0., 0.;
  delta[1] << 2., 0., 0.;

  snap::vic_constituent_column<double, 3>(du.data(), w.data(), mass_fix.data(),
                                          delta.data(), vol.data(), nlayer, 0,
                                          ny, stride1, stride2);

  // the fixture must REACH the branch: face transfer up, dry transfer down
  ASSERT_NEAR(mass_fix[snap::IVX * stride1 + 1], 2., 1.e-12);
  // bound by the upper cell's availability, not by Mf * dryfrac = -1
  EXPECT_NEAR(mass_fix[snap::IVY * stride1 + 1], -0.1, 1.e-12);
  EXPECT_EQ(mass_fix[snap::IPR * stride1 + 1], 1.);
  EXPECT_EQ(mass_fix[snap::IPR * stride1 + 0], 0.);
}

// A donor's species availability is rebuilt from the primitives, rho * y, which
// can sit an ulp above the conserved density u. A pass-3b clamp that drains
// rho * y to exactly zero then leaves u + du = -1 ulp, and the column repair
// re-mixes that cell with the one below: 1.8e-13 moved against a 9.4e-38
// deficit on examples/uranus.yaml (#260). The donor keeps 4096 ulp instead,
// in each precision the kernel dispatches to (float32 and float64).
namespace {

template <typename T>
void drain_a_species_donor() {
  constexpr int nlayer = 2;
  constexpr int ny = 1;
  constexpr int nhydro = snap::ICY + ny;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;
  constexpr T eps = std::numeric_limits<T>::epsilon();

  std::vector<T> du(nhydro * nlayer, 0.);
  std::vector<T> w(nhydro * nlayer, 0.);
  std::vector<T> vol(nlayer, 1.);
  std::vector<T> mass_fix(nhydro * nlayer, 0.);
  std::vector<Eigen::Matrix<T, 3, 1>> delta(nlayer);

  // the top cell of the uranus column: a trace of precipitation
  T rho = 1.19606e-2;
  w[snap::IDN * stride1 + 0] = 1.39720e-2;
  w[snap::IDN * stride1 + 1] = rho;
  w[snap::ICY * stride1 + 0] = 2.9e-11;
  w[snap::ICY * stride1 + 1] = T(7.52353515609675681e-22) / rho;
  T rhoy = w[snap::IDN * stride1 + 1] * w[snap::ICY * stride1 + 1];
  T u = std::nextafter(rhoy, T(0));  // conserved density, one ulp below

  // the explicit update leaves 2^-10 of it; a + d is exact (Sterbenz)
  T d = -rhoy * (1 - std::ldexp(T(1), -10));
  du[snap::ICY * stride1 + 1] = d;

  // move 1e-3 of total mass down through the face, far more than is left
  T c = 1.e-3;
  delta[0] << c, 0., 0.;
  delta[1] << d - c, 0., 0.;

  auto original = du;
  snap::vic_constituent_column<T, 3>(du.data(), w.data(), mass_fix.data(),
                                     delta.data(), vol.data(), nlayer, 0, ny,
                                     stride1, stride2);
  for (int i = 0; i < nlayer; ++i) {
    snap::vic_redistribute_cell<T, 3>(du.data(), mass_fix.data(), delta.data(),
                                      i, 0, ny, stride1, stride2);
  }

  // the fixture must REACH the clamp: Mf * y drains more than is available
  ASSERT_LT(mass_fix[snap::IVX * stride1 + 1] * w[snap::ICY * stride1 + 1],
            -(rhoy + d));

  T kept = u + du[snap::ICY * stride1 + 1];
  EXPECT_GE(kept, 0.) << "u = " << u << ", du = " << du[snap::ICY * stride1 + 1]
                      << ": the donor was drained below zero";
  // 4096 ulp of rho * y, less the ulp that u sits below it
  EXPECT_GE(kept, 4000 * eps * rhoy)
      << "kept " << kept / (eps * rhoy) << " ulp of rho * y";
  EXPECT_NEAR(column_integral(du, vol, snap::ICY, stride1, 0, 1),
              column_integral(original, vol, snap::ICY, stride1, 0, 1),
              4 * eps * rhoy);
}

}  // namespace

TEST(vic_redistribution, a_drained_species_donor_stays_non_negative) {
  drain_a_species_donor<double>();
}

TEST(vic_redistribution, a_drained_float_species_donor_stays_non_negative) {
  drain_a_species_donor<float>();
}

// Partial matrix: the face mass transfers carry the donor cell's tangential
// momentum, so a uniform tangential wind stays uniform and the column sum of
// each tangential momentum is unchanged; a wind-free column is untouched.
TEST(vic_redistribution, tangential_momentum_follows_the_face_mass) {
  constexpr int nlayer = 4;
  constexpr int ny = 0;
  constexpr int nhydro = snap::ICY;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;
  double const v2 = 400., v3 = -30.;

  for (bool wind : {true, false}) {
    std::vector<double> du(nhydro * nlayer, 0.);
    std::vector<double> w(nhydro * nlayer, 0.);
    std::vector<double> vol = {1.0, 2.0, 1.5, 0.5};
    std::vector<double> rho = {3.0, 2.5, 4.0, 1.8};
    std::vector<double> corr = {0.4, -0.2, 0.1, -0.3};
    std::vector<double> mass_fix(nhydro * nlayer, 0.);
    std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);
    for (int i = 0; i < nlayer; ++i) {
      w[snap::IDN * stride1 + i] = rho[i];
      w[snap::IVY * stride1 + i] = wind ? v2 : 0.;
      w[snap::IVZ * stride1 + i] = wind ? v3 : 0.;
      du[snap::IVY * stride1 + i] = -0.;  // a signed zero must survive
      delta[i] << corr[i], 0., 0.;
    }

    auto original = du;
    snap::vic_constituent_column<double, 3>(
        du.data(), w.data(), mass_fix.data(), delta.data(), vol.data(), nlayer,
        0, ny, stride1, stride2);
    snap::vic_tangential_momentum_column(du.data(), w.data(), mass_fix.data(),
                                         vol.data(), nlayer, 0, stride1,
                                         stride2);
    for (int i = 0; i < nlayer; ++i)
      snap::vic_redistribute_cell<double, 3>(du.data(), mass_fix.data(),
                                             delta.data(), i, 0, ny, stride1,
                                             stride2);

    if (!wind) {
      for (int n : {snap::IVY, snap::IVZ})
        for (int i = 0; i < nlayer; ++i) {
          EXPECT_EQ(du[n * stride1 + i], original[n * stride1 + i]);
          EXPECT_EQ(std::signbit(du[n * stride1 + i]),
                    std::signbit(original[n * stride1 + i]));
        }
      continue;
    }
    for (int n : {snap::IVY, snap::IVZ})
      EXPECT_NEAR(column_integral(du, vol, n, stride1, 0, nlayer - 1), 0.,
                  1.e-12);
    for (int i = 0; i < nlayer; ++i) {
      double rho1 = rho[i] + du[snap::IDN * stride1 + i];
      EXPECT_NE(du[snap::IDN * stride1 + i], 0.);
      EXPECT_NEAR((rho[i] * v2 + du[snap::IVY * stride1 + i]) / rho1, v2,
                  1.e-12);
      EXPECT_NEAR((rho[i] * v3 + du[snap::IVZ * stride1 + i]) / rho1, v3,
                  1.e-12);
    }
  }
}

TEST(implicit_options, parses_implicit_scheme_bits) {
  auto partial = snap::ImplicitOptionsImpl::from_yaml(YAML::Load("1"));
  ASSERT_TRUE(partial);
  EXPECT_EQ(partial->scheme(), 1);
  EXPECT_EQ(partial->size(), 3);
  EXPECT_EQ(partial->type(), "vic-partial");

  auto full = snap::ImplicitOptionsImpl::from_yaml(YAML::Load("9"));
  ASSERT_TRUE(full);
  EXPECT_EQ(full->scheme(), 9);
  EXPECT_EQ(full->size(), 5);
  EXPECT_EQ(full->type(), "vic-full");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
