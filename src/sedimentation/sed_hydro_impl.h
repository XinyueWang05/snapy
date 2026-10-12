#pragma once

// base
#include <configure.h>

// C/C++
#include <cstdint>

// snap
#include <snap/coord/coord_utils_impl.h>
#include <snap/snap.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO T sedimentation_feps(T const* w, int flat, int stride_var,
                                           int ny, int nvapor,
                                           T const* inv_mu_ratio_m1) {
  T feps = 1.;
  for (int n = 0; n < nvapor; ++n) {
    feps += w[(ICY + n) * stride_var + flat] * inv_mu_ratio_m1[n];
  }
  for (int n = nvapor; n < ny; ++n) {
    feps -= w[(ICY + n) * stride_var + flat];
  }
  return feps;
}

//! Adds to x1 face `face` the sedimentation flux leaving cell `cell`, for the
//! particles moving toward that face: rising (vsed > 0) when `rising`, else
//! settling (vsed < 0). The carried velocity is accumulated into vsed_out.
template <typename T>
inline DISPATCH_MACRO void sedimentation_donor_impl(
    T const* w, T* flux, T* vsed_out, T const* cosine_cell_kj, T const* radius,
    T const* density, T const* const_vsed, int64_t const* hydro_ids,
    T const* inv_mu_ratio_m1, T const* cv_ratio_m1, T const* u0, int nparticle,
    int ny, int nvapor, int nc3, int nc2, int nc1, int cell, int face,
    bool rising, T grav, T gas_constant_dry, T cv_dry, T gas_diameter,
    T gas_epsilon_lj, T gas_mass, T upper_limit, T pi, T kboltz) {
  int ncells = nc1 * nc2 * nc3;
  int j = (cell / nc1) % nc2;
  int k = cell / (nc1 * nc2);

  T rho = w[IDN * ncells + cell];
  T pres = w[IPR * ncells + cell];
  T feps = sedimentation_feps(w, cell, ncells, ny, nvapor, inv_mu_ratio_m1);
  T temp = pres / (rho * gas_constant_dry * feps);

  T v1 = w[IVX * ncells + cell];
  T v2 = w[IVY * ncells + cell];
  T v3 = w[IVZ * ncells + cell];
  T cth = cosine_cell_kj[k * nc2 + j];
  coord_vec_lower_impl(&v2, &v3, cth);
  T ke = T(0.5) * (w[IVX * ncells + cell] * v1 + w[IVY * ncells + cell] * v2 +
                   w[IVZ * ncells + cell] * v3);

  // eta and lambda are computed on the Stokes branch only, once per cell
  T eta = T(0), lambda = T(0);
  bool have_eta = false;
  // with these, a Stokes vsed has the sign of its drive term (or is NaN)
  bool drive_sign = pres > T(0) && temp > T(0) && !(upper_limit < T(0));

  for (int p = 0; p < nparticle; ++p) {
    int hydro_id = static_cast<int>(hydro_ids[p]);
    int species_id = hydro_id - ICY;
    // A non-zero const_vsed is the whole velocity (prescribed, athena
    // convention); otherwise Stokes. grav is the SIGNED x1 acceleration,
    // so dense particles settle toward -x1.
    T vsed;
    if (const_vsed[p] != T(0)) {
      vsed = const_vsed[p];
    } else {
      T r = radius[p];
      T drive = T(2) * r * r * grav * (density[p] - rho);
      // skip a particle moving away from this face before evaluating its vsed
      if (drive_sign && r > T(0) &&
          (rising ? !(drive > T(0)) : !(drive < T(0))))
        continue;
      if (!have_eta) {
        eta = (T(5) / T(16)) * sqrt(pi * kboltz) * sqrt(gas_mass) * sqrt(temp) *
              pow(kboltz / gas_epsilon_lj * temp, T(0.16)) /
              (pi * gas_diameter * gas_diameter * T(1.22));
        lambda =
            (eta * sqrt(pi * kboltz * temp)) / (pres * sqrt(T(2) * gas_mass));
        have_eta = true;
      }
      T kn = lambda / r;
      T beta = T(1) + kn * (T(1.256) + T(0.4) * exp(-T(1.1) / kn));
      vsed = beta / (T(9) * eta) * drive;
    }
    if (vsed < -upper_limit) vsed = -upper_limit;
    if (vsed > upper_limit) vsed = upper_limit;
    if (rising ? !(vsed > T(0)) : !(vsed < T(0))) continue;
    vsed_out[p * ncells + face] += vsed;

    T y = w[hydro_id * ncells + cell];
    T rhos_vsed = rho * y * vsed;
    T species_energy = rho * y *
                       (u0[1 + species_id] +
                        (T(1) + cv_ratio_m1[species_id]) * cv_dry * temp + ke);

    flux[hydro_id * ncells + face] += rhos_vsed;
    flux[IVX * ncells + face] += v1 * rhos_vsed;
    flux[IVY * ncells + face] += v2 * rhos_vsed;
    flux[IVZ * ncells + face] += v3 * rhos_vsed;
    flux[IPR * ncells + face] += vsed * species_energy;
  }
}

//! Sedimentation flux at x1 face `flat` (between cells i-1 and i), taken from
//! the upwind cell of each particle: cell i if it settles, cell i-1 if it
//! rises. Faces at or below il and above iu are sealed.
template <typename T>
inline DISPATCH_MACRO void sedimentation_flux_impl(
    T const* w, T* flux, T* vsed_out, T const* cosine_cell_kj, T const* radius,
    T const* density, T const* const_vsed, int64_t const* hydro_ids,
    T const* inv_mu_ratio_m1, T const* cv_ratio_m1, T const* u0, int nparticle,
    int ny, int nvapor, int nvar, int nc3, int nc2, int nc1, int flat, int il,
    int iu, T grav, T gas_constant_dry, T cv_dry, T gas_diameter,
    T gas_epsilon_lj, T gas_mass, T upper_limit, T pi, T kboltz) {
  int ncells = nc1 * nc2 * nc3;
  int i = flat % nc1;

  for (int p = 0; p < nparticle; ++p) vsed_out[p * ncells + flat] = T(0);
  if (i <= il || i > iu) return;

  sedimentation_donor_impl(w, flux, vsed_out, cosine_cell_kj, radius, density,
                           const_vsed, hydro_ids, inv_mu_ratio_m1, cv_ratio_m1,
                           u0, nparticle, ny, nvapor, nc3, nc2, nc1, flat, flat,
                           /*rising=*/false, grav, gas_constant_dry, cv_dry,
                           gas_diameter, gas_epsilon_lj, gas_mass, upper_limit,
                           pi, kboltz);
  if (i > 0) {
    sedimentation_donor_impl(w, flux, vsed_out, cosine_cell_kj, radius, density,
                             const_vsed, hydro_ids, inv_mu_ratio_m1,
                             cv_ratio_m1, u0, nparticle, ny, nvapor, nc3, nc2,
                             nc1, flat - 1, flat, /*rising=*/true, grav,
                             gas_constant_dry, cv_dry, gas_diameter,
                             gas_epsilon_lj, gas_mass, upper_limit, pi, kboltz);
  }
}

}  // namespace snap
