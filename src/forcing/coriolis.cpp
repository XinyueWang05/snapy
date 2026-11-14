// yaml
#include <cstdlib>
#include <string>

#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

CoriolisOptions CoriolisOptionsImpl::from_yaml(YAML::Node const &forcing) {
  if (!forcing["coriolis"]) return nullptr;

  auto node = forcing["coriolis"];
  check_keys(node, "forcing/coriolis",
             {"type", "omega1", "omega2", "omega3", "traditional"});
  auto op = CoriolisOptionsImpl::create();

  op->type() = node["type"].as<std::string>("xyz");
  TORCH_CHECK(op->type() == "xyz" || op->type() == "123",
              "CoriolisOptions: unsupported type ", op->type());

  op->omega1() = node["omega1"].as<double>(0.);
  op->omega2() = node["omega2"].as<double>(0.);
  op->omega3() = node["omega3"].as<double>(0.);
  op->traditional() = node["traditional"].as<bool>(false);

  return op;
}

Coriolis123Impl::Coriolis123Impl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void Coriolis123Impl::reset() {
  TORCH_CHECK(phydro, "[Coriolis123] Parent Hydro is null");
}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (options->omega1() != 0.0 || options->omega2() != 0.0 ||
      options->omega3() != 0.0) {
    auto m1 = w[IDN] * w[IVX];
    auto m2 = w[IDN] * w[IVY];
    auto m3 = w[IDN] * w[IVZ];
    du[IVX] += 2. * dt * (options->omega3() * m2 - options->omega2() * m3);
    du[IVY] += 2. * dt * (options->omega1() * m3 - options->omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[IVZ] += 2. * dt * (options->omega2() * m1 - options->omega1() * m2);
    }
  }

  return du;
}

CoriolisXYZImpl::CoriolisXYZImpl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void CoriolisXYZImpl::reset() {
  TORCH_CHECK(phydro, "[CoriolisXYZ] Parent Hydro is null");
  auto pcoord = phydro->pmb->pcoord;

  cubed_sphere = false;
  shallow_water = false;
  traditional = false;
  face_id = -1;

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");

  auto omegaz = options->omega1();
  auto omegax = options->omega2();
  auto omegay = options->omega3();

  if (pcoord->options->type() == "cartesian") {
    omega1 = omegaz * ones_like(mesh[0]);
    omega2 = omegax * ones_like(mesh[0]);
    omega3 = omegay * ones_like(mesh[0]);
  } else if (pcoord->options->type() == "cylindrical") {
    auto theta = mesh[1];

    omega1 = theta.cos() * omegax + theta.sin() * omegay;
    omega2 = -theta.sin() * omegax + theta.cos() * omegay;
    omega3 = omegaz * ones_like(mesh[0]);
  } else if (pcoord->options->type() == "spherical-polar") {
    auto theta = mesh[1];
    auto phi = mesh[0];

    omega1 = theta.sin() * phi.cos() * omegax +
             theta.sin() * phi.sin() * omegay + theta.cos() * omegaz;
    omega2 = theta.cos() * phi.cos() * omegax +
             theta.cos() * phi.sin() * omegay - theta.sin() * omegaz;
    omega3 = -phi.sin() * omegax + phi.cos() * omegay;
  } else if (pcoord->options->type() == "gnomonic-equiangle") {
    int r = phydro->pmb->options->layout()->rank();
    auto layout = phydro->pmb->get_layout();
    auto face = std::get<2>(layout->loc_of(r));

    cubed_sphere = true;
    shallow_water = phydro->peos->options->type() == "shallow-water";
    traditional = options->traditional() || shallow_water;
    face_id = face;
    alpha = register_buffer("alpha", mesh[1]);
    beta = register_buffer("beta", mesh[0]);

    auto omega =
        torch::empty({3, mesh[0].size(0), mesh[0].size(1), mesh[0].size(2)},
                     mesh[0].options());
    omega[VEL1].fill_(omegaz);
    omega[VEL2].fill_(omegax);
    omega[VEL3].fill_(omegay);
    cs_cart_to_contra_(omega, alpha, beta, face_id);
    cs_contra_to_sph_(omega, alpha, beta, face_id);

    omega1 = omega[VEL1];
    omega2 = omega[VEL2];
    omega3 = omega[VEL3];

    auto alpha_plane = alpha.narrow(-1, 0, 1);
    auto beta_plane = beta.narrow(-1, 0, 1);
    contra_to_spherical = register_buffer(
        "contra_to_spherical",
        cs_velocity_transform_matrix(alpha_plane, beta_plane, face_id, true));
    spherical_to_covariant = register_buffer(
        "spherical_to_covariant",
        cs_velocity_transform_matrix(alpha_plane, beta_plane, face_id, false,
                                     pcoord->cosine_cell_kj.narrow(-1, 0, 1)));
  } else {
    throw std::runtime_error("CoriolisXYZ: unsupported coordinate system");
  }

  register_buffer("omega1", omega1);
  register_buffer("omega2", omega2);
  register_buffer("omega3", omega3);
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (cubed_sphere) {
    auto force = w.narrow(0, IVX, 3).clone();
    force *= w[IDN].unsqueeze(0);
    cs_apply_velocity_transform_(force, contra_to_spherical);

    auto o1 = omega1;
    auto o2 = omega2;
    auto o3 = omega3;
    if (traditional) {
      o2 = torch::zeros_like(omega2);
      o3 = torch::zeros_like(omega3);
    }

    auto m1 = force[VEL1].clone();
    auto m2 = force[VEL2].clone();
    auto m3 = force[VEL3].clone();
    force[VEL1] = 2. * (o3 * m2 - o2 * m3);
    force[VEL2] = 2. * (o1 * m3 - o3 * m1);
    force[VEL3] = 2. * (o2 * m1 - o1 * m2);

    cs_apply_velocity_transform_(force, spherical_to_covariant);
    // dynamics/conserve-axial-momentum: full (#314): the component of the
    // Coriolis force along the rotation about the omega1 axis in flux form,
    // from the face mass fluxes Hydro names for this call, so the exchange
    // between relative and planetary AAM closes exactly
    // (axial_coriolis_rate). The traditional approximation (and shallow
    // water) has no w rows: only the x2 and x3 faces enter, and what the
    // planetary AAM gains from mass moved along x1 is the term those
    // equations leave out. This makes the force not skew-symmetric (the rest
    // of it uses the cell momentum): see the note at the option.
    if (phydro->axial_skew_form()) {
      // #357 option A (traditional / shallow water): the horizontal force is
      // the skew-symmetric operator from interpolated momentum, which does no
      // work (GnomonicEquiangle::axial_skew_coriolis_). Its AAM exchange
      // closes against the interpolated momentum flux; the rest of the
      // solver's mass-flux torque (its dissipative part and the operator's
      // own K residual) is stored here and carried by one global uniform
      // angular acceleration after the stage (MeshBlock::apply_axial_torque).
      auto pmb = phydro->pmb;
      auto pcoord = pmb->pcoord;
      int sj = pcoord->jl(), ej = pcoord->ju() + 1;
      int sk = pcoord->kl(), ek = pcoord->ku() + 1;
      int si = pcoord->il(), ei = pcoord->iu() + 1;
      // internal comparison switch (#357 option A): variant C keeps the
      // cell-form force (pointwise skew) and lets the global uniform torque
      // carry the whole Coriolis + mass-flux AAM error; default S
      static bool const variant_c = [] {
        char const *v = std::getenv("SNAP_AAM_FULL_VARIANT");
        return v && std::string(v) == "C";
      }();
      torch::Tensor sums;
      if (variant_c) {
        auto bufs = pmb->named_buffers();
        auto in = [&](torch::Tensor const &t) {
          return t.slice(-3, sk, ek).slice(-2, sj, ej).slice(-1, si, ei);
        };
        auto k = bufs["coord.aam_k_c"].slice(1, sk, ek).slice(2, sj, ej);
        auto kk = bufs["coord.aam_kk_c"].slice(0, sk, ek).slice(1, sj, ej);
        auto r = bufs["coord.x1v"].slice(0, si, ei);
        auto vol = in(pcoord->cell_volume());
        auto ktq = (vol * r * (k[VEL2] * in(force[VEL2]) +
                               k[VEL3] * in(force[VEL3])))
                       .sum();
        auto inertia = (vol * in(w[IDN]) * r * r * kk).sum();
        sums = torch::stack(std::vector<torch::Tensor>{ktq, inertia});
      } else {
      force[VEL2].zero_();
      force[VEL3].zero_();
      auto cells = pcoord->axial_skew_cells(w);
      // per direction: "l" carries the own last cell at the upper face (sent
      // +) and receives the lower neighbour at the lower face; "r" the
      // reverse. The received slots start as the ghost cells, which an
      // intra-panel neighbour's are (exact copies in the same basis); the
      // cross-panel exchange overwrites them with the neighbour's own cell
      // (no interpolation, rotated into this panel's face basis).
      auto edges = [&](int dim, int s, int e) {
        auto el = torch::zeros_like(cells), er = torch::zeros_like(cells);
        el.select(dim, e).copy_(cells.select(dim, e - 1));
        el.select(dim, s).copy_(cells.select(dim, s - 1));
        er.select(dim, s).copy_(cells.select(dim, s));
        er.select(dim, e).copy_(cells.select(dim, e));
        return std::make_pair(el, er);
      };
      auto [e2l, e2r] = edges(2, sj, ej);
      auto [e3l, e3r] = edges(1, sk, ek);
      auto playout = pmb->get_layout();
      if (playout && playout->options->type() == "cubed-sphere") {
        SyncOptions sync_opts;
        sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);
        std::vector<CommWorkPtr> works2, works3;
        Variables sv2, sv3;
        sv2["aam_skew_l:+"] = e2l;
        sv2["aam_skew_r:-"] = e2r;
        sv3["aam_skew_l:+"] = e3l;
        sv3["aam_skew_r:-"] = e3r;
        pmb->begin_exchange(sv2, sync_opts.dim(SyncOptions::DIM2));
        pmb->begin_exchange(sv3, sync_opts.dim(SyncOptions::DIM3));
        pmb->launch_exchange(sync_opts.dim(SyncOptions::DIM2), works2);
        pmb->launch_exchange(sync_opts.dim(SyncOptions::DIM3), works3);
        pmb->finalize_exchange(sv2, sync_opts.dim(SyncOptions::DIM2), works2);
        pmb->finalize_exchange(sv3, sync_opts.dim(SyncOptions::DIM3), works3);
      }
      sums = pcoord->axial_skew_coriolis_(force[VEL2], force[VEL3], w[IDN],
                                          cells, e2l, e2r, e3l, e3r,
                                          options->omega1());
      }
      // the torque the solver's mass fluxes move between relative and
      // planetary AAM (x2, x3 faces), summed over the interior
      auto const &mf = phydro->axial_mass_fluxes();
      double target = 0.;
      if (mf[1].defined() || mf[2].defined()) {
        auto rate = pcoord->axial_coriolis_rate(torch::Tensor(), mf[1], mf[2],
                                                options->omega1(),
                                                phydro->axial_mass_species());
        auto vol = pcoord->cell_volume()
                       .slice(0, sk, ek)
                       .slice(1, sj, ej)
                       .slice(2, pcoord->il(), pcoord->iu() + 1);
        target = (rate * vol).sum().item<double>();
      }
      auto host = sums.to(torch::kCPU, torch::kFloat64);
      phydro->axial_torque_sums()[0].fill_(target);
      phydro->axial_torque_sums()[1].fill_(host[0].item<double>());
      phydro->axial_torque_sums()[2].fill_(host[1].item<double>());
      phydro->axial_torque_rho().set_(w[IDN]
                                          .slice(0, sk, ek)
                                          .slice(1, sj, ej)
                                          .slice(2, pcoord->il(), pcoord->iu() + 1)
                                          .clone());
    } else if (phydro->axial_coriolis_form()) {
      auto pcoord = phydro->pmb->pcoord;
      auto const &mf = phydro->axial_mass_fluxes();
      auto f1 = traditional ? torch::Tensor() : mf[0];
      if (f1.defined() || mf[1].defined() || mf[2].defined()) {
        auto target = pcoord->axial_coriolis_rate(
            f1, mf[1], mf[2], options->omega1(), phydro->axial_mass_species());
        pcoord->axial_rate_to_rows_(force[VEL2], force[VEL3], target);
      }
    }
    if (traditional) {
      du.narrow(0, IVY, 2) += dt * force.narrow(0, VEL2, 2);
    } else {
      du.narrow(0, IVX, 3) += dt * force;
    }
    return du;
  }

  auto m1 = w[IDN] * w[IVX];
  auto m2 = w[IDN] * w[IVY];
  auto m3 = w[IDN] * w[IVZ];

  du[IVX] += 2. * dt * (omega3 * m2 - omega2 * m3);
  du[IVY] += 2. * dt * (omega1 * m3 - omega3 * m1);
  du[IVZ] += 2. * dt * (omega2 * m1 - omega1 * m2);

  return du;
}
}  // namespace snap
