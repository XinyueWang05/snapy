// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>
#include <snap/coord/x1_centroid.hpp>
#include <snap/mesh/meshblock.hpp>

#include "internal_boundary.hpp"

namespace snap {

InternalBoundaryOptions InternalBoundaryOptionsImpl::from_yaml(
    std::string const &filename) {
  YAML::Node config = YAML::LoadFile(filename);
  auto op = InternalBoundaryOptionsImpl::create();

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["internal"]) return op;

  return from_yaml(config["boundary-condition"]["internal"]);
}

InternalBoundaryOptions InternalBoundaryOptionsImpl::from_yaml(
    const YAML::Node &node) {
  check_keys(node, "boundary-condition/internal",
             {"max-iter", "solid-density", "solid-pressure"});
  auto op = InternalBoundaryOptionsImpl::create();

  op->max_iter() = node["max-iter"].as<int>(5);
  op->solid_density() = node["solid-density"].as<double>(1.e3);
  op->solid_pressure() = node["solid-pressure"].as<double>(1.e9);

  return op;
}

InternalBoundaryImpl::InternalBoundaryImpl(
    InternalBoundaryOptions const &options_, torch::nn::Module *p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const *>(p);
  reset();
}

void InternalBoundaryImpl::reset() {}

void InternalBoundaryImpl::mark_prim_solid_(torch::Tensor w,
                                            torch::Tensor solid) const {
  if (!solid.defined()) return;

  w[IDN].masked_fill_(solid, options->solid_density());
  w[IPR].masked_fill_(solid, options->solid_pressure());
  w.narrow(0, IVX, 3).masked_fill_(solid, 0.);

  int ny = w.size(0) - ICY;
  if (ny > 0) {
    w.narrow(0, ICY, ny)
        .masked_fill_(solid.unsqueeze(0).expand_as(w.narrow(0, ICY, ny)), 0.);
  }
}

void InternalBoundaryImpl::fill_cons_solid_(torch::Tensor u,
                                            torch::Tensor solid,
                                            torch::Tensor fill) const {
  if (!solid.defined()) return;

  u.set_(torch::where(solid.unsqueeze(0).expand_as(u), fill, u));
}

torch::Tensor InternalBoundaryImpl::forward(torch::Tensor wlr, int dim,
                                            torch::Tensor solid) const {
  if (!solid.defined()) return wlr;

  auto solidl = solid;
  auto solidr = solid.roll(1, dim - 1);
  solidr.select(dim - 1, 0) = solidl.select(dim - 1, 0);

  for (size_t n = 0; n < wlr.size(1); ++n) {
    wlr[IRT][n] = torch::where(solidl, wlr[ILT][n], wlr[IRT][n]);
    wlr[ILT][n] = torch::where(solidr, wlr[IRT][n], wlr[ILT][n]);
  }

  if (dim == 3) {
    wlr[IRT][IVX] = torch::where(solidl, -wlr[ILT][IVX], wlr[IRT][IVX]);
    wlr[ILT][IVX] = torch::where(solidr, -wlr[IRT][IVX], wlr[ILT][IVX]);
  } else if (dim == 2) {
    wlr[IRT][IVY] = torch::where(solidl, -wlr[ILT][IVY], wlr[IRT][IVY]);
    wlr[ILT][IVY] = torch::where(solidr, -wlr[IRT][IVY], wlr[ILT][IVY]);
  } else if (dim == 1) {
    wlr[IRT][IVZ] = torch::where(solidl, -wlr[ILT][IVZ], wlr[IRT][IVZ]);
    wlr[ILT][IVZ] = torch::where(solidr, -wlr[IRT][IVZ], wlr[ILT][IVZ]);
  }

  return wlr;
}
InternalBoundaryImpl::Mirror const *InternalBoundaryImpl::mirror_index(
    std::map<std::string, torch::Tensor> const &other, int dim) const {
  if (!other.count("solid")) return nullptr;
  auto solid = other.at("solid");
  if (mirror_src_ != solid.data_ptr() || mirror_ver_ != solid._version()) {
    mirrors_.clear();
    mirror_src_ = solid.data_ptr(), mirror_ver_ = solid._version();
  }
  if (mirrors_.count(dim)) return &mirrors_.at(dim);
  // x2/x3 only: x1 is the per-column wall (x1_wall)
  TORCH_CHECK(dim != 3, "mirror_index: x1 goes through x1_wall");
  int reach = pmb->pcoord->options->nghost() - 1;
  int axis = dim - 1, il = pmb->pcoord->il();
  bool phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = pmb->options->is_physical_boundary(0, 0, 1);
  auto s = solid.to(torch::kCPU, torch::kBool).clone();
  // x2/x3: a solid end cell continues into the x1 ghosts beyond it
  if (phys_in) s.narrow(2, 0, il) |= s.narrow(2, il, 1);
  int top = s.size(2) - il;  // the first x1 ghost above
  if (phys_out) s.narrow(2, top, il) |= s.narrow(2, top - 1, 1);
  s = s.movedim(axis, -1).contiguous();
  int n = s.size(-1), lines = s.numel() / n;
  auto sp = s.data_ptr<bool>();
  // the solid runs [a, b] of each line; two copies only on a line with an
  // interior run too thin to serve both sides
  std::vector<std::vector<std::pair<int, int>>> lruns(lines);
  std::vector<char> thin(lines, 0);
  bool two = false;
  for (int l = 0; l < lines; ++l)
    for (int a = 0, b; a < n; a = b + 1) {
      for (; a < n && !sp[l * n + a]; ++a) {
      }
      if (a == n) break;
      for (b = a; b + 1 < n && sp[l * n + b + 1]; ++b) {
      }
      lruns[l].push_back({a, b});
      if (a > 0 && b + 1 < n && b - a + 1 < 2 * reach)
        two = thin[l] = 1;
    }
  int nc = two ? 2 : 1;
  auto idx = torch::arange(n, torch::kLong).repeat({nc, lines, 1});
  auto odd = torch::zeros({nc, lines, n}, torch::kBool);
  auto pick = torch::zeros({lines, n}, torch::kLong);
  auto ip = idx.data_ptr<int64_t>(), pp = pick.data_ptr<int64_t>();
  auto op = odd.data_ptr<bool>();
  for (int l = 0; l < lines; ++l) {
    bool tl = thin[l];
    auto const &runs = lruns[l];
    // the fluid source of cell j mirrored across face `face`, folded back
    // into the fluid gap [lo, hi]; returns (source, odd reflection count)
    auto reflect = [&](int j, int face, int lo, int hi) {
      int src = 2 * face - 1 - j, k = 1;
      while (src < lo || src > hi) {
        src = src < lo ? 2 * lo - 1 - src : 2 * (hi + 1) - 1 - src;
        ++k;
      }
      return std::make_pair(src, k % 2 == 1);
    };
    int nr = runs.size();
    for (int c = 0; c < (tl ? 2 : 1); ++c) {
      int64_t *ic = ip + (c * lines + l) * n;
      bool *oc = op + (c * lines + l) * n;
      auto put = [&](int j, int face, int lo, int hi) {
        auto [src, o] = reflect(j, face, lo, hi);
        ic[j] = src, oc[j] = o;
      };
      // one copy: each run's own cells from its nearer fluid side; two copies:
      // run r from below in copy (r % 2), from above in the other, its
      // mirror running past a thin run into the fluid beyond (extensions
      // first, the runs' own cells last)
      for (int pass = 0; pass < 2; ++pass)
        for (int r = 0; r < nr; ++r) {
          auto [a, b] = runs[r];
          int glo = r ? runs[r - 1].second + 1 : 0;
          int ghi = r + 1 < nr ? runs[r + 1].first - 1 : n - 1;
          bool below = a > 0 && (!tl || (r + c) % 2 == 0);
          bool above = b + 1 < n && (!tl || (r + c) % 2 == 1);
          for (int k = 0; k < reach; ++k) {
            int jb = a + k, ja = b - k;
            if (below && jb < n && (jb <= b) == (pass == 1) &&
                (tl || !above || 2 * k < b - a + 1))
              put(jb, a, glo, a - 1);
            if (above && ja >= 0 && (ja >= a) == (pass == 1) &&
                (tl || !below || 2 * k < b - a + 1))
              put(ja, b + 1, b + 1, ghi);
          }
        }
    }
    // a cell reads the copy in which the run above it is seen from below
    for (int j = 0, r = 0; tl && j < n; ++j) {
      for (; r < nr && runs[r].second < j; ++r) {
      }
      pp[l * n + j] = r < nr ? r % 2 : nr % 2;
    }
  }
  auto shape = s.sizes().vec();
  auto back = [&](torch::Tensor t, int lead) {
    std::vector<int64_t> sh(shape);
    if (lead) sh.insert(sh.begin(), lead);
    return t.view(sh).movedim(-1, axis + (lead ? 1 : 0)).contiguous().to(
        solid.device());
  };
  // one gather per copy; the normal velocity times a sign (odd: -1)
  Mirror m;
  auto real = solid.is_mps() ? torch::kFloat32 : torch::kFloat64;
  auto i2 = back(idx, nc), sg = 1. - 2. * back(odd, nc).to(real);
  m.idx = i2[0], m.sign = sg[0];
  if (two) {  // copy 1 only on the x1 rows (lines: [other, x1]) of a thin line
    m.rows = torch::tensor(std::vector<int64_t>(thin.begin(), thin.end()))
                 .view({shape[0], shape[1]}).any(0).nonzero().flatten()
                 .to(solid.device());
    m.idx1 = i2[1].index_select(-1, m.rows);
    m.sign1 = sg[1].index_select(-1, m.rows);
    auto pr = back(pick, 0).eq(1);  // ILT from cell i - 1, IRT from cell i
    m.pick = torch::stack({pr.roll(1, axis), pr}).index_select(-1, m.rows);
  }
  // x2/x3 face covariance: no covariance on a face beside a solid
  auto f = solid.to(torch::kBool) | solid.to(torch::kBool).roll(1, axis);
  m.cov_keep = (~f).to(real);
  return &(mirrors_[dim] = m);
}

torch::Tensor InternalBoundaryImpl::mirror(torch::Tensor v, Mirror const &m,
                                           int dim, int normal,
                                           int copy) const {
  auto idx = copy ? m.idx1 : m.idx;
  int lead = v.dim() - 3;
  for (int k = 0; k < lead; ++k) idx = idx.unsqueeze(0);
  auto out = v.gather(lead + dim - 1, idx.expand_as(v));
  auto sign = copy ? m.sign1 : m.sign;
  if (normal >= 0) out[normal].mul_(sign.to(v.scalar_type()));
  return out;
}

torch::Tensor InternalBoundaryImpl::reconstruct(
    torch::Tensor v, Mirror const &m, int dim, int normal,
    std::function<torch::Tensor(torch::Tensor const &)> const &recon) const {
  auto lr = recon(mirror(v, m, dim, normal));
  if (!m.rows.defined()) return lr;
  auto lr1 = recon(mirror(v.index_select(-1, m.rows), m, dim, normal, 1));
  auto pick = lr.dim() == 5 ? m.pick.unsqueeze(1) : m.pick;
  return lr.index_copy_(
      -1, m.rows, torch::where(pick, lr1, lr.index_select(-1, m.rows)));
}

namespace {
InternalBoundaryImpl::X1Group x1_group(torch::Tensor const &depth,
                                       torch::Device dev) {
  InternalBoundaryImpl::X1Group g;
  g.depth = depth;
  auto flat = depth.flatten();
  auto u = std::get<0>(torch::_unique(flat, /*sorted=*/true));
  for (int k = 0; k < u.numel(); ++k) {
    int d = u[k].item<int64_t>();
    g.depths.push_back(d);
    g.cols.push_back(flat.eq(d).nonzero().flatten().to(dev));
  }
  return g;
}
}  // namespace

InternalBoundaryImpl::X1Wall const *InternalBoundaryImpl::x1_wall(
    std::map<std::string, torch::Tensor> const &other) const {
  if (!other.count("solid")) return nullptr;
  auto solid = other.at("solid");
  if (x1wall_ && x1wall_src_ == solid.data_ptr() &&
      x1wall_ver_ == solid._version())
    return x1wall_.get();
  auto pc = pmb->pcoord;
  int il = pc->il(), iu = pc->iu(), nx1 = iu - il + 1;
  auto s = solid.to(torch::kCPU, torch::kBool);
  // the bottom-attached run of each column: solid from il up; a column with
  // no fluid cell has no surface and stays as on main (depth 0)
  auto run = s.narrow(-1, il, nx1).to(torch::kLong).cumprod(-1).sum(-1);
  auto depth = run.masked_fill(run == nx1, 0);
  auto w = std::make_shared<X1Wall>();
  auto dev = solid.device();
  w->cell = x1_group(depth, dev);
  w->face2 = x1_group(torch::maximum(depth, depth.roll(1, 1)), dev);
  w->face3 = x1_group(torch::maximum(depth, depth.roll(1, 0)), dev);
  auto at = torch::arange(s.size(-1), torch::kLong);
  auto bottom = (at < (depth + il).unsqueeze(-1)) & (depth > 0).unsqueeze(-1);
  auto rest = s & ~bottom;
  if (rest.any().item<bool>()) w->rest = rest.to(dev);
  if (!w->cell.uniform()) {
    // the wall's own requirements, column by column: the whole x1 column on
    // this block, and the three fluid cells of the one-sided slopes
    TORCH_CHECK(pmb->options->is_physical_boundary(0, 0, -1) &&
                    pmb->options->is_physical_boundary(0, 0, 1),
                "a solid surface needs the whole x1 column on one block "
                "(nb1 = 1)");
    // paths the per-column wall does not take yet, refused by name
    auto const &hy = pmb->phydro;
    TORCH_CHECK(hy->options->conversion_order() != 4,
                "a solid surface with dynamics/conversion-order 4 (key-4 "
                "wall ghosts, wb_ref4) is not supported");
    TORCH_CHECK(!(x1_centroid_exact_enabled() &&
                  pc->options->type() == "spherical-polar"),
                "a solid surface with SNAP_X1_CENTROID_EXACT (x1 plain-mean "
                "and pressure-source stencils) is not supported");
    TORCH_CHECK(!hy->fp32(),
                "a solid surface with fp32 compute is not supported");
    TORCH_CHECK(!hy->options->perturbation_flux(),
                "perturbation-flux does not support immersed solids");
    int dmax = w->cell.depths.back();
    TORCH_CHECK(nx1 - dmax >= 3, "a solid column has ", nx1 - dmax,
                " fluid cells; the x1 wall's one-sided 3-point slopes "
                "(d1_pressure, x1 mass covariance, corrected-PE work) need at "
                "least 3");
  }
  x1wall_ = w;
  x1wall_src_ = solid.data_ptr(), x1wall_ver_ = solid._version();
  return x1wall_.get();
}

torch::Tensor InternalBoundaryImpl::per_column(
    X1Group const &g, torch::Tensor const &t,
    std::function<torch::Tensor(torch::Tensor const &, int)> const &fn,
    bool zero_pad) const {
  if (g.uniform()) return fn(t, 0);
  int64_t nc3 = t.size(-3), nc2 = t.size(-2), n = t.size(-1);
  auto flat = t.flatten(-3, -2);
  torch::Tensor out;
  for (size_t k = 0; k < g.depths.size(); ++k) {
    int o = g.depths[k];
    auto const &c = g.cols[k];
    auto r = fn(flat.index_select(-2, c).unsqueeze(-3).narrow(-1, o, n - o), o)
                 .squeeze(-3);
    if (o > 0) {
      auto sh = r.sizes().vec();
      sh.back() = o;
      auto low = zero_pad ? torch::zeros(sh, r.options())
                          : r.narrow(-1, 0, 1).expand(sh);
      r = torch::cat({low, r}, -1);
    }
    if (!out.defined()) {
      auto sh = r.sizes().vec();
      sh[sh.size() - 2] = nc3 * nc2;
      out = torch::zeros(sh, r.options());
    }
    out.index_copy_(-2, c, r);
  }
  auto sh = out.sizes().vec();
  int64_t m = sh.back();
  sh.resize(sh.size() - 2);
  sh.insert(sh.end(), {nc3, nc2, m});
  return out.view(sh);
}

torch::Tensor InternalBoundaryImpl::x1_wall_fill(X1Group const &g,
                                                 torch::Tensor const &v,
                                                 int normal) const {
  if (g.uniform()) return v;
  int ng = pmb->pcoord->il();
  return per_column(g, v, [&](torch::Tensor const &t, int o) {
    if (o == 0) return t;
    auto c = t.clone();  // reflecting_inner (bc_func.cpp) at this column's lo
    c.narrow(-1, 0, ng).copy_(c.narrow(-1, ng, ng).flip(-1));
    if (normal >= 0) c[normal].narrow(-1, 0, ng).neg_();
    return c;
  });
}

std::shared_ptr<InternalBoundaryImpl> InternalBoundaryImpl::create(
    InternalBoundaryOptions const &opts, torch::nn::Module *p,
    std::string const &name) {
  TORCH_CHECK(p != nullptr, "[InternalBoundary] Parent module is null");
  TORCH_CHECK(opts != nullptr, "[InternalBoundary] Options pointer is null");

  return p->register_module(name, InternalBoundary(opts, p));
}

}  // namespace snap
