// C/C++
#include <algorithm>

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
  if (!mirror_src_.is_same(solid) || mirror_ver_ != solid._version()) {
    mirrors_.clear();
    mirror_src_ = solid, mirror_ver_ = solid._version();
  }
  if (mirrors_.count(dim)) return &mirrors_.at(dim);
  // x2/x3 only: x1 is the per-column wall (x1_wall)
  TORCH_CHECK(dim != 3, "mirror_index: x1 goes through x1_wall");
  int ng = pmb->pcoord->options->nghost(), axis = dim - 1,
      il = pmb->pcoord->il();
  auto s = solid.to(torch::kCPU, torch::kBool).clone();
  // a solid end cell continues into the x1 ghosts beyond it
  if (pmb->options->is_physical_boundary(0, 0, -1))
    s.narrow(2, 0, il) |= s.narrow(2, il, 1);
  int top = s.size(2) - il;
  if (pmb->options->is_physical_boundary(0, 0, 1))
    s.narrow(2, top, il) |= s.narrow(2, top - 1, 1);
  auto sl = s.movedim(axis, -1).contiguous();
  int n = sl.size(-1), lines = sl.numel() / n;
  auto sp = sl.data_ptr<bool>();
  // each fluid run [a, b) with a solid end is a wall frame
  Mirror m;
  std::map<std::pair<int, int>, std::vector<int64_t>> runs;
  for (int l = 0; l < lines; ++l)
    for (int a = 0, b; a < n; a = b) {
      for (; a < n && sp[l * n + a]; ++a) {
      }
      for (b = a; b < n && !sp[l * n + b]; ++b) {
      }
      if (a < b && (a > 0 || b < n)) runs[{a, b}].push_back(l);
    }
  auto dev = solid.device();
  for (auto const &[ab, ls] : runs) {
    auto [a, b] = ab;
    // frame cell p is cell a - ng + p, folded into [a, b) at solid ends
    int len = b - a + 2 * ng;
    std::vector<int64_t> src(len);
    std::vector<double> sg(len, 1.);
    for (int p = 0; p < len; ++p) {
      int x = a - ng + p;
      while ((x < a && a > 0) || (x >= b && b < n))
        x = x < a ? 2 * a - 1 - x : 2 * b - 1 - x, sg[p] = -sg[p];
      src[p] = std::clamp(x, 0, n - 1);
    }
    m.span.push_back({a, b});
    m.lines.push_back(torch::tensor(ls, torch::kLong).to(dev));
    m.src.push_back(torch::tensor(src, torch::kLong).to(dev));
    auto wide = dev.is_mps() ? torch::kFloat32 : torch::kFloat64;  // MPS
    m.sign.push_back(torch::tensor(sg, torch::kFloat64).to(dev, wide));
  }
  // x2/x3 face covariance: no covariance on a face beside a solid
  auto f = solid.to(torch::kBool) | solid.to(torch::kBool).roll(1, axis);
  m.cov_keep = (~f).to(solid.is_mps() ? torch::kFloat32 : torch::kFloat64);
  return &(mirrors_[dim] = m);
}

torch::Tensor InternalBoundaryImpl::reconstruct(
    torch::Tensor v, Mirror const &m, int dim, int normal,
    std::function<torch::Tensor(torch::Tensor const &)> const &recon) const {
  auto lr = recon(v);  // lines without a solid end
  if (m.span.empty()) return lr;
  int ng = pmb->pcoord->options->nghost(), n = v.size(dim);
  auto vl = v.movedim(dim, -1).flatten(1, 2);  // [nv, lines, n]
  auto out = lr.movedim(dim + 1, -1).contiguous();
  auto sh = out.sizes().vec();
  out = out.flatten(2, 3);  // [2, nv, lines, n]
  for (size_t k = 0; k < m.span.size(); ++k) {
    auto [a, b] = m.span[k];
    auto f = vl.index_select(1, m.lines[k]).index_select(-1, m.src[k]);
    if (normal >= 0) f[normal].mul_(m.sign[k].to(f.dtype()));
    auto r = recon(f.unsqueeze(1).movedim(-1, dim)).movedim(dim + 1, -1);
    // the run's faces a..b that the block sweep defines: frame index - a + ng
    int fa = std::max(a, ng), fb = std::min(b, n - ng);
    if (fb >= fa)
      out.narrow(-1, fa, fb - fa + 1)
          .index_copy_(2, m.lines[k],
                       r.flatten(2, 3).narrow(-1, fa - a + ng, fb - fa + 1));
  }
  return out.view(sh).movedim(-1, dim + 1);
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
  if (x1wall_ && x1wall_src_.is_same(solid) && x1wall_ver_ == solid._version())
    return x1wall_.get();
  auto pc = pmb->pcoord;
  int il = pc->il(), iu = pc->iu(), nx1 = iu - il + 1;
  auto s = solid.to(torch::kCPU, torch::kBool);
  // depth: the solid run from il up; no fluid cell: depth 0
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
    // a surface needs its whole x1 column and 3 fluid cells
    TORCH_CHECK(pmb->options->is_physical_boundary(0, 0, -1) &&
                    pmb->options->is_physical_boundary(0, 0, 1),
                "a solid surface needs the whole x1 column on one block "
                "(nb1 = 1): the implicit scheme requires it, and an explicit "
                "run can split its blocks in x2/x3 instead");
    // paths the per-column wall does not take yet, refused by name
    auto const &hy = pmb->phydro;
    TORCH_CHECK(!(x1_centroid_exact_enabled() &&
                  pc->options->type() == "spherical-polar"),
                "a solid surface with SNAP_X1_CENTROID_EXACT (x1 plain-mean "
                "and pressure-source stencils) is not supported");
    TORCH_CHECK(!hy->fp32(),
                "a solid surface with fp32 compute is not supported");
    TORCH_CHECK(!hy->options->perturbation_flux(),
                "perturbation-flux does not support immersed solids");
    int dmax = w->cell.depths.back();
    // key 4 needs 4 fluid cells, as a wall column does
    TORCH_CHECK(hy->options->conversion_order() != 4 || nx1 - dmax >= 4,
                "a solid column has ", nx1 - dmax,
                " fluid cells; dynamics/"
                "conversion-order 4 (key-4 wall ghosts, wb_ref4 face density) "
                "needs at least 4");
    TORCH_CHECK(nx1 - dmax >= 3, "a solid column has ", nx1 - dmax,
                " fluid cells; the x1 wall's one-sided 3-point slopes "
                "(d1_pressure, x1 mass covariance, corrected-PE work) need at "
                "least 3");
  }
  x1wall_ = w;
  x1wall_src_ = solid, x1wall_ver_ = solid._version();
  return x1wall_.get();
}

torch::Tensor InternalBoundaryImpl::per_column(
    X1Group const &g, torch::Tensor const &t,
    std::function<torch::Tensor(torch::Tensor const &, int)> const &fn,
    bool zero_pad, bool input_pad) const {
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
      if (input_pad)  // t's own levels: x1v keeps a nonzero spacing there
        low = flat.index_select(-2, c).narrow(-1, 0, o).to(r.options());
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
