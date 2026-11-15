// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>
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
InternalBoundaryImpl::Mirror const &InternalBoundaryImpl::mirror_index(
    torch::Tensor solid, int dim, int reach, bool bottom_only) const {
  if (mirror_src_ != solid.data_ptr() || mirror_ver_ != solid._version()) {
    mirrors_.clear();
    mirror_src_ = solid.data_ptr(), mirror_ver_ = solid._version();
  }
  auto key = std::make_tuple(dim, reach, bottom_only);
  if (mirrors_.count(key)) return mirrors_.at(key);
  int axis = dim - 1, il = pmb ? pmb->pcoord->il() : reach;
  bool phys_in = !pmb || pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = !pmb || pmb->options->is_physical_boundary(0, 0, 1);
  auto s = solid.to(torch::kCPU, torch::kBool).clone();
  // x2/x3: a solid end cell continues into the x1 ghosts beyond it
  if (!bottom_only && phys_in) s.narrow(2, 0, il) |= s.narrow(2, il, 1);
  int top = s.size(2) - il;  // the first x1 ghost above
  if (!bottom_only && phys_out) s.narrow(2, top, il) |= s.narrow(2, top - 1, 1);
  s = s.movedim(axis, -1).contiguous();
  int n = s.size(-1), lines = s.numel() / n;
  auto sp = s.data_ptr<bool>();
  // two copies only along an axis with a run too thin to serve both sides
  bool two = false;
  for (int l = 0; !bottom_only && l < lines; ++l)
    for (int a = 0, b; a < n; a = b + 1) {
      for (; a < n && !sp[l * n + a]; ++a) {
      }
      for (b = a; b + 1 < n && sp[l * n + b + 1]; ++b) {
      }
      if (a < n && a > 0 && b + 1 < n && b - a + 1 < 2 * reach) two = true;
    }
  int nc = two ? 2 : 1;
  auto idx = torch::arange(n, torch::kLong).repeat({nc, lines, 1});
  auto odd = torch::zeros({nc, lines, n}, torch::kBool);
  auto pick = torch::zeros({lines, n}, torch::kLong);
  auto surf = torch::zeros({lines}, torch::kInt);
  auto ip = idx.data_ptr<int64_t>(), pp = pick.data_ptr<int64_t>();
  auto op = odd.data_ptr<bool>();
  for (int l = 0; l < lines; ++l) {
    bool const *f = sp + l * n;
    std::vector<std::pair<int, int>> runs;
    for (int a = 0, b; a < n; a = b + 1) {
      for (; a < n && !f[a]; ++a) {
      }
      if (a == n) break;
      for (b = a; b + 1 < n && f[b + 1]; ++b) {
      }
      runs.push_back({a, b});
    }
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
    for (int c = 0; c < nc; ++c) {
      int64_t *ic = ip + (c * lines + l) * n;
      bool *oc = op + (c * lines + l) * n;
      auto put = [&](int j, int face, int lo, int hi) {
        auto [src, o] = reflect(j, face, lo, hi);
        ic[j] = src, oc[j] = o;
      };
      // one copy: each run's own cells from its nearer fluid side (bottom
      // run in x1: from above, into the ghosts below it too); two copies:
      // run r from below in copy (r % 2), from above in the other, its
      // mirror running past a thin run into the fluid beyond (extensions
      // first, the runs' own cells last)
      for (int pass = 0; pass < 2; ++pass)
        for (int r = 0; r < nr; ++r) {
          auto [a, b] = runs[r];
          if (bottom_only && !(a <= il && il <= b)) continue;
          // no fluid cell above: no surface, the column stays as on main
          if (bottom_only && phys_out && b >= n - 1 - il) continue;
          if (bottom_only) surf[l] = b + 1 - il;
          int glo = r ? runs[r - 1].second + 1 : 0;
          int ghi = r + 1 < nr ? runs[r + 1].first - 1 : n - 1;
          bool below = !bottom_only && a > 0 && (!two || (r + c) % 2 == 0);
          bool above = b + 1 < n && (!two || (r + c) % 2 == 1);
          for (int k = 0; k < reach; ++k) {
            int jb = a + k, ja = b - k;
            if (below && jb < n && (jb <= b) == (pass == 1) &&
                (two || !above || 2 * k < b - a + 1))
              put(jb, a, glo, a - 1);
            if (above && ja >= 0 && (ja >= a || bottom_only) == (pass == 1) &&
                (two || !below || 2 * k < b - a + 1))
              put(ja, b + 1, b + 1, ghi);
          }
        }
    }
    // a cell reads the copy in which the run above it is seen from below
    for (int j = 0, r = 0; two && j < n; ++j) {
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
  Mirror m;
  m.copies = nc;
  m.idx = back(idx, nc);
  m.odd = back(odd, nc);
  m.pick = two ? back(pick, 0) : torch::Tensor();
  if (bottom_only) {
    // the surface needs its nghost mirror cells and its reference rows on
    // this block: keep x1 block seams nghost cells clear of the terrain
    int iu = n - 1 - il, sm = surf.max().item<int>();
    TORCH_CHECK((phys_in || (surf.eq(0) | surf.ge(reach)).all().item<bool>()) &&
                    (phys_out || sm <= iu - il + 1 - reach),
                "a solid surface lies within ", reach,
                " cells of an x1 block seam; move the seam (nb1) at least ",
                reach, " cells above the highest terrain, or use one x1 block");
    m.surf = surf.view({shape[0], shape[1]}).to(solid.device());
  }
  return mirrors_[key] = m;
}

torch::Tensor InternalBoundaryImpl::mirror(torch::Tensor v, Mirror const &m,
                                           int copy, int dim,
                                           int normal) const {
  auto idx = m.idx[copy], odd = m.odd[copy];
  int lead = v.dim() - 3;
  for (int k = 0; k < lead; ++k) idx = idx.unsqueeze(0), odd = odd.unsqueeze(0);
  auto out = v.gather(lead + dim - 1, idx.expand_as(v));
  if (normal >= 0) out[normal] = torch::where(odd[0], -out[normal], out[normal]);
  return out;
}

torch::Tensor InternalBoundaryImpl::pick_lr(torch::Tensor lr0,
                                            torch::Tensor lr1, Mirror const &m,
                                            int dim) const {
  auto pr = m.pick.eq(1), pl = pr.roll(1, dim - 1);
  lr0[ILT] = torch::where(pl, lr1[ILT], lr0[ILT]);
  lr0[IRT] = torch::where(pr, lr1[IRT], lr0[IRT]);
  return lr0;
}

std::shared_ptr<InternalBoundaryImpl> InternalBoundaryImpl::create(
    InternalBoundaryOptions const &opts, torch::nn::Module *p,
    std::string const &name) {
  TORCH_CHECK(p != nullptr, "[InternalBoundary] Parent module is null");
  TORCH_CHECK(opts != nullptr, "[InternalBoundary] Options pointer is null");

  return p->register_module(name, InternalBoundary(opts, p));
}

}  // namespace snap
