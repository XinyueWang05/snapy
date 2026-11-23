"""apply aneos diagnostics in the checked-out tree (prints only; no behaviour change)"""
import re, sys
p = "src/eos/aneos.cpp"; s = open(p).read()
s = s.replace("#include <snap/snap.h>\n", "#include <snap/snap.h>\n#include <cstdio>\n", 1)
s = s.replace("""  pthermo = ANEOSThermoImpl::create(options->eos_file(), this);
}""", """  std::fprintf(stderr, "[diag] ANEOSImpl::reset this=%p before create pthermo.is_empty=%d\\n", (void*)this, (int)pthermo.is_empty());
  pthermo = ANEOSThermoImpl::create(options->eos_file(), this);
  std::fprintf(stderr, "[diag] ANEOSImpl::reset this=%p after create pthermo.is_empty=%d impl=%p\\n", (void*)this, (int)pthermo.is_empty(), pthermo.is_empty() ? nullptr : (void*)pthermo.get());
}""", 1)
assert "[diag]" in s; open(p, "w").write(s)
p = "src/eos/aneos/aneos_thermo_dummy.cpp"; s = open(p).read()
s = s.replace("""  return p->register_module("thermo", ANEOSThermo(filename));""", """  std::fprintf(stderr, "[diag] ANEOSThermoImpl::create (library) p=%p\\n", (void*)p);
  auto h = ANEOSThermo(filename);
  std::fprintf(stderr, "[diag] create: holder empty=%d\\n", (int)h.is_empty());
  auto r = p->register_module("thermo", h);
  std::fprintf(stderr, "[diag] create: registered ptr=%p\\n", (void*)r.get());
  return r;""", 1)
s = s.replace("#include \"aneos_thermo.hpp\"\n", "#include \"aneos_thermo.hpp\"\n#include <cstdio>\n", 1)
assert "[diag]" in s; open(p, "w").write(s)
p = "tests/test_aneos_gamma.cpp"; s = open(p).read()
s = s.replace("ANEOSThermoImpl::ANEOSThermoImpl(const std::string& fname) : filename(fname) {}",
 "ANEOSThermoImpl::ANEOSThermoImpl(const std::string& fname) : filename(fname) {\n  std::fprintf(stderr, \"[diag] test stand-in ctor this=%p\\n\", (void*)this);\n}", 1)
s = s.replace("""  auto du = torch::zeros_like(w);
""", """  auto du = torch::zeros_like(w);
  {
    auto eos = hydro->peos;
    std::fprintf(stderr, "[diag] peos=%p typeid=%s\\n", (void*)eos.get(), typeid(*eos).name());
    auto an = std::dynamic_pointer_cast<ANEOSImpl>(eos);
    std::fprintf(stderr, "[diag] dynamic_pointer_cast<ANEOSImpl> = %p\\n", (void*)an.get());
    if (an) {
      std::fprintf(stderr, "[diag] pthermo.is_empty=%d\\n", (int)an->pthermo.is_empty());
      for (auto const& c : an->named_children()) std::fprintf(stderr, "[diag] child %s %p\\n", c.key().c_str(), (void*)c.value().get());
    }
    try { auto c = eos->sound_speed(w); std::fprintf(stderr, "[diag] direct sound_speed ok\\n"); }
    catch (std::exception const& e) { std::fprintf(stderr, "[diag] direct sound_speed threw: %.200s\\n", e.what()); }
  }
""", 1)
if "sound_speed" not in open("src/eos/equation_of_state.hpp").read():
    s = s.replace("eos->sound_speed(w)", "eos->compute(\"W->L\", {w})")
s = s.replace("#include <snap/hydro/hydro.hpp>\n", "#include <snap/hydro/hydro.hpp>\n#include <snap/eos/aneos.hpp>\n#include <typeinfo>\n", 1)
assert s.count("[diag]") >= 4; open(p, "w").write(s)
print("diag applied")
