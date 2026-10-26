#!/bin/bash
# CPU-only Release build of a snapy checkout + pip --target install into <srcdir>/pyinst
# (the runs put <srcdir>/pyinst first on PYTHONPATH via SNAPY_PY).
# usage: build.sh <snapy srcdir> <jobs>     (run with the python env that has torch, kintera, pyharp)
set -e
cd "$1"
T0=$(date +%s)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA=OFF -DNETCDF=ON > build-config.log 2>&1
cmake --build build --parallel $2 > build-make.log 2>&1
T1=$(date +%s); echo "$1 cmake build done $((T1-T0)) s"
pip install --no-build-isolation --no-deps --target "$PWD/pyinst" . > pip-target.log 2>&1
echo "$1 pip done $(( $(date +%s)-T1 )) s"
