#!/bin/bash
# CPU Release, Gloo-only (UCX off) per README step 4/5; install to a private --target dir
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
source /data00/ai_workspace/worker4/pr357/venv-base/bin/activate
S=/data00/ai_workspace/worker1/mainsplit/snapy; I=/data00/ai_workspace/worker1/mainsplit/pyinst; L=/data00/ai_workspace/worker1/mainsplit
cd $S || exit 1; T0=$(date +%s)
echo "configure: cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DCUDA=OFF -DUCX=OFF -DBUILD_TESTS=OFF"
[ -f build/CMakeCache.txt ] || nice -n 19 cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DCUDA=OFF -DUCX=OFF -DBUILD_TESTS=OFF > $L/config.log 2>&1 || { echo CONFIG FAILED; exit 1; }
nice -n 19 cmake --build build --parallel 2 > $L/make.log 2>&1 || { echo "BUILD FAILED $(( $(date +%s)-T0 ))s"; exit 1; }
echo "build ok $(( $(date +%s)-T0 )) s"
nice -n 19 pip install --no-build-isolation --no-deps --no-cache-dir --target "$I" --upgrade . > $L/pip.log 2>&1 || { echo PIP FAILED; exit 1; }
echo "pip ok $(( $(date +%s)-T0 )) s"
