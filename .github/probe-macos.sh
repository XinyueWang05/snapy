#!/bin/bash
# times test_exchange under torchrun on macOS: where do the 300 s go?
set -x
cd build/tests
mkdir -p /tmp/probe
stamp() { python3 -u -c 'import sys,datetime
for l in sys.stdin: print(datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3], l, end="")'; }
cat > /tmp/probe/w.sh <<'EOW'
#!/bin/bash
echo "WORKER rank=$RANK start $(date -u +%T) $(python3 -c 'import time;print(time.time())')"
"$@"; rc=$?
echo "WORKER rank=$RANK end rc=$rc $(date -u +%T) $(python3 -c 'import time;print(time.time())')"
exit $rc
EOW
chmod +x /tmp/probe/w.sh
EXE=$PWD/../bin/test_exchange.release
hostname; hostname -f || true; python3 -c 'import socket,time;t=time.time();print("getfqdn",socket.getfqdn(),time.time()-t)'
scutil --get LocalHostName || true; ifconfig lo0 | head -3

run() {  # name, then torchrun options
  local name=$1; shift
  echo "=== $name begin $(date -u +%T)"
  GLOO_SOCKET_IFNAME=lo0 torchrun --log-dir /tmp/probe/$name "$@" --no-python --nproc-per-node=2 /tmp/probe/w.sh $EXE 2>&1 | stamp &
  local p=$!
  sleep 45
  for pid in $(pgrep -f "torch.distributed.run|torchrun"); do
    echo "--- py-spy dump pid $pid at $(date -u +%T)"; sudo py-spy dump --pid $pid 2>&1 | head -60
  done
  for pid in $(pgrep -f test_exchange.release); do
    echo "--- py-spy native dump worker $pid at $(date -u +%T)"; sudo py-spy dump --native --pid $pid 2>&1 | head -40
  done
  wait $p
  echo "=== $name end $(date -u +%T)"
}
run A_ctest_launch --local-addr=127.0.0.1
run B_standalone --standalone --local-addr=127.0.0.1

echo "=== C_no_torchrun begin $(date -u +%T)"
for r in 0 1; do
  RANK=$r LOCAL_RANK=$r WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 GLOO_SOCKET_IFNAME=lo0 /tmp/probe/w.sh $EXE > /tmp/probe/C_rank$r.log 2>&1 &
done
wait
tail -5 /tmp/probe/C_rank0.log /tmp/probe/C_rank1.log
echo "=== C_no_torchrun end $(date -u +%T)"
exit 0
