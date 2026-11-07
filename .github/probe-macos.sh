#!/bin/bash
# where do the remaining ~36 s of a torchrun launch go on macOS (v2 fix in place)?
cd build/tests
EXE=$(find .. -name test_exchange.release -type f -perm -u+x | head -1); EXE=$(cd $(dirname $EXE) && pwd)/$(basename $EXE); echo "EXE=$EXE"
stamp() { python3 -u -c 'import sys,datetime
for l in sys.stdin: print(datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3], l, end="")'; }
cat > /tmp/w.sh <<'EOW'
#!/bin/bash
echo "WORKER rank=$RANK start $(python3 -c 'import time;print(time.time())')"
"$@"; rc=$?
echo "WORKER rank=$RANK end rc=$rc $(python3 -c 'import time;print(time.time())')"
exit $rc
EOW
chmod +x /tmp/w.sh
dumps() { for i in 1 2 3 4 5 6; do sleep 6
  for pid in $(pgrep -f "torchrun|torch.distributed.run"); do echo "--- py-spy agent $pid $(date -u +%T)"; sudo py-spy dump --pid $pid 2>&1 | grep -v "^ *$" | head -25; done
  for pid in $(pgrep -f test_exchange.release); do echo "--- py-spy native worker $pid $(date -u +%T)"; sudo py-spy dump --native --pid $pid 2>&1 | head -25; done
done; }
for args in "--local-addr=127.0.0.1" "--standalone --local-addr=127.0.0.1"; do
  echo "=== torchrun $args begin $(date -u +%T)"
  GLOO_SOCKET_IFNAME=lo0 torchrun $args --no-python --nproc-per-node=2 /tmp/w.sh $EXE 2>&1 | grep -v "^\[" | stamp &
  p=$!; dumps & d=$!
  wait $p; echo "=== torchrun $args end $(date -u +%T)"; wait $d
done
exit 0
