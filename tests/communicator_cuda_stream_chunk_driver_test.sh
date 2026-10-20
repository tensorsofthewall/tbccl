#!/bin/bash
# Self-test of communicator_cuda_stream_chunk_test.sh (port selection, failure propagation, cleanup). It needs no GPU: the ranks are a fake program.
#   communicator_cuda_stream_chunk_driver_test.sh <path to communicator_cuda_stream_chunk_test.sh>
set -u
driver=${1:?usage: $0 <communicator_cuda_stream_chunk_test.sh>}
work=$(mktemp -d "${TMPDIR:-/tmp}/driver_selftest.XXXXXX") || exit 1
export TMPDIR=$work # the driver's temporary directory must be removed from here on every path
holder=
trap '[ -n "$holder" ] && kill "$holder" 2>/dev/null; pkill -f "$work/fake_rank" 2>/dev/null; rm -rf "$work"' EXIT
fails=0
check() { if [ "$2" = ok ]; then echo "PASS $1"; else echo "FAIL $1: $3"; fails=$((fails + 1)); fi; }

# fake rank: FAKE_MODE=pass | fail0 | fail1 | hang_both; records its pid
cat > "$work/fake_rank" <<'FAKE'
#!/bin/bash
rank=; while [ $# -gt 0 ]; do [ "$1" = --rank ] && rank=$2; shift; done
echo $$ > "$FAKE_PIDDIR/rank$rank.pid"
echo "fake rank $rank"
case "$FAKE_MODE" in
  pass) exit 0;;
  fail0) [ "$rank" = 0 ] && exit 3; exec sleep 60;;
  fail1) [ "$rank" = 1 ] && exit 3; exec sleep 60;;
esac
FAKE
chmod +x "$work/fake_rank"
export FAKE_PIDDIR=$work

# 1. selected ports: all four endpoints free at selection time and below the kernel's ephemeral range, over many selections
floor=32768; [ -r /proc/sys/net/ipv4/ip_local_port_range ] && floor=$(cut -f1 /proc/sys/net/ipv4/ip_local_port_range)
bad=
for i in $(seq 1 30); do
  p=$(TBCCL_STREAM_TEST_PRINT_PORT=1 "$driver") || { bad="selection $i exited $?"; break; }
  python3 - "$p" "$floor" <<'PY' || { bad="port $p: an endpoint was not bindable or the range was wrong"; break; }
import socket, sys
p, floor = int(sys.argv[1]), int(sys.argv[2])
assert 20000 <= p and p + 1001 < floor
for q in (p, p + 1, p + 1000, p + 1001):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", q)); s.close()
PY
done
[ -z "$bad" ] && check "30 selections: p, p+1, p+1000, p+1001 free, below the ephemeral range" ok || check "port selection" bad "$bad"

# 2. an occupied endpoint is rejected: a one-candidate range, each of the four endpoints occupied in turn
for off in 0 1 1000 1001; do
  python3 - $((31000 + off)) <<'PY' &
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1)
print("held", flush=True)
time.sleep(30)
PY
  holder=$!
  sleep 0.5
  out=$(TBCCL_STREAM_TEST_PORT_RANGE=31000:31001 TBCCL_STREAM_TEST_PORT_ATTEMPTS=5 TBCCL_STREAM_TEST_PRINT_PORT=1 "$driver" 2>&1); rc=$?
  kill "$holder" 2>/dev/null; wait "$holder" 2>/dev/null; holder=
  if [ $rc = 2 ] && echo "$out" | grep -q "no free port set"; then check "occupied endpoint p+$off is rejected (exit 2, message)" ok; else check "occupied endpoint p+$off" bad "rc=$rc out=$out"; fi
done

# 3. an empty or invalid range is rejected with exit 2
out=$(TBCCL_STREAM_TEST_PORT_RANGE=40000:40000 TBCCL_STREAM_TEST_PRINT_PORT=1 "$driver" 2>&1); rc=$?
[ $rc = 2 ] && check "empty range rejected (exit 2)" ok || check "empty range" bad "rc=$rc out=$out"

# 4. both ranks pass
FAKE_MODE=pass "$driver" "$work/fake_rank" - - zero > "$work/pass.out" 2>&1; rc=$?
if [ $rc = 0 ] && grep -q "fake rank 0" "$work/pass.out" && grep -q "fake rank 1" "$work/pass.out"; then check "both ranks pass: exit 0, both outputs captured" ok; else check "pass" bad "rc=$rc"; fi

# 5./6. one rank fails: the driver exits 1 promptly, the other rank is terminated, nothing is left behind
for mode in fail0 fail1; do
  rm -f "$work"/rank*.pid
  start=$(date +%s)
  FAKE_MODE=$mode "$driver" "$work/fake_rank" - - zero > "$work/$mode.out" 2>&1; rc=$?
  took=$(( $(date +%s) - start ))
  survivor=$([ $mode = fail0 ] && echo 1 || echo 0)
  sleep 0.3
  alive=no; kill -0 "$(cat "$work/rank$survivor.pid" 2>/dev/null)" 2>/dev/null && alive=yes
  if [ $rc = 1 ] && [ $took -lt 10 ] && [ $alive = no ] && grep -q "other rank was terminated" "$work/$mode.out"; then check "$mode: exit 1 in ${took}s, surviving rank terminated" ok; else check "$mode" bad "rc=$rc took=${took}s survivor_alive=$alive"; fi
done

# 7. no temporary directory is left behind by any run above
left=$(ls -d "$work"/cuda_stream_chunk.* 2>/dev/null | wc -l)
[ "$left" = 0 ] && check "no driver temporary directory left behind" ok || check "temporary files" bad "$left left"

[ $fails = 0 ] && echo "driver self-test passed" || { echo "driver self-test: $fails failure(s)"; exit 1; }
