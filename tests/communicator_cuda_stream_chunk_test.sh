#!/bin/bash
# Driver for communicator_cuda_stream_chunk_test: starts the two ranks as separate processes on one GPU, each with its own TBCCL_STAGED_CHUNK_BYTES.
#   communicator_cuda_stream_chunk_test.sh <test-binary> <rank0 setting|-> <rank1 setting|-> [scenarios]
# "-" leaves the variable unset for that rank.
set -u
bin=$1; s0=$2; s1=$3; scenarios=${4:-zero,producer,delayed,outstanding,war,repeated,collectives,abort}
# The Communicator listens on peers[r].port (control) and peers[r].port + 1000 (data). Pick a base below the OS ephemeral range, where outgoing connections of other
# processes take their source ports, such that the control and data ports of both ranks (p, p+1, p+1000, p+1001) are free right now.
port=$(python3 - <<'PY'
import random, socket
try:
    floor = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
except OSError:
    floor = 32768
while True:
    p = random.randrange(20000, max(20001, floor - 1002))
    socks = []
    try:
        for q in (p, p + 1, p + 1000, p + 1001):
            s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", q)); socks.append(s)
        print(p); break
    except OSError:
        continue
    finally:
        for s in socks: s.close()
PY
)
run() { # rank setting
  if [ "$2" = "-" ]; then env -u TBCCL_STAGED_CHUNK_BYTES "$bin" --rank "$1" --port "$port" --scenarios "$scenarios";
  else env TBCCL_STAGED_CHUNK_BYTES="$2" "$bin" --rank "$1" --port "$port" --scenarios "$scenarios"; fi
}
run 0 "$s0" > "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r0" 2>&1 &
p0=$!
sleep 0.3
run 1 "$s1" > "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r1" 2>&1 &
p1=$!
rc=0
wait $p0 || rc=1
wait $p1 || rc=1
echo "== rank 0 (chunk $s0)"; cat "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r0"
echo "== rank 1 (chunk $s1)"; cat "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r1"
rm -f "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r0" "${TMPDIR:-/tmp}/cuda_stream_chunk_$$.r1"
exit $rc
