#!/bin/bash
# Driver for communicator_cuda_stream_chunk_test: starts the two ranks as separate processes on one GPU, each with its own TBCCL_STAGED_CHUNK_BYTES.
#   communicator_cuda_stream_chunk_test.sh <test-binary> <rank0 setting|-> <rank1 setting|-> [scenarios]
# "-" leaves the variable unset for that rank.
# Exit status: 0 both ranks passed, 1 a rank failed (the other is then terminated), 2 no usable port was found.
# Test seams (used by communicator_cuda_stream_chunk_driver_test.sh): TBCCL_STREAM_TEST_PORT_RANGE=<lo>:<hi> restricts the candidate base ports,
# TBCCL_STREAM_TEST_PORT_ATTEMPTS bounds the search, TBCCL_STREAM_TEST_PRINT_PORT=1 prints the chosen port and exits.
set -u
bin=${1:-}; s0=${2:-}; s1=${3:-}; scenarios=${4:-zero,producer,delayed,outstanding,war,repeated,collectives,abort}

# The Communicator listens on peers[r].port (control) and peers[r].port + 1000 (data), for both ranks: the endpoint set is p, p+1, p+1000, p+1001.
# Pick a base below the OS ephemeral range, where outgoing connections of other processes take their source ports, such that all four can be bound right now.
# This is a check, not a reservation: another process can still take one of them between this check and the ranks' own bind (time-of-check/time-of-use).
port=$(python3 - <<'PY'
import os, random, socket, subprocess, sys

def ephemeral_floor():
    try:
        return int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(subprocess.check_output(["sysctl", "-n", "net.inet.ip.portrange.first"], text=True).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 32768

lo, hi = 20000, ephemeral_floor() - 1002
rng = os.environ.get("TBCCL_STREAM_TEST_PORT_RANGE")
if rng:
    lo, hi = (int(x) for x in rng.split(":"))
attempts = int(os.environ.get("TBCCL_STREAM_TEST_PORT_ATTEMPTS", "500"))
if hi <= lo or lo < 1 or hi + 1001 > 65535:
    sys.exit("no usable base port range [%d, %d) below the ephemeral range" % (lo, hi))
for _ in range(attempts):
    p = random.randrange(lo, hi)
    socks = []
    try:
        for q in (p, p + 1, p + 1000, p + 1001):
            s = socket.socket()
            socks.append(s)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", q))
        print(p)
        sys.exit(0)
    except OSError:
        continue
    finally:
        for s in socks:
            s.close()
sys.exit("no free port set (p, p+1, p+1000, p+1001) found in [%d, %d) after %d attempts" % (lo, hi, attempts))
PY
) || { echo "communicator_cuda_stream_chunk_test.sh: could not select a port" >&2; exit 2; }
case $port in ''|*[!0-9]*) echo "communicator_cuda_stream_chunk_test.sh: invalid port '$port'" >&2; exit 2;; esac
if [ "${TBCCL_STREAM_TEST_PRINT_PORT:-}" = 1 ]; then echo "$port"; exit 0; fi
if [ -z "$bin" ] || [ -z "$s0" ] || [ -z "$s1" ]; then echo "usage: $0 <test-binary> <rank0 setting|-> <rank1 setting|-> [scenarios]" >&2; exit 2; fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/cuda_stream_chunk.XXXXXX") || exit 2
p0=; p1=
cleanup() {
  for p in $p0 $p1; do kill "$p" 2>/dev/null; done
  for p in $p0 $p1; do wait "$p" 2>/dev/null; done
  rm -rf "$tmp"
}
trap cleanup EXIT
trap 'exit 1' INT TERM

run() { # rank setting
  # exec: the background process id must be the rank itself, so that killing it really stops the rank
  if [ "$2" = "-" ]; then exec env -u TBCCL_STAGED_CHUNK_BYTES "$bin" --rank "$1" --port "$port" --scenarios "$scenarios";
  else exec env TBCCL_STAGED_CHUNK_BYTES="$2" "$bin" --rank "$1" --port "$port" --scenarios "$scenarios"; fi
}
run 0 "$s0" > "$tmp/r0" 2>&1 &
p0=$!
sleep 0.3
run 1 "$s1" > "$tmp/r1" 2>&1 &
p1=$!

# Wait for both ranks. As soon as one of them fails, stop waiting for the other one (cleanup terminates it): a rank cannot make progress without its peer, and
# a failed correctness scenario is never retried.
rc=0; d0=0; d1=0
while [ $d0 = 0 ] || [ $d1 = 0 ]; do
  if [ $d0 = 0 ] && ! kill -0 "$p0" 2>/dev/null; then wait "$p0" || rc=1; d0=1; fi
  if [ $d1 = 0 ] && ! kill -0 "$p1" 2>/dev/null; then wait "$p1" || rc=1; d1=1; fi
  [ $rc = 0 ] || break
  sleep 0.05
done
if [ $rc != 0 ]; then # stop the surviving rank before reading its output
  for p in $p0 $p1; do kill "$p" 2>/dev/null; done
  for p in $p0 $p1; do wait "$p" 2>/dev/null; done
  echo "communicator_cuda_stream_chunk_test.sh: a rank failed; the other rank was terminated" >&2
fi
echo "== rank 0 (chunk $s0)"; cat "$tmp/r0"
echo "== rank 1 (chunk $s1)"; cat "$tmp/r1"
exit $rc
