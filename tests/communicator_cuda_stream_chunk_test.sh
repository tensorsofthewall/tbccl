#!/bin/bash
# Driver for communicator_cuda_stream_chunk_test: starts the two ranks as separate processes on one GPU, each with its own TBCCL_STAGED_CHUNK_BYTES.
#   communicator_cuda_stream_chunk_test.sh <test-binary> <rank0 setting|-> <rank1 setting|-> [scenarios]
# "-" leaves the variable unset for that rank.
set -u
bin=$1; s0=$2; s1=$3; scenarios=${4:-zero,producer,delayed,outstanding,war,repeated,collectives,abort}
port=$((30000 + ($$ % 20000)))
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
