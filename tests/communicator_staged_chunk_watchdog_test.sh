#!/bin/sh
# Checks that the progress watchdog of communicator_staged_chunk_test reports a stall: the test's "watchdog-selftest" section blocks on purpose, the watchdog (2 s here) must print the
# last marker and end the process with status 124. A watchdog that stayed silent, or that let the process exit with another status, fails this test.
#   communicator_staged_chunk_watchdog_test.sh <path to communicator_staged_chunk_test>
bin=${1:?usage: $0 <communicator_staged_chunk_test>}
out=$(TBCCL_TEST_WATCHDOG_S=2 "$bin" --section watchdog-selftest 2>&1); rc=$?
echo "$out" | tail -n 25
[ "$rc" -eq 124 ] || { echo "FAIL: expected exit status 124, got $rc"; exit 1; }
echo "$out" | grep -q "WATCHDOG: no completion after 2 s; last marker: STAGED_CHUNK_TEST SECTION watchdog-selftest blocking" || { echo "FAIL: the last marker was not reported"; exit 1; }
if [ "$(uname -s)" = Linux ]; then
  echo "$out" | grep -q "WATCHDOG threads" || { echo "FAIL: no thread dump"; exit 1; }
  echo "$out" | grep -q "backtrace tid=" || { echo "FAIL: no thread backtrace"; exit 1; }
fi
echo "watchdog reported the stall and ended the process with status 124"
