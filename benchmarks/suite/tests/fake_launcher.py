"""Stands in for ssh in tests: `fake_launcher.py HOST COMMAND` runs COMMAND through a POSIX shell, like a remote login would. FAKE_LAUNCHER_FAIL_CHECK=1 makes
a command containing 'check-clean' fail, to simulate a host that cannot be asked a second time."""
import os
import sys

if os.environ.get("FAKE_LAUNCHER_FAIL_CHECK") and "check-clean" in sys.argv[2]:
    sys.stderr.write("fake launcher: connection refused\n")
    sys.exit(255)
os.execvp("sh", ["sh", "-c", sys.argv[2]])
