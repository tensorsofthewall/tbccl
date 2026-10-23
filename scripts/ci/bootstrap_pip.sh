#!/usr/bin/env bash
# bootstrap_pip.sh [python]   Make sure the pip of the interpreter (default: python3 on the PATH) is at least 26.2.2.
# 26.2.2 is the first pip whose vendored CA bundle no longer trusts the emSign roots revoked in October 2026. pip always trusts its own vendored certifi in
# addition to the system store, so an older pip cannot be repaired by upgrading the certifi package or by pointing it at another bundle: it has to be replaced.
# The replacement is the hash-pinned wheel from the Python Security Response Team announcement of 2026-10-20; pip checks the sha256 fragment, so a download
# that was redirected through a forged certificate is rejected. TLS verification is never disabled. An interpreter whose pip is already new enough is left alone.
set -euo pipefail
PY=${1:-python3}
MIN=26.2.2
URL=https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl
SHA256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70

pip_ok() {
    "$PY" - "$MIN" <<'PY'
import re, subprocess, sys
want = tuple(int(x) for x in sys.argv[1].split("."))
try:
    out = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
except Exception:
    sys.exit(1)
have = tuple(int(x) for x in re.match(r"\d+(?:\.\d+)*", out).group(0).split("."))
sys.exit(0 if have >= want else 1)
PY
}

if pip_ok; then
    echo "pip is already >= $MIN: $("$PY" -m pip --version)"
    exit 0
fi
"$PY" -m pip install --disable-pip-version-check --quiet --no-deps "pip @ ${URL}#sha256=${SHA256}"
pip_ok || { echo "pip is still older than $MIN after the hash-pinned install" >&2; exit 1; }
echo "pip upgraded: $("$PY" -m pip --version)"
