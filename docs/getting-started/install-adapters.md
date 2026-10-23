# Installing an adapter with the helper

The adapter wheels (`torch-tbccl`, `vllm-tbccl`, `exo-tbccl`) are attached to GitHub Releases. `scripts/install_adapter.py` finds the right one for an environment you name, verifies it and installs it. It is optional: the manual commands in each adapter's installation guide keep working, and `--dry-run` prints them for your environment.

The helper uses only the Python standard library. It reads [`release/wheels.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/wheels.json), a generated inventory of the wheels on the current stable releases (CI compares it with each release's `SHA256SUMS` and with the downloaded files). Framework pins come from `release/ecosystem.json`.

## Use

```sh
curl -LO https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/scripts/install_adapter.py
python3 install_adapter.py create-env .venv --python-version 3.13          # an empty environment (needs uv)
uv pip install --python .venv/bin/python "torch==2.13.0" --torch-backend cu130   # you install the framework; macOS: no --torch-backend
python3 install_adapter.py install --adapter torch --python .venv/bin/python --device cuda
```

`--device` is `cpu`, `cuda` (Linux x86-64) or `mps`/`metal` (Apple silicon). It only checks that your PyTorch matches: the wheel is the same file for every device. Other commands: `list` shows every wheel in the inventory; `install --dry-run` shows what would happen without downloading anything; `--reinstall` replaces an installed adapter of the same version.

The helper uses the `uv` or `pip` you have. Use uv 0.13.4 or later and pip 26.2.2 or later: older versions trust the emSign root that was revoked in October 2026 (see the [security policy](../project/security-policy.md)). The wheels themselves contain no CA bundle.

| Adapter | Supported environments |
|---|---|
| `torch` | CPython 3.13, Linux x86-64 and macOS arm64, PyTorch 2.13.x |
| `vllm` | CPython 3.12 or 3.13, Linux x86-64 and macOS arm64, vLLM 0.31.0 (install vLLM first, as in the vllm-tbccl guide) |
| `exo` | not installed by this script: use `uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl` (macOS: `--extra mlx --extra tbccl`) in a checkout of `tbccl-integration-v2` |

## What it checks before changing anything

1. The environment's Python, platform and glibc/macOS version against the one matching wheel (it never guesses a filename).
2. The installed framework against the recorded pin. A missing or different PyTorch/vLLM stops the install; the framework is never installed, upgraded or removed.
3. On Linux: that the installed PyTorch is a CUDA build (the wheels are linked against `libtorch_cuda`, so a CPU-only PyTorch cannot import them; no GPU is needed). For `--device cuda` also that an NVIDIA driver is present.
4. That the environment is writable.
5. After downloading to a temporary file: the SHA-256 must equal both the inventory and the release's own `SHA256SUMS`, and the wheel's tags and version must match.

Then it installs with `--no-deps` (uv if available, otherwise pip) and runs the adapter's verification (`python -m torch_tbccl.info` or `vllm_tbccl.info`). Exit codes: 2 unsupported configuration, 3 network, 4 integrity (checksum or metadata), 5 installer failure. Every error names the problem and what is supported.

It does not self-update, does not use `sudo`, and sends nothing anywhere except the downloads above.
