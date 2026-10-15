# Privacy and telemetry

**The TBCCL projects collect no telemetry.** Nothing in the runtime, the three adapter packages, the install helper or the diagnostics tool reports usage, errors, versions, hardware or anything else to the maintainers or to any third party, and there is no analytics service, account, registration or licence check.

## What the software does with the network

| Component | Network use |
|---|---|
| libtbccl (the runtime, inside every adapter wheel) | TCP connections to the peer endpoints you configure, and to nothing else. No name lookups of external services, no HTTP code. |
| `torch-tbccl`, `vllm-tbccl`, `exo-tbccl` (Python packages) | None of their own. They import no HTTP client library. The framework they plug into keeps its own network behaviour, below. |
| `scripts/install_adapter.py` (optional) | Downloads the release inventory, `SHA256SUMS` and one wheel from GitHub (`github.com`, HTTPS), and runs `pip`/`uv` to install it. `--dry-run` downloads nothing. It sends no identifying data beyond what any HTTPS download reveals to GitHub. |
| `tools/collect_diagnostics.py` | None. It prints to standard output or a file you choose; you decide whether to post it. |
| Maintenance and validation tools in the repositories (not part of any wheel or archive) | Talk to GitHub's API or to a local server you started (for example a vLLM server on your own machine), for release checks and tests. |

## What is outside our control

The frameworks have their own settings and defaults, which are independent of TBCCL and unchanged by it: PyTorch, vLLM (including its usage-statistics option), exo, MLX and Hugging Face downloads all make their own choices about network access. Check their documentation if that matters to you; using TBCCL does not turn any of it on or off. Package installers and GitHub also log ordinary download requests.

## How this was checked

On 2026-10-15 the shipped Python sources of the three adapters were searched for HTTP, socket and analytics libraries and for telemetry or analytics names (none found), and the native sources of the runtime and the adapters for HTTP URLs and client libraries (none found). The repository tools that use `urllib` were listed and read: they contact GitHub or a local server only. This is a point-in-time check, not a guarantee about future releases; a change that adds any outbound reporting would have to be a documented, opt-in, reviewed change, and this page would change with it.

If you find something that contradicts this page, report it as a bug.
