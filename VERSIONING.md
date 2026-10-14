# Versioning policy

All four projects use Semantic Versioning, with the caveat that 0.x software is experimental: a minor release may change behaviour that was documented as experimental, and the changelog says so. Release candidates are `X.Y.ZrcN`; a stable release is promoted from a candidate without a code change except the version.

Three numbers are different things and are never confused:

- the **project version** of each repository (this file);
- the **C ABI version** of the TBCCL C API (currently 1);
- the **wire protocol version** between ranks (currently 4). Ranks that speak different wire protocols cannot communicate.

## TBCCL

| Release | Meaning |
|---|---|
| Patch (0.6.x) | compatible bug fixes, documentation and maintenance; no ABI or wire protocol change; drop-in for the minor line |
| Minor (0.x.0) | backward-compatible functionality, including support for more platforms, world sizes or data types; may add to the C ABI without breaking version 1; a wire protocol bump is called out and requires every rank to upgrade |
| Major (x.0.0) | an incompatible change to the public API, the C ABI policy or the support policy |

## Adapters (torch-tbccl, vllm-tbccl, exo-tbccl)

| Release | Meaning |
|---|---|
| Patch | compatible bug fixes, documentation and maintenance within the same framework line and the same pins |
| Minor | new or changed support: a different supported major or minor line of the framework, a new immutable fork pin, a new TBCCL minimum, new hardware or topology |
| Major | incompatible change to what the adapter exposes |

Rules that follow:

- Changing the supported major or minor framework line (for example vLLM 0.31 to 0.32, PyTorch 2.13 to 2.14) normally needs an adapter minor release and physical requalification.
- Moving an immutable framework-fork pin needs a compatibility validation and is a minor release of the adapter; the old tag stays where it is.
- Stable releases never pin a moving `main` of a fork or of anything else.
- An adapter requires a TBCCL C ABI and wire protocol and a minimum TBCCL version; it works with any TBCCL satisfying those, and its wheels statically link the TBCCL they were built with.
- A defect that makes a published release unsafe to use is fixed by a patch release; the unsafe release is not edited, only its GitHub Release description gains a warning.

## What a release does not do

A release does not rewrite earlier tags, artifacts or documentation. The current state of the family is always [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json) on `main` of tbccl, shown on the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html). See [MAINTENANCE.md](https://tbccl.tensorsofthewall.com/en/latest/development/maintenance.html) for the maintenance policy.
