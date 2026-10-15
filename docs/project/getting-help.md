# Getting help and giving feedback

The TBCCL projects are experimental and maintained on a best-effort basis: there is no support contract and no response-time commitment. Issues are the one channel, and the forms below ask for what makes a report answerable. GitHub Discussions are not enabled.

| You have | Open | In |
|---|---|---|
| a bug, a crash or a wrong result | the **Bug report** form | the repository of the package you ran (tbccl for the runtime and the C/C++ API; torch-tbccl, vllm-tbccl, exo-tbccl for an adapter) |
| an installation that failed, or the install helper refusing something it should accept | the **Installation problem** form | tbccl (`scripts/install_adapter.py` and the release inventory) or the adapter's repository |
| hardware or a software combination that worked, or did not, and is not on the [validated configurations](validated-configurations.md) page | the **Tested configuration** form | tbccl |
| an error, a gap or an unclear sentence in the documentation | the **Documentation feedback** form | the repository whose documentation it is |
| a question | the **Documentation feedback** form, or a bug report if you think something is broken | tbccl |
| a security problem | do not open an issue: follow the [security policy](security-policy.md) | |

## Before you post

- Check the [ecosystem status](ecosystem.md) and the [limitations](limitations.md): the supported versions and platforms are narrow on purpose (for example, PyTorch 2.13.x with CPython 3.13, or vLLM 0.31.0 with CPython 3.12).
- Run `tools/collect_diagnostics.py` and read what it prints. It reports versions, interface names and MTU, and the names (never the values) of TBCCL environment variables. It does not read your files, prompts, models, tokens or the full environment, and it uploads nothing.
- For the install helper, add `--dry-run` output: it shows the wheel, the checks and the manual commands without changing anything.
- Remove anything private before posting: credentials, user names and home paths, private network details and private model names.

## Hardware reports are useful

Almost everything here was qualified on one Linux CUDA host and one Apple-silicon Mac joined by a Thunderbolt 4 cable. A report that another combination works (or exactly how it fails) is the most useful feedback the project can get. The **Tested configuration** form asks for the versions, the machines, the link and what you ran, and nothing more. A report does not change the supported matrix by itself; maintainers decide what is added to the validated configurations, and only after reproducing it or reading evidence they can check.

## What not to expect

Replies about unsupported versions or platforms (Windows, Linux on Arm, Intel Macs, other framework versions), about RDMA (the transport is TCP), or about model-quality issues that come from a framework rather than the transport will usually point you to the limitations page. Performance questions are answered with the [benchmark methodology](../development/benchmark-methodology.md): there are no published figures to compare against.
