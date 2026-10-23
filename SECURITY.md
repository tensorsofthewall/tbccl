# Security policy

## Supported versions

Security fixes are made against the latest release series only (currently 0.7.x). Development versions and unreleased branches are not supported.

## Package-manager and trust-store notice (October 2026)

Private keys of eight eMudhra intermediate certificates were reported compromised in October 2026, and the emSign root was removed from the CA bundles that Python tooling uses. This is not a vulnerability of the TBCCL runtime or of any published TBCCL release: the data transport is plain TCP without TLS (see the security model), and none of the released archives or wheels contains a CA bundle or performs TLS. It concerns the tools that fetch packages and the environments that build them.

- Use at least certifi 2026.10.19, pip 26.2.2 and uv 0.13.4 (sigstore 4.5.1, conda 26.9.4 and pixi 0.82.1 if you use them), in every environment where you install TBCCL or its adapters. pip trusts its own vendored certifi in addition to the system store, so upgrading the `certifi` package does not patch an older pip: upgrade pip itself, with the hash-pinned wheel in the Python Security Response Team announcement when the installed pip is older.
- The release workflows install a specific pip (from its hash-pinned wheel) and uv, build inside a manylinux image whose tools carry these versions, and block the emSign roots in that image's system bundle before downloading anything. `tools/check_trust_dependencies.py` rejects affected certifi, uv and container pins and TLS-verification bypasses in active files.
- Nothing here adds authentication or encryption to TBCCL's data path.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub private vulnerability reporting: open the repository's Security tab and choose "Report a vulnerability". If that option is not available, open an issue that says only that you have a security report and asks for a private channel; do not put details in it.

## What to include

- The affected version or commit and the platform.
- What you observed and what you expected.
- The smallest reproduction you can give: code, input bytes or steps.
- Whether you believe it is exploitable, and under which deployment assumptions.
- Whether and how you want to be credited.

Please do not include credentials, private keys or details of a private network.

## What to expect

This is a volunteer-maintained project. Reports are acknowledged on a best-effort basis, normally within a few days. A confirmed issue is fixed on the main branch first and released in the next release of the supported series; the advisory credits the reporter unless they ask otherwise. Please allow a reasonable time to fix a confirmed issue before you disclose it publicly.

## Scope

This policy covers the TBCCL runtime: the C++ library, the C ABI, the TCP transport and the collective algorithms. Memory-safety or crash bugs reachable from malformed network input are in scope and are treated as bugs even though the runtime is not designed for hostile networks (see the TBCCL security model in the documentation). The absence of peer authentication, transport encryption or authorization is a documented property of the current design, not a vulnerability report; suggestions are welcome as ordinary issues.

The supported deployment assumptions are described in the security model in the documentation: TBCCL assumes trusted peers on a trusted network.
