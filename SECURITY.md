# Security policy

## Supported versions

tbccl has not been released yet; 0.6.0 is the first planned release. Once releases exist, security fixes are made against the latest release series only. Development versions and unreleased branches are not supported.

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
