# Maintenance policy

This is the maintenance policy of the TBCCL ecosystem: [tbccl](https://github.com/tensorsofthewall/tbccl), [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl), [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) and [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl), and the two framework forks they pin. It lives here, in the runtime repository, and the three adapters keep a short `MAINTENANCE.md` that links to it and records what is specific to them. The versioning rules are in [VERSIONING.md](https://tbccl.tensorsofthewall.com/en/latest/project/release-versioning.html); the release procedure is in [RELEASE.md](https://tbccl.tensorsofthewall.com/en/latest/development/release-process.html). The current stable versions and pins are in [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json), rendered as the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Supported release

The newest stable release of each project is the supported one. The ecosystem file names it; a release candidate is never supported for new deployments once its stable release exists, and a superseded candidate's GitHub Release description says so. The stable line is experimental software (0.x): the C ABI (1) and the wire protocol (4) are the compatibility contract, and nothing else is promised across releases.

## Branches

`main` is the only long-lived branch and is protected: changes arrive by pull request and are squash-merged. Releases are made from `main`. Short-lived branches (`maintenance/*`, `sync/*`, `fix/*`) are deleted after they merge, because the commits and the tags keep everything reproducible. A release branch is cut only if a supported line must receive fixes while `main` moves to an incompatible one; none exists today.

## Tags are immutable

A release tag (`vX.Y.Z`, `vX.Y.ZrcN`) and a framework integration tag are never moved, deleted or recreated, and a published artifact, checksum file, SBOM or attestation is never replaced. A broken release is superseded by a later one. A GitHub Release description may be edited to add a warning or navigation, because that changes neither the tag nor the artifacts. Documentation built from an old tag stays as it was and is historical by design (see below).

## Compatibility pinning

- A project's `compatibility.json` records what that release supports and how it was validated. The sources stay the single definition; `tools/check_compatibility_manifest.py` fails when the two drift.
- An adapter requires a TBCCL C ABI and wire protocol, and a minimum TBCCL version. It never reaches into another adapter.
- `release/ecosystem.json` in this repository is the one canonical statement of the current ecosystem (versions, tags, commits, framework pins). Adapters link to it for current status and keep only release-specific compatibility of their own. `tools/check_ecosystem_manifest.py` validates it, and `--online` confirms every tag resolves to the recorded commit.

Two kinds of authority are kept apart, in the code and in every diagnostic:

- **Release pin integrity** (release-specific): the adapter's `compatibility.json` plus the immutable framework repository, tag and SHA, the reference-patch provenance and the adapter's own documentation. A malformed manifest, a missing tag, a tag that resolves to another commit, a provenance mismatch or a document that disagrees with the manifest is an INTEGRITY FAILURE in every release state.
- **Current-ecosystem consistency** (current): the adapter against `release/ecosystem.json` on `main` of tbccl. For the version the ecosystem file publishes as current and the manifest marks `released`, a disagreement is an INTEGRITY FAILURE, because the public state contradicts itself. For an unreleased candidate (or an older release) it is a WARNING, "the current ecosystem manifest has not yet been updated for this candidate": the candidate's own immutable pin stays authoritative, so a new candidate that needs a new framework pin can be preflighted while `main` still correctly describes the old stable. An unreachable or unreadable ecosystem file is also a WARNING, never an integrity verdict. A coordinated release can compare against a candidate file or ref with `check_framework_pin.py --ecosystem-file <path>` or `--ecosystem-ref <git-ref>`.

## Framework fork pins

vllm-tbccl pins `tensorsofthewall/vllm-metal` at the immutable tag `v0.31.0-tbccl1`; exo-tbccl pins `tensorsofthewall/exo-fork` at `tbccl-integration-v2`. A stable release never pins a moving `main`. Each adapter has `tools/check_framework_pin.py`: a tag that resolves to a different commit than the recorded one is an integrity failure and is never "fixed" by updating the recorded SHA. Moving a pin means a new immutable tag, a compatibility validation and a new adapter release (see the matrix below). The `vllm-metal` reference patch kept in vllm-tbccl is provenance only: automation checks that the diff from the upstream base to the fork tag still equals it; users never apply it.

## Release cadence

There is no calendar. A release happens when a change needs one: a correctness or security fix, a supported framework line moving, or a feature that has been qualified. Documentation, CI and test changes land on `main` without a release.

## Security and correctness fixes

A correctness defect (wrong results, data corruption, a protocol disagreement) or a security issue is fixed in a new patch release of the affected project, with the defect described in the CHANGELOG and the earlier release's GitHub Release description edited to point to the fix. Report security issues as described in `SECURITY.md`. A fix that changes runtime behaviour needs physical requalification before it is released.

## Build environment and trust stores

The release builds run in a pinned manylinux image and on GitHub-hosted runners, and they fetch build tools over HTTPS. Their trust stores are part of the build, so they are pinned and checked like any other input:

- Every job that runs `pip` first runs `scripts/ci/bootstrap_pip.sh` (`tools/ci/bootstrap_pip.sh` in the adapters), which installs pip 26.2.2 from its hash-pinned wheel when the interpreter's pip is older. pip trusts its own vendored certifi in addition to the system store, so it can only be repaired by replacing pip. The same applies to the documentation build (`Makefile`, `.readthedocs.yaml`). uv is installed as a specific version (0.13.5), never as "latest".
- The manylinux image is pinned by digest. A digest is replaced when its pip, certifi or uv is affected, never refreshed by pulling the same digest again. The container build runs `distrust_emsign.sh` before it downloads anything, because the distribution's `ca-certificates` package can lag behind a root removal; that script only removes trust.
- `docs/requirements.txt` pins certifi exactly. `tools/check_trust_dependencies.py` (run by the `docs`, `maintenance` and `release-preflight` workflows) rejects an affected certifi, uv or image digest, a job that uses pip without the bootstrap, a `dnf` step without the blocklist and any TLS-verification bypass in an active file. Validation pages, the changelog and benchmark records are historical and are not read.
- Runner images and the Read the Docs base image are managed by their providers; the repository cannot change their system trust stores, only the tools it installs on top.

## Upstream update policy

vLLM, vllm-metal, exo, MLX and PyTorch are qualification boundaries: nothing moves them automatically. Upstream drift of the two forks is monitored weekly (`upstream-drift` workflows in the forks) and reported; it is never merged by automation. The next sync is a planned piece of work: merge upstream into the fork, re-apply and re-validate the integration, publish a new immutable tag, then release the adapter.

## What needs physical requalification

GitHub-hosted CI has no GPU, no Mac pair and no Thunderbolt link, so it cannot qualify behaviour that depends on them. The physical gates are a person's job and are recorded in the validation pages.

| Change | Local checks | Hosted CI | Physical Thunderbolt 4 |
|---|:-:|:-:|:-:|
| README or documentation only | yes | yes | no |
| Test only | yes | yes | no |
| CI or workflow only | yes | yes | no |
| Release metadata or GitHub Release edit | yes | yes | no |
| Maintenance automation | yes | yes | no |
| Framework pin change | yes | yes | yes |
| Adapter runtime change | yes | yes | yes |
| TBCCL transport or collective change | yes | yes | yes |
| Metal bridge, row mapping or scheduling change | yes | yes | yes |
| Fork integration runtime change | yes | yes | yes |
| Wire protocol or C ABI change | yes | yes | full qualification |

"Yes" means a targeted run chosen to exercise the change, from the public artifacts, with the evidence recorded; "full qualification" means the whole campaign of the previous release. If a change that was supposed to be documentation or CI touches a runtime source file, stop: it is a development change and gets its own plan.

## Maintenance automation

| What | Where | When | Reads | Writes |
|---|---|---|---|---|
| Release state, release documentation (the README, homepage and installation commands against the package version), tooling tests, internal links, framework pin (offline) | `docs` workflow | every pull request | the repository | nothing |
| Weekly health check (`maintenance`): release state, pins, stable release surface, links, documentation hosts, documentation build | `maintenance` workflow | weekly, and by hand | the repository, GitHub, the documentation hosts | job summary only |
| Release readiness (`release-preflight`), including the release documentation against the intended tag before it is tagged | `release-preflight` workflow | by hand, with the intended tag | the repository, the release-artifacts run | job summary only |
| Wrong-version guard | `guard` job of `release-artifacts` | every artifact build | the repository, the TBCCL core | nothing |
| Upstream drift | `upstream-drift` workflow in each fork | weekly, and by hand | the fork and its upstream | job summary only |

None of them publishes a package, creates a tag or a release, merges anything or changes a manifest, and they run with `contents: read` (preflight adds `actions: read`). The scheduled and manual jobs make network checks; pull-request checks are offline, so a GitHub or documentation outage never blocks a pull request.

Every check classifies what it finds, so a red run says what kind of problem it is:

| Category | Meaning | Fails the run |
|---|---|:-:|
| INTEGRITY FAILURE | an immutable pin or tag no longer resolves as recorded, or the repository state is inconsistent | yes |
| RELEASE SURFACE FAILURE | an expected stable release, artifact, SBOM or checksum disappeared or does not verify | yes |
| ERROR | the automation itself failed | yes |
| DRIFT | an upstream moved | no |
| WARNING | a website or API was temporarily unreachable; never a mismatch | no |

## Cadence

- Weekly, automated: manifest and release-state consistency, fork-pin integrity, upstream drift, documentation and link checks, stable release existence.
- Monthly, by a person: failures of scheduled workflows, the drift summaries, open compatibility issues, the monthly Dependabot pull requests (GitHub Actions and documentation tooling only; reviewed and merged by hand), and disk and worktree hygiene on the machines used for physical runs.
- Before any release: the `maintenance` workflow, the full repository CI, the `release-preflight` workflow, the physical qualification the matrix requires, and `tools/verify_public_release.py` afterwards.

Runtime dependencies and framework pins are deliberately outside automatic dependency updates.

## Historical documentation

Documentation built from a release tag (the `vX.Y.Z` and `stable` versions on Read the Docs) describes what that release supports and was validated against. It is frozen with the tag, so a statement in it about another project is true as of that release. For the current state of the ecosystem use the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html) on `latest`. We do not cut a release only to refresh cross-project statements in a frozen page.

## Reporting a problem

Use the issue templates. `tools/collect_diagnostics.py` prints a short environment report (versions, interface names and MTU, environment variable names, `tbccl-info`) that does not read prompts, files, tokens or the environment's values; review it before posting.

## Actions in workflows

Third-party actions in the release and maintenance workflows are pinned to commit SHAs with the version in a comment. The only exceptions are the workflows the framework forks inherit from their upstreams, which are not part of this policy.
