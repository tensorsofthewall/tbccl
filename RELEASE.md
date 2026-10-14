# Release checklist

This is the maintainer checklist for releasing tbccl. It contains no credentials and no machine-specific paths. The automation does the repetitive checks; the physical qualification is still a person's job. GitHub CI does not replace physical qualification.

First decide what kind of change this is: the table in [the maintenance policy](https://tbccl.tensorsofthewall.com/en/latest/development/maintenance.html#what-needs-physical-requalification) says which changes need none, a targeted run, or the full physical qualification. A documentation, test, CI or release-metadata change needs no physical run and normally needs no release at all.

## Before tagging

1. The CHANGELOG describes what changed for users in the version, with no development chronology.
2. Set the version fields in `compatibility.json` (keep `released` false until the release is published) and run `python3 tools/check_release_state.py --tag vX.Y.Z`. It checks the version against the manifest, the sources, the CHANGELOG, the ABI and wire protocol and refuses a stable tag on a release-candidate version.
3. Run the `maintenance` workflow by hand on `main`: no INTEGRITY FAILURE or RELEASE SURFACE FAILURE.
4. Run the full test matrix on every platform the manifest lists, and the documented validation for the release; record the results in the validation page for the version.
5. Physical qualification: the Thunderbolt 4 and loopback qualification of the transport and collectives that the validation page lists (a change to the transport, a collective, the wire protocol or the C ABI needs it). Record the commands, the commit and the artifact checksums in the validation page.
6. Run the secret scan (gitleaks, pinned version) over the working tree and the history; check the security policy, license and third-party documentation.

## Build, inspect and preflight

7. Run the `release-artifacts` workflow on the release commit, passing the intended tag as `release_tag` where the workflow has that input. The guard job refuses to build when the tag, the package version, the manifest, the framework pin or the TBCCL line disagree. Only this workflow builds artifacts.
8. Run the `release-preflight` workflow with the intended tag. It must end with `READY`: version and manifest consistency, the pins, the tooling tests, the links, the documentation build, a free tag name, and a successful `release-artifacts` run for exactly this commit whose artifacts have the expected names, SBOMs and checksums. It publishes nothing.
9. Public-artifact smoke: install the built artifacts in a clean environment (never an editable install, never a path inside a working tree) and run the smoke check from the installation documentation.

## Tag and publish

10. Create a signed, annotated tag `vX.Y.Z` on the commit that `release-artifacts` built (see below). Release tags are immutable.
11. Create the GitHub release from the tag (a pre-release for a release candidate) with the changelog section, the artifacts, `SHA256SUMS` and the SBOMs. The release body links the installation guide, the compatibility page, the known limitations, the validation page and the current ecosystem page.
12. Publish to a package registry only after the artifacts have been inspected: none is used at present.
13. In a follow-up change on `main`, set `released` to true and update `release/ecosystem.json` and its table.

## After publishing

14. `python3 tools/verify_public_release.py --download` verifies the release from the outside: the release and tag exist, the artifacts match the manifest's `release_assets`, every file matches `SHA256SUMS`, the SBOMs are attached and the attestations verify. The scheduled `maintenance` workflow repeats the metadata part every week.
15. Run the physical smoke from the public artifacts again where the checklist above required physical qualification.
16. Confirm the published documentation matches the tag. Documentation built from a tag is historical by design; the current ecosystem status lives in the TBCCL documentation (`docs/project/ecosystem.md`, from `release/ecosystem.json`).

## Release artifacts

tbccl releases a source archive (and build artifacts only if a binary distribution is adopted). Every artifact is built by CI from the tagged commit, never from a maintainer's machine, and is accompanied by a SHA-256 checksum file and a software bill of materials.

## Software bill of materials

- Format: SPDX 2.3 JSON for each artifact.
- Tool: a CI step run on the tagged commit generates it (the Syft command line is the intended tool); the exact version is pinned in the workflow.
- Attached to: the GitHub release next to the artifact it describes, and verified after publishing.
- Scope: the contents of the artifact and its declared dependencies. For artifacts that statically link TBCCL, the SBOM lists the linked TBCCL version.

## Provenance

- Build provenance: GitHub artifact attestations created by the release workflow with the repository's built-in OIDC identity. No long-lived credential is required.
- Verification: `gh attestation verify <artifact> --repo <owner>/tbccl`.
- Checksums: a `SHA256SUMS` file is attached to each release. Signing the checksum file is optional until a maintainer signing key is set up.
- The `release-artifacts` workflow builds the artifacts, writes the SBOMs and `SHA256SUMS`, and attests them; it is the only place artifacts are built.

## Release tag policy

- Tags that match `v*` are immutable. A released tag, for example `v0.6.0`, is never moved, deleted or re-pointed; the repository ruleset blocks it.
- A broken release is fixed by a later patch release, never by changing the existing tag or its artifacts.
- Release tags are signed annotated tags (`git tag -s`) created by a maintainer whose signing key is registered with GitHub. Commit signing is a separate matter and is not required to release.
- Pre-release tags use the same `v*` form with a SemVer pre-release suffix and follow the same immutability rule.
