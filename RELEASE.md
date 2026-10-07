# Release checklist

This is the maintainer checklist for releasing tbccl. It contains no credentials and no machine-specific paths. Nothing in this file is performed by contributors, and no release of tbccl has been made yet.

## Before tagging

1. Confirm the target version (0.6.0 for the first planned release) and that `CHANGELOG.md` describes what changed for users in that version, with no development chronology.
2. Update `compatibility.json`: set the version fields for the release, keep `released` false until the release is published, and run `python3 scripts/check_compatibility_manifest.py`.
3. Run the full test matrix on every platform the compatibility manifest lists, and the documented validation for the release. Record the results in the validation page for the version.
4. Run the sanitizer builds that the testing documentation lists, where the code base has native code.
5. Build the documentation (`make docs` and `make docs-linkcheck`) with zero warnings, and confirm the `docs` check is green on the release commit.
6. Run the secret scan (gitleaks, pinned version) over the working tree and the history: zero unresolved findings.
7. Check that the security policy, license and third-party documentation are current for the dependencies of this release.

## Build and inspect artifacts

8. Build and install from a clean checkout of the release commit (`cmake --install`), then build an external consumer against the installed prefix (`find_package(TBCCL)`) to check the exported targets and headers.
9. Generate a software bill of materials for each artifact (see below) and compute SHA-256 checksums of every artifact.
10. Attach provenance (see below).

## Tag and publish

11. Create a signed, annotated tag `v0.6.0` on the release commit on `main` (see below). Release tags are immutable.
12. Create the GitHub release from the tag with the changelog section, the artifacts, the checksums and the SBOMs.
13. Publish to the package registry only after the artifacts have been inspected: not applicable to the core library until a binary distribution is adopted.
14. Set `released` to true in `compatibility.json` in a follow-up change on `main`.

## After publishing

15. Install from the published artifacts in a clean environment and run the smoke check from the installation documentation.
16. Verify the checksums, the provenance attestation and the SBOM of what was published.
17. Confirm the published documentation matches the tag.

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
- The release workflow is added when the first release is prepared; it is not part of the current repository.

## Release tag policy

- Tags that match `v*` are immutable. A released tag, for example `v0.6.0`, is never moved, deleted or re-pointed; the repository ruleset blocks it.
- A broken release is fixed by a later patch release, never by changing the existing tag or its artifacts.
- Release tags are signed annotated tags (`git tag -s`) created by a maintainer whose signing key is registered with GitHub. Commit signing is a separate matter and is not required to release.
- Pre-release tags use the same `v*` form with a SemVer pre-release suffix and follow the same immutability rule.
