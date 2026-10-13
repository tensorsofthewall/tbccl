# Documentation versions

The hosted documentation is built by Read the Docs from the configuration in `.readthedocs.yaml`, with the pinned requirements in `docs/requirements.txt`, and warnings fail the build.

| Version | Built from | Meaning |
|---|---|---|
| `latest` | the `main` branch | Development documentation. It is not a release, carries a banner that says so, and may describe behavior that no release contains yet. |
| `stable` | the newest release tag | The documentation of the latest release. It does not exist until the first release is tagged; nothing unreleased is ever published as `stable`. |
| a version such as `v0.6.0rc1` | a release-candidate tag | The documentation of that pre-release, frozen to the tag. Numbered documentation for the final `0.6.0` exists only after that tag does. |

Until a final release exists, links between the four projects' sites point at `latest`; they switch to `stable` with the final releases. See the [ecosystem overview](../project/ecosystem.md).

## Building and checking locally

`make docs` builds the documentation, `make docs-linkcheck` checks the local links, and the compatibility manifest check runs in the same CI job. None of them needs the network after the requirements are installed.

Cross-project references (Intersphinx) are off by default so that a build never depends on another site. Set `DOCS_INTERSPHINX=1` to enable them against the other projects' published development inventories (`latest`); set `DOCS_INVENTORY_<PROJECT>` to a local `objects.inv` to use a local copy.
