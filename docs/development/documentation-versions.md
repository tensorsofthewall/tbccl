# Documentation versions

The hosted documentation is built by Read the Docs from the configuration in `.readthedocs.yaml`, with the pinned requirements in `docs/requirements.txt`, and warnings fail the build.

| Version | Built from | Meaning |
|---|---|---|
| `latest` | the `main` branch | Development documentation. It is not a release, carries a banner that says so, and may describe behavior that no release contains yet. |
| `stable` | the newest final release tag | The documentation of the latest final release, frozen to its tag. Nothing unreleased is ever published as `stable`. |
| a version such as `v0.6.0` or `v0.6.0rc1` | a release or release-candidate tag | The documentation of that release, frozen to the tag. |

## Release-specific and current information

Documentation built from a tag is historical by design: it records what that release supports, what was validated for it, its dependencies and its limitations, and it never changes. Anything it says about the other projects was true when it was built and is not updated when they release later. The current state of the whole family (versions, tags, framework pins) is on the [ecosystem page](../project/ecosystem.md) of the `latest` TBCCL documentation, built from `release/ecosystem.json`; use it, not a versioned page, to answer "what is current".


## Building and checking locally

`make docs` builds the documentation, `make docs-linkcheck` checks the local links, and the compatibility manifest check runs in the same CI job. None of them needs the network after the requirements are installed.

Cross-project references (Intersphinx) are off by default so that a build never depends on another site. Set `DOCS_INTERSPHINX=1` to enable them against the other projects' published development inventories (`latest`); set `DOCS_INVENTORY_<PROJECT>` to a local `objects.inv` to use a local copy.
