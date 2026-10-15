```{include} ../../CONTRIBUTING.md
:parser: myst_parser.sphinx_
```

## Commit and pull request metadata

Do not put AI assistant session links, conversation identifiers or session trailers in commit messages, tags, pull request descriptions or release notes. Describing AI assistance in plain words is fine. Enable the local hook with `git config core.hooksPath .githooks`; the `session-metadata` workflow repeats the check on pull requests, on `main`, on release tags and on the CHANGELOG, and reports the category and line without printing the text. Existing history is not rewritten because the published releases are signed and attested.
