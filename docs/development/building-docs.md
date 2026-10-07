# Building the documentation

The documentation is a Sphinx project using MyST Markdown and the Furo theme. It builds without network access to any hosting service.

```sh
make docs            # create .venv-docs, install the pinned requirements, build into docs/_build/html
make docs-linkcheck  # check local links and anchors (external URLs are not fetched)
```

Requirements: Python 3.11 or newer and Doxygen 1.9 or newer on `PATH` (the C and C++ API reference is generated from the public headers).

The pinned dependencies are in `docs/requirements.txt`. Warnings are treated as errors, so a build that prints a warning fails; the same command runs in continuous integration.
