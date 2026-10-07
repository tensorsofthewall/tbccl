DOCS_VENV ?= .venv-docs
PYTHON ?= python3

.PHONY: docs docs-linkcheck docs-clean

$(DOCS_VENV)/.installed: docs/requirements.txt
	$(PYTHON) -m venv $(DOCS_VENV)
	$(DOCS_VENV)/bin/python -m pip install --quiet -r docs/requirements.txt
	touch $@

docs: $(DOCS_VENV)/.installed
	$(DOCS_VENV)/bin/sphinx-build -W --keep-going -b html docs docs/_build/html

docs-linkcheck: $(DOCS_VENV)/.installed
	$(DOCS_VENV)/bin/sphinx-build -W --keep-going -b linkcheck -D linkcheck_ignore='^https?://' docs docs/_build/linkcheck

docs-clean:
	rm -rf docs/_build
