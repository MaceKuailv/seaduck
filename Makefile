PROJECT := seaduck
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests type-check

qa:
	pre-commit run --all-files

unit-tests:
	python -m pytest -vv --cov=seaduck --cov-report=$(COV_REPORT) --doctest-glob="*.md" --doctest-glob="*.rst" -W ignore::RuntimeWarning

type-check:
	python -m mypy .

conda-env-update:
	$(CONDA) env update $(CONDAFLAGS) -f ci/environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f environment.yml

# Regenerate the API-reference Markdown from the package docstrings.
# Output goes to docs/api/ and is git-ignored (see docs/generate_api.py).
docs-api:
	python docs/generate_api.py

# Build the static HTML site with mystmd (Jupyter Book 2).
# --execute runs the notebooks so their figures appear in the output.
docs-build: docs-api
	cd docs && npx myst build --html --execute

# Live-reloading preview server at http://localhost:3000
docs-serve: docs-api
	cd docs && npx myst start --execute

link-check: docs-api
	cd docs && npx myst build --html --check-links

joss:
	pandoc paper/paper.md --bibliography paper/paper.bib -o paper/paper_local.pdf
