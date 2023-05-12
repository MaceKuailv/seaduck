PROJECT := seaduck
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests type-check

qa:
	pre-commit run --all-files

unit-tests:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	python -m mypy .

conda-env-update:
	$(CONDA) env update $(CONDAFLAGS) -f ci/environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f environment.yml


docs-build:
	jupyter-book build docs/

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
