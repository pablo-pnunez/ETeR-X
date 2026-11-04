# Usage: make env | make test | make prune-dry
# Raw datasets are read from $ETERX_DATA_DIR (see src/config.py), e.g.
#   export ETERX_DATA_DIR=/path/to/datasets
.PHONY: env test prune-dry

env:
	conda env create -f environment.yml

test:
	python -m pytest -q

prune-dry:
	python scripts/prune_runs.py
