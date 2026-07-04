all: format pyright build pytest

check:
	uv run ruff check

build:
	uv build

pyright:
	PYRIGHT_PYTHON_CACHE_DIR=.pyright-python uv run pyright agent_from_scratch

format:
	uv run ruff format

pytest:
	uv run pytest -vv

.PHONY: all ruff format pyright build pytest
