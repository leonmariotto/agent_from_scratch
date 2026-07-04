all: format pyright build pytest

check:
	uv run ruff check

build:
	uv build

pyright:
	PYRIGHT_PYTHON_CACHE_DIR=.pyright-python uv run pyright LLLM

format:
	uv run ruff format

pytest:
	uv run pytest -vv -m "not slow"

.PHONY: all ruff format pyright build pytest
