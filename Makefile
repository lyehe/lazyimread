.PHONY: help install test lint format type-check clean ci

PYTHON := python3
TOX := tox

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install the package and development dependencies"
	@echo "  test        - Run tests using pytest"
	@echo "  lint        - Run linting checks with Ruff"
	@echo "  format      - Auto-format code with Ruff"
	@echo "  type-check  - Run type checking"
	@echo "  clean       - Remove build artifacts and cache files"
	@echo "  ci          - Run all checks (test, lint, format, type-check)"

install:
	$(PYTHON) -m pip install -e .[dev]
	pre-commit install

test:
	$(PYTHON) -m pytest tests --asyncio-mode=auto

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

type-check:
	$(PYTHON) -m mypy .

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

ci:
	$(TOX) -e py310,py311,py312,lint-format-type

# Tox-specific targets
.PHONY: tox tox-test tox-lint tox-format

tox:
	$(TOX)

tox-test:
	$(TOX) -e py310,py311,py312

tox-lint:
	$(TOX) -e lint-format-type
