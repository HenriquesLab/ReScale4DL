.PHONY: help install lint format pytest mypy mypy-types docs download-structures

help:
	@echo "Available commands:"
	@echo "  install                    Install source code in environment"
	@echo "  install-editable           Install source code in environment in editable mode"
	@echo "  install-all                Install source code in environment and all dependencies"
	@echo "  install-editable-all       Install source code in environment and all dependencies in editable mode"
	@echo "  lint                       Run linters using pre-commit"
	@echo "  format                     Run formatters using pre-commit"
	@echo "  pytest                     Run tests using pytest"
	@echo "  mypy                       Run type-checking using mypy"
	@echo "  mypy-types                 Install missing types using mypy"
	@echo "  docs                       Generate documentation using pdoc"
	@echo "  package                    Builds python package"

install:
	pip install .

install-editable:
	pip install --editable .

install-all:
	pip install ".[dev, test]"

install-editable-all:
	pip install --editable ".[dev, test]"

lint:
	pre-commit run ruff --all-files

format:
	pre-commit run ruff-format --all-files

pytest:
	pytest --cov=rescale4dl --cov-report=term-missing tests

mypy:
	mypy --ignore-missing-imports src

mypy-types:
	mypy --install-types

docs:
	rm -rf docs
	pdoc src/rescale4dl -o docs

.DEFAULT_GOAL := help

package:
	python -m build