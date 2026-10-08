.PHONY: install lint format typecheck test check run

install:
	uv sync --all-extras --dev
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src scripts

test:
	uv run pytest --cov=src --cov-report=term-missing

check: lint typecheck test

run:
	uv run python -m src.credit_risk.main
