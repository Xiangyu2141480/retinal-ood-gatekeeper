.PHONY: test lint format

test:
	pytest

lint:
	ruff check .

format:
	black src scripts tests
	ruff check . --fix
