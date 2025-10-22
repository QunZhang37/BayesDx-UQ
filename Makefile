.PHONY: lint test fmt docs

lint:
	ruff check . || true

test:
	pytest -q

fmt:
	ruff check --fix . || true

docs:
	@echo "Build docs (LaTeX) in docs/paper-latex"
