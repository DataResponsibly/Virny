COMMIT_HASH := $(shell eval git rev-parse HEAD)

convert-notebooks:
	jupyter nbconvert --to markdown docs/examples/**.ipynb

doc:
	yamp virny --out docs/api --verbose
	mkdocs build

livedoc: doc
	mkdocs serve

test:
	python -m pytest tests

develop:
	python ./setup.py develop

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "lightning_logs" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	rm -rf coverage.xml
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf docs/_build
	rm -rf docs/_generated