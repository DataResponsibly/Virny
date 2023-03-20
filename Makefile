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
