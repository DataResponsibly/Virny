COMMIT_HASH := $(shell eval git rev-parse HEAD)

convert-notebooks:
	jupyter nbconvert --to markdown docs/examples/**.ipynb

doc:
	yamp source --out docs/api --verbose
	mkdocs build

livedoc: doc
	mkdocs serve

develop:
	python ./setup.py develop
