import os
import pathlib
import pkg_resources

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path


NAME = 'virny'
DESCRIPTION = "Responsible AI Toolset in Python"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
# TODO: change
URL = "https://medium-multiply.readthedocs.io/"
EMAILS = "herasymuk@ucu.edu.ua and fa2161@nyu.edu"
AUTHORS = "Denys Herasymuk and Falaah Arif Khan"

# The directory containing this file
HERE = os.getcwd()

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

about: dict = {}
with open(os.path.join(HERE, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# TODO: clean requirements later
with pathlib.Path('requirements.txt').open() as requirements_txt:
    base_packages = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

# This call to setup() does all the work
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    author=AUTHORS,
    author_email=EMAILS,
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=base_packages,
    extras_require={
        "docs": base_packages
                + [
                    "dominate",
                    "flask",
                    "ipykernel",
                    "jupyter-client",
                    "mike",
                    "mkdocs",
                    "mkdocs-awesome-pages-plugin",
                    "mkdocs-material",
                    "nbconvert",
                    "python-slugify",
                    "spacy",
                    "watermark",
                ],
    },
)
