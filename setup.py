import os
import pathlib
import pkg_resources

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path


NAME = 'virny'
DESCRIPTION = "Python library for in-depth profiling of model performance across overall and disparity dimensions"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/DataResponsibly/Virny"
EMAILS = "denis.gerasymuk799@gmail.com"
AUTHORS = "Denys Herasymuk"

# The directory containing this file
HERE = os.getcwd()

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

about: dict = {}
with open(os.path.join(HERE, NAME, "__version__.py")) as f:
    exec(f.read(), about)

with pathlib.Path('requirements.txt').open() as requirements:
    base_packages = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements)
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
    license="BSD-3",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=base_packages,
    extras_require={
        "test": base_packages
                + [
                    "pytest~=7.2.1",
                ],
        "dev": base_packages
                + [
                    "pytest~=7.2.1",
                    "aif360>=0.6.1",
                    "fairlearn>=0.9.0",
                    "xgboost>=1.7.2",
                    "python-dotenv>=1.0.0",
                    "pytorch-tabular>=1.1.0",
                    "pymongo==4.3.3",
                ],
        "docs": [
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "pandas",
                    "dominate",
                    "flask",
                    "ipykernel",
                    "jupyter-client",
                    "mike",
                    "mkdocs",
                    "mkdocs-awesome-pages-plugin",
                    "mkdocs-material",
                    "mkdocs-redirects",
                    "nbconvert",
                    "python-slugify",
                    "spacy",
                ],
    },
    python_requires='>=3.9',
)
