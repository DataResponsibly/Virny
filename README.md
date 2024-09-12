# Virny Software Library

<p align="center">
  <!-- Tests -->
  <a href="https://github.com/DataResponsibly/Virny/actions/workflows/ci.yml">
    <img src="https://github.com/DataResponsibly/Virny/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline">
  </a>
  <!-- Documentation -->
  <a href="https://dataresponsibly.github.io/Virny/">
    <img src="https://img.shields.io/website?label=docs&style=flat-square&url=https://dataresponsibly.github.io/Virny/" alt="documentation">
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/virny">
    <img src="https://img.shields.io/pypi/v/virny.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- Python Versions -->
  <a href="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"></a>
  <!-- Code Size -->
  <a href="">
    <img src="https://img.shields.io/github/languages/code-size/DataResponsibly/Virny.svg" alt="code_size">
  </a>
  <!-- Last Commit -->
  <a href="">
    <img src="https://img.shields.io/github/last-commit/DataResponsibly/Virny.svg" alt="last_commit">
  </a>
  <!-- License -->
  <a href="https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
  </a>
</p>


## 📜 Description

**Virny** is a Python library for in-depth profiling of model performance across overall and disparity dimensions. 
In addition to its metric computation capabilities, the library provides an interactive tool called _VirnyView_ 
to streamline responsible model selection and generate nutritional labels for ML models.

The Virny library was developed based on three fundamental principles: 

1) easy extensibility of model analysis capabilities;

2) compatibility to user-defined/custom datasets and model types;

3) simple composition of disparity metrics based on the context of use.

Virny decouples model auditing into several stages, including: **subgroup metric computation**, **disparity metric composition**,
and **metric visualization**. This gives data scientists more control and flexibility to use the library
for model development and monitoring post-deployment.

For quickstart, look at [use case examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/), [an interactive demo](https://huggingface.co/spaces/denys-herasymuk/virny-demo), and [a demonstrative Jupyter notebook](https://huggingface.co/spaces/denys-herasymuk/virny-demo/blob/main/notebooks/ACS_Income_Demo.ipynb).


## 🛠 Installation

Virny supports **Python 3.9-3.12** and can be installed with `pip`:

```bash
pip install virny
```


## 📒 Documentation

* [Introduction](https://dataresponsibly.github.io/Virny/)
* [API Reference](https://dataresponsibly.github.io/Virny/api/overview/)
* [Use Case Examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/)
* [Interactive Demo](https://huggingface.co/spaces/denys-herasymuk/virny-demo)


## 😎 Why Virny

In contrast to existing fairness software libraries and model card generating frameworks, our system stands out in four key aspects:

1. Virny facilitates the measurement of **all normatively important performance dimensions** (including _fairness_, _stability_, and _uncertainty_) for a set of initialized models, both overall and broken down by user-defined subgroups of interest.

2. Virny enables data scientists to analyze performance using **multiple sensitive attributes** (including _non-binary_) and their _intersections_.

3. Virny offers **diverse APIs for metric computation**, designed to analyze multiple models in a single execution, assessing stability and uncertainty on correct and incorrect predictions broken down by protected groups, and testing models on multiple test sets, including in-domain and out-of-domain.

4. Virny implements streamlined flow design tailored for **responsible model selection**, reducing the complexity associated with numerous model types, performance dimensions, and data-centric and model-centric interventions.


## 💡 List of Features

* Profiling of all normatively important performance dimensions: accuracy, stability, uncertainty, and fairness
* Ability to analyze non-binary sensitive attributes and their intersections
* Convenient metric computation interfaces: an interface for multiple models, an interface for multiple test sets, and an interface for saving results into a user-defined database
* Interactive _VirnyView_ visualizer that profiles dataset properties related to protected groups, computes comprehensive [nutritional labels](http://sites.computer.org/debull/A19sept/p13.pdf) for individual models, compares multiple models according to multiple metrics, and guides users through model selection
* Compatibility with [pre-, in-, and post-processors](https://aif360.readthedocs.io/en/latest/modules/algorithms.html#) for fairness enhancement from AIF360
* An `error_analysis` computation mode to analyze model stability and confidence for correct and incorrect prodictions broken down by groups
* Metric static and interactive visualizations
* Data loaders with subsampling for popular fair-ML benchmark datasets
* User-friendly parameters input via config yaml files 

Check out [our documentation](https://dataresponsibly.github.io/Virny/) for a comprehensive overview.


## 🤗 Affiliations

![NYU-UCU-Logos](https://user-images.githubusercontent.com/42843889/216840888-071bf184-f0e3-4a3e-94dc-c0d1c7784143.png)


## 💬 Citation

If Virny has been useful to you, and you would like to cite it in a scientific publication, please refer to the [paper](https://dl.acm.org/doi/abs/10.1145/3626246.3654738) published at SIGMOD:

```bibtex
@inproceedings{herasymuk2024responsible,
  title={Responsible Model Selection with Virny and VirnyView},
  author={Herasymuk, Denys and Arif Khan, Falaah and Stoyanovich, Julia},
  booktitle={Companion of the 2024 International Conference on Management of Data},
  pages={488--491},
  year={2024}
}
```


## 📝 License

**Virny** is free and open-source software licensed under the [3-clause BSD license](https://github.com/DataResponsibly/Virny/blob/main/LICENSE).
