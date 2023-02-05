# Virny Software Library

<p align="left">
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
  <!-- License -->
  <a href="https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
  </a>
  <!-- Code Size -->
  <a href="">
    <img src="https://img.shields.io/github/languages/code-size/DataResponsibly/Virny.svg" alt="code_size">
  </a>
  <!-- Last Commit -->
  <a href="">
    <img src="https://img.shields.io/github/last-commit/DataResponsibly/Virny.svg" alt="last_commit">
  </a>
</p>


## Description 📜

**Virny** is a Python library for auditing model stability and fairness. The Virny library was
developed based on three fundamental principles: 

1) easy extensibility of model analysis capabilities;
2) compatibility to user-defined/custom datasets and model types;
3) simple composition of parity metrics based on context of use.

Virny decouples model auditing into several stages, including: **subgroup metrics computation**, **group metrics composition**,
and **metrics visualization and reporting**. This gives data scientists and practitioners more control and flexibility 
to use the library for model development and monitoring post-deployment.

For quickstart, look at our [Use Case Examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Runs_Interface_Use_Case/).


## Documentation 📒

* [Introduction](https://dataresponsibly.github.io/Virny/)
* [API Reference](https://dataresponsibly.github.io/Virny/api/overview/)
* [Use Case Examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Runs_Interface_Use_Case/)


## Features 💡

* Entire pipeline for auditing model stability and fairness
* Metrics reports and visualizations
* Ability to analyze intersections of sensitive attributes
* Blind classifiers audit
* Interface for multiple runs and multiple models
* User-friendly parameters input via config yaml files
* Built-in preprocessing techniques for raw classification datasets
* Check out [our documentation](https://dataresponsibly.github.io/Virny/) for a comprehensive overview


## Library Terminology 📖

This section briefly explains the main terminology used in our library.

* A **sensitive attribute** is an attribute that partitions the population into groups with unequal benefits received.
* A **protected group** (or simply _group_) is created by partitioning the population by one or many sensitive attributes.
* A **privileged value** of a sensitive attribute is a value that gives more benefit to a protected group, which includes it, than to protected groups, which do not include it.
* A **subgroup** is created by splitting a protected group by privileges and disprivileged values.
* A **group metric** is a metric that shows the relation between privileged and disprivileged subgroups created based on one or many sensitive attributes.


## Installation 🛠

Virny supports **Python 3.8, 3.9** and can be installed with `pip`:

```bash
pip install virny
```


## Affiliations 🤗

![NYU-UCU-Logos](https://user-images.githubusercontent.com/42843889/216840888-071bf184-f0e3-4a3e-94dc-c0d1c7784143.png)


## License 📝

**Virny** is free and open-source software licensed under the [3-clause BSD license](https://github.com/DataResponsibly/Virny/blob/main/LICENSE).

