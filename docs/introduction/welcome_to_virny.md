# Welcome to Virny

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


## Installation 🛠

Virny supports **Python 3.8** and can be installed with `pip`:

```bash
pip install virny
```


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
