# Welcome to Virny

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

Virny supports **Python 3.8 and 3.9** and can be installed with `pip`:

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


## 💡 Full List of Features

* Entire pipeline for profiling model accuracy, stability, uncertainty, and fairness
* Ability to analyze non-binary sensitive attributes and their intersections
* Convenient metric computation interfaces: an interface for multiple models, an interface for multiple test sets, and an interface for saving results into a user-defined database
* Interactive _VirnyView_ visualizer that profiles dataset properties related to protected groups, computes comprehensive [nutritional labels](http://sites.computer.org/debull/A19sept/p13.pdf) for individual models, compares multiple models according to multiple metrics, and guides users through model selection
* Compatibility with [pre-, in-, and post-processors](https://aif360.readthedocs.io/en/latest/modules/algorithms.html#) for fairness enhancement from AIF360
* An `error_analysis` computation mode to analyze model stability and confidence for correct and incorrect prodictions broken down by groups
* Metric static and interactive visualizations
* Data loaders with subsampling for popular fair-ML benchmark datasets
* User-friendly parameters input via config yaml files


## 📖 Library Overview

![Virny_Architecture](https://github.com/DataResponsibly/Virny/assets/42843889/91620e0f-11ff-4093-8fb6-c88c90bff711)

The software framework decouples the process of model profiling into several stages, including **subgroup metric computation**,
**disparity metric composition**, and **metric visualization**. This separation empowers data scientists with greater control and
flexibility in employing the library, both during model development and for post-deployment monitoring. The above figure demonstrates
how the library constructs a pipeline for model analysis. Inputs to a user interface are shown in green, pipeline stages are shown in blue,
and the output of each stage is shown in purple.


## 🤗 Affiliations

![NYU-UCU-Logos](https://user-images.githubusercontent.com/42843889/216840888-071bf184-f0e3-4a3e-94dc-c0d1c7784143.png)


## 📝 License

**Virny** is free and open-source software licensed under the [3-clause BSD license](https://github.com/DataResponsibly/Virny/blob/main/LICENSE).
