# Welcome to Virny

## 📜 Description

**Virny** serves as a model profiling library, empowering data scientists _to engage in responsible model selection_ and
_to generate a "nutritional label" for an ML model_, which uncovers the model performance across different overall and disparity dimensions. 
The Virny library was developed based on three fundamental principles: 

1) easy extensibility of model analysis capabilities;

2) compatibility to user-defined/custom datasets and model types;

3) simple composition of parity metrics based on the context of use.

Virny decouples model auditing into several stages, including: **subgroup metric computation**, **disparity metric composition**,
and **metric visualization**. This gives data scientists more control and flexibility to use the library
for model development and monitoring post-deployment.

For quickstart, look at our [Use Case Examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/).


## 🛠 Installation

Virny supports **Python 3.8 and 3.9** and can be installed with `pip`:

```bash
pip install virny
```


## 📒 Documentation

* [Introduction](https://dataresponsibly.github.io/Virny/)
* [API Reference](https://dataresponsibly.github.io/Virny/api/overview/)
* [Use Case Examples](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/)


## 💡 Features

* Entire pipeline for profiling model accuracy, stability, uncertainty, and fairness
* Metric static and interactive visualizations
* Ability to analyze non-binary sensitive attributes and their intersections
* Convenient metric computation interfaces: an interface for multiple models, an interface for multiple test sets, and an interface for saving results into a user-defined database
* An `error_analysis` computation mode to analyze model stability and confidence for correct and incorrect prodictions broken down by groups
* Data loaders with subsampling for popular fair-ML benchmark datasets
* User-friendly parameters input via config yaml files
* Check out [our documentation](https://dataresponsibly.github.io/Virny/) for a comprehensive overview


## 📖 Library Overview

![Virny_Architecture](https://github.com/DataResponsibly/Virny/assets/42843889/91620e0f-11ff-4093-8fb6-c88c90bff711)

The software framework decouples the process of model profiling into several stages, including **subgroup metric computation**,
**disparity metric composition**, and **metric visualization**. This separation empowers data scientists with greater control and
flexibility in employing the library, both during model development and for post-deployment monitoring. The above figure demonstrates
how the library constructs a pipeline for model analysis. Inputs to a user interface are shown in green, pipeline stages are shown in blue,
and the output of each stage is shown in purple.
