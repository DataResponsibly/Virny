# Virny Overview

![Virny_Architecture](https://github.com/DataResponsibly/Virny/assets/42843889/91620e0f-11ff-4093-8fb6-c88c90bff711)

The software framework decouples the process of model profiling into several stages, including **subgroup metric computation**,
**disparity metric composition**, and **metric visualization**. This separation empowers data scientists with greater control and
flexibility in employing the library, both during model development and for post-deployment monitoring. The above figure demonstrates
how the library constructs a pipeline for model analysis. Inputs to a user interface are shown in green, pipeline stages are shown in blue,
and the output of each stage is shown in purple.

See more details in [our SIGMOD demo paper](https://dl.acm.org/doi/abs/10.1145/3626246.3654738).
