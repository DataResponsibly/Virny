# How These Metrics Should Be Chosen

While numerous performance metrics have been proposed, there is no clear guidance on which metrics should be 
used in practice to evaluate and audit ML systems. Generally, the choice of metrics is always context-specific,
and on this page, we aim to summarize suggestions for choosing an appropriate set of metrics.


## Suggestions about error disparity metrics

In Saleiro et al.'s study[^1], the authors endeavor to establish a clear guideline for linking various 
group fairness metrics (error disparity metrics in Virny) with real-world challenges. Consequently, they collaborated 
with policymakers to develop the "Fairness Tree," which serves as a comprehensive roadmap of the most relevant bias metrics.
This navigation is crafted from the perspective of decision-makers and presupposes fundamental policy choices made at the beginning of a data science/AI project. 
The "Fairness Tree" is accessible via their web application on the [provided website](http://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/).


## Suggestions about stability and uncertainty disparity metrics


**References**

[^1]: Pedro Saleiro et al. “Aequitas: A bias and fairness audit toolkit”. In: arXiv preprint arXiv:1811.05577 (2018).
