# Overall Performance Dimensions

This page contains short descriptions and mathematical definitions for each overall metric implemented in Virny.


## Correctness

### Accuracy

Accuracy[^1] is the fraction of correct predictions. If $\hat{Y}_i$ is the predicted value of the $i$-th sample and $Y_i$
is the corresponding true value, then the fraction of correct predictions over $n_{samples}$ is defined as:

$$
\text{Accuracy } (Y, \hat{Y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \mathbf{1}(\hat{Y}_i = Y_i)
$$


### F1 Score

The F1 score[^2] can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches 
its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. 
The formula for the F1 score is:

$$
F_1 = \frac{2}{\text{ recall }^{-1} + \text{ precision }^{-1}} = 2 \frac{\text{ precision } \cdot \text{ recall }}{\text{ precision } + \text{ recall }} = \frac{2 \mathrm{TP}}{2 \mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
$$

where $TP$ is the number of true positives, $FN$ is the number of false negatives, and $FP$ is the number of false positives. 
F1 is by default calculated as 0.0 when there are no true positives, false negatives, or false positives.


## Representation

### Selection Rate

Selection Rate (or Base Rate) means the fraction of data points in each class classified as 1. The formula for the Selection Rate is:

$$
\text{Selection Rate} = P(\hat{Y} = 1) = \frac{TP + FP}{TP + FP + TN + FN}
$$


## Stability

### Label Stability
<a name="label-stability"></a>

Label Stability[^3] is defined as the normalized absolute difference between the number of times 
a sample is classified as positive or negative:

$$ 
U_{h, \text{stability}}(x) = \frac{1}{m} (\sum_{j=1}^m \mathbf{1}[h_{D_j}(x)>=0.5] - \sum_{i=1}^m \mathbf{1}[h_{D_i}(x)<0.5])
$$

where $x$ is an unseen test sample, and $h_{D_j}(x)$ is the predicted probability of the positive class of the $j^{\text{th}}$ model in the ensemble that has $m$ estimators.

Thus, Label Stability is a metric used to evaluate the level of disagreement between estimators in the ensemble. 
When the absolute difference is large, the label is more stable. On the other hand, if the difference is exactly zero, 
the estimator is said to be "highly unstable", meaning that a test sample has an equal probability of being classified 
as positive or negative by the ensemble.


### Jitter

Jitter[^4] is another measure of the disagreement between estimators in the ensemble, for each individual test example. 
It reuses a notion of $\textit{Churn}$[^5] to define a "$\textit{pairwise jitter}$":

$$
J_{i, j}\left(p_\theta\right)=\text{Churn}_{i, j}\left(p_\theta\right)=\frac{\left|p_{\theta i}(x) \neq p_{\theta j}(x)\right|_{x \in X}}{|X|}
$$

where $x$ is an unseen test sample, and $p_{\theta i}(x)$, $p_{\theta j}(x)$ are the prediction labels of the $i^{\text{th}}$ and $j^{\text{th}}$ estimator in the ensemble for $x$, respectively.

To obtain the overall measure of disagreement across all models in the ensemble, we need to calculate 
the average of $\textit{pairwise jitters}$ for all model pairs. This broader definition is referred to as $\textit{jitter}$:

$$
J\left(p_\theta\right)=\frac{\sum_{\forall i, j \in N} J_{i, j}\left(p_\theta\right)}{N \cdot(N-1) \cdot \frac{1}{2}} \text{, where } i<j
$$


## Uncertainty

### Epistemic Uncertainty
<a name="epistemic-uncertainty"></a>

Epistemic (model) uncertainty captures arbitrariness in outcomes due to uncertainty over the "true" model parameters. If we knew
the "best" model-type or hypothesis class to model each of our estimators in the bootstrap ($h_D$s) with, then, for different training datasets $D$,
we would fit the exact same model, and thereby have zero predictive variance. Following Tahir et al.[^6], we define
the epistemic uncertainty as a measure of predictive variance:

$$
U_{h, \text{epistemic}}(x) = \text{Var}_j[h_{D_j}(x)]
$$

where  $h_{D_j}(x)$ is the predicted probability of the positive class.


### Aleatoric Uncertainty
<a name="aleatoric-uncertainty"></a>

The source of aleatoric uncertainty is the inherent stochasticity present in the data that, in general, cannot be reduced/mitigated. 
An intuitive way to think about aleatoric uncertainty is through an information-theoretic perspective: if there simply is not enough 
(target-specific) information in a given data point, then even the "best" model will not be able to make a confident prediction for it.
Following Tahir et al.[^6], we define the aleatoric uncertainty as the expected entropy for the prediction:

$$
U_{h, \text{aleatoric}}(x) = \mathbb{E}_j[h_{D_j}(x)log(h_{D_j}(x)) + (1-h_{D_j}(x))log(1-h_{D_j}(x))]
$$

where  $h_{D_j}(x)$ and 1- $h_{D_j}(x)$ are the predicted probabilities of the positive and negative class respectively.


**References**

[^1]: Accuracy Score. Metrics and scoring: quantifying the quality of predictions. [Link](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)

[^2]: F1 score, scikit-learn documentation. [Link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

[^3]: Michael C. Darling and David J. Stracuzzi. “Toward Uncertainty Quantification for Supervised Classification”. In: 2018.

[^4]: Huiting Liu et al. “Model Stability with Continuous Data Updates”. In: arXiv preprint arXiv:2201.05692 (2022).

[^5]: Mahdi Milani Fard et al. “Launch and iterate: Reducing prediction churn”. In: Advances in Neural Information Processing Systems 29 (2016).

[^6]: Anique Tahir, Lu Cheng, and Huan Liu. 2023. Fairness through Aleatoric Uncertainty. In Proceedings of the 32nd ACM International Conference on
Information and Knowledge Management (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 2372–2381.
