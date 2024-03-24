# Performance Dimensions

This page contains short descriptions and mathematical definitions for each overall and disparity metrics implemented in Virny.


## Overall Performance Dimensions

### Correctness

#### Accuracy

Accuracy[^11] is the fraction of correct predictions. If $\hat{Y}_i$ is the predicted value of the $i$-th sample and $Y_i$
is the corresponding true value, then the fraction of correct predictions over $n_{samples}$ is defined as:

$$
\text{Accuracy } (Y, \hat{Y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \mathbf{1}(\hat{Y}_i = Y_i)
$$


#### F1 Score

The F1 score[^12] can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches 
its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. 
The formula for the F1 score is:

$$
F_1 = \frac{2}{\text{ recall }^{-1} + \text{ precision }^{-1}} = 2 \frac{\text{ precision } \cdot \text{ recall }}{\text{ precision } + \text{ recall }} = \frac{2 \mathrm{TP}}{2 \mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
$$

where $TP$ is the number of true positives, $FN$ is the number of false negatives, and $FP$ is the number of false positives. 
F1 is by default calculated as 0.0 when there are no true positives, false negatives, or false positives.


### Representation

#### Selection Rate

Selection Rate (or Base Rate) means the fraction of data points in each class classified as 1. The formula for the Selection Rate is:

$$
\text{Selection Rate} = \frac{TP + FP}{TP + FP + TN + FN}
$$


### Stability

#### Label Stability
<a name="label-stability"></a>

Label Stability[^1] is defined as the normalized absolute difference between the number of times 
a sample is classified as positive or negative:

$$ 
\text{Label Stability } U_{h, \text{stability}}(x^*) = \frac{1}{m} (\sum_{j=1}^m \mathbf{1}[h_{D_j}(x^*)>=0.5] - \sum_{i=1}^m \mathbf{1}[h_{D_i}(x^*)<0.5])
$$

where $x^*$ is an unseen test sample, and $h_{D_j}(x)$ is the predicted probability of the positive class of the $j^{\text{th}}$ model in the ensemble that has $m$ estimators.

Thus, Label Stability is a metric used to evaluate the level of disagreement between estimators in the ensemble. 
When the absolute difference is large, the label is more stable. On the other hand, if the difference is exactly zero, 
the estimator is said to be "highly unstable", meaning that a test sample has an equal probability of being classified 
as positive or negative by the ensemble.


#### Jitter

Jitter[^2] is another measure of the disagreement between estimators in the ensemble, for each individual test example. 
It reuses a notion of $\textit{Churn}$[^4] to define a "$\textit{pairwise jitter}$":

$$
J_{i, j}\left(p_\theta\right)=\text{Churn}_{i, j}\left(p_\theta\right)=\frac{\left|p_{\theta i}(x) \neq p_{\theta j}(x)\right|_{x \in X}}{|X|}
$$

where $x$ is an unseen test sample, and $p_{\theta i}(x)$, $p_{\theta j}(x)$ are the prediction labels of the $i^{\text{th}}$ and $j^{\text{th}}$ estimator in the ensemble for $x$, respectively.

To obtain the overall measure of disagreement across all models in the ensemble, we need to calculate 
the average of $\textit{pairwise jitters}$ for all model pairs. This broader definition is referred to as $\textit{jitter}$:

$$
J\left(p_\theta\right)=\frac{\sum_{\forall i, j \in N} J_{i, j}\left(p_\theta\right)}{N \cdot(N-1) \cdot \frac{1}{2}} \text{, where } i<j
$$


### Uncertainty

#### Epistemic Uncertainty
<a name="epistemic-uncertainty"></a>

Epistemic (model) uncertainty captures arbitrariness in outcomes due to uncertainty over the "true" model parameters. If we knew
the "best" model-type or hypothesis class to model each of our estimators in the bootstrap ($h_D$s) with, then, for different training datasets $D$,
we would fit the exact same model, and thereby have zero predictive variance. Following Tahir et al.[^3], we define
the epistemic uncertainty as a measure of predictive variance:

$$
U_{h, \text{epistemic}}(x) = \text{Var}_j[h_{D_j}(x)]
$$

where  $h_{D_j}(x)$ is the predicted probability of the positive class.


#### Aleatoric Uncertainty
<a name="aleatoric-uncertainty"></a>

The source of aleatoric uncertainty is the inherent stochasticity present in the data that, in general, cannot be reduced/mitigated. 
An intuitive way to think about aleatoric uncertainty is through an information-theoretic perspective: if there simply is not enough 
(target-specific) information in a given data point, then even the "best" model will not be able to make a confident prediction for it.
Following Tahir et al.[^3], we define the aleatoric uncertainty as the expected entropy for the prediction:

$$
U_{h, \text{aleatoric}}(x) = \mathbb{E}_j[h_{D_j}(x)log(h_{D_j}(x)) + (1-h_{D_j}(x))log(1-h_{D_j}(x))]
$$

where  $h_{D_j}(x)$ and 1- $h_{D_j}(x)$ are the predicted probabilities of the positive and negative class respectively.


## Disparity Performance Dimensions

**Notation.** Let $Y$ be the true outcome, $X$ be the covariates or features, and $A$ be a set of sensitive attributes. 
To start, we restrict our analysis to the binary classification of group membership, letting $A=0$ denote the disadvantaged group 
and $A=1$ denote the privileged group. We are interested in constructing a classifier $\hat{Y} = f(X,A)$ that predicts $Y$ using 
an appropriate loss function. In fair-ML, we apply additional constraints on the interaction between $\hat{Y}$, $Y$, 
and $A$ to ensure that the classifier $\hat{Y}$ does not discriminate on the basis of sensitive attributes $A$. 
Various fairness definitions are formalized as distinct constraints, and when the fairness constraint is not satisfied, 
it is commonly known as the measure of model unfairness, which we will discuss next.

We will now rewrite influential fairness measures by expressing them as the difference or ratio between different 
$\textit{base measures}$ on the disadvantage ($\textit{dis}$) and privileged ($\textit{priv}$) groups: $\Delta f = f_{dis} - f_{priv}$, $\mathcal{Q} f = f_{dis} / f_{priv}$.


### Error Disparity

#### Equalized Odds

Hardt, Price, and Srebro[^5] state that a predictor $\hat{Y}$ satisfies $\textit{equalized odds}$ with respect 
to sensitive attribute $A$ and outcome $Y$, if $\hat{Y}$ and $A$ are independent conditional on $Y$. In our framework,
we focus on the following expression of equalized odds:

$$
P(\hat{Y}=\hat{y}|A=0,Y=y)=P(\hat{Y}=\hat{y}|A=1,Y=y), y \in \{0,1\}
$$

For $\hat{Y} = 1$ (the predicted positive outcome) and $Y = 0$ (the true negative outcome), this fairness constraint 
requires parity in false positive rates (FPR) across the groups $A = 0$ and $A = 1$. For $\hat{Y} = 0$ (the predicted negative outcome) 
and $Y = 1$ (the true positive outcome), the constraint requires parity in false negative rates (FNR). A violation of this constraint 
(e.g., the disparity in FPR and FNR across groups) is reported as a measure of model unfairness. 
In our library, we refer to FPR and FNR as the $\textit{base measures}$, and we say that the fairness criterion of 
Equalized Odds is $\textit{composed}$ as the difference between these $\textit{base measures}$ computed 
for the disadvantaged group ($A=0$, which we call $\textit{dis}$) and for the privileged group ($A=1$, which we call $\textit{priv}$), respectively.

Equalized Odds Violation (False Positive):

$$
\Delta\text{FPR} = P(\hat{Y}=1|A=0,Y=0)- P(\hat{Y}=1|A=1,Y=0)
$$

Equalized Odds Violation (False Negative):

$$
\Delta\text{FNR} = P(\hat{Y}=0|A=0,Y=1)- P(\hat{Y}=0|A=1,Y=1)
$$


#### Accuracy Difference

Accuracy Difference is a fairness notion that requires equal accuracy across groups.

$$
\Delta(\text{Accuracy}) = P(\hat{Y}=Y|A=0) - P(\hat{Y}=Y|A=1)
$$


### Representation Disparity

#### Disparate Impact

Inspired by the 4/5th's rule in legal doctrines, Disparate Impact[^6][^13] has been formulated as a fairness metric:

$$
\text{Disparate Impact} = \mathcal{Q}(\text{Selection Rate}) = \frac{P(\hat{Y}=1|A=0)}{P(\hat{Y}=1|A=1)}
$$

$P(\hat{Y}=1)$ is simply the Selection Rate of the classifier, and so the measure of Disparate Impact is composed 
as the ratio of the Selection Rate on the $\textit{dis}$ and $\textit{priv}$ groups, respectively.


#### Statistical Parity Difference

Statistical Parity[^7][^8][^9][^10] is the fairness notion that asks if comparable proportions of samples from 
each protected group receive the positive outcome:

$$
P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)
$$

Statistical Parity Difference (SPD) is a popular fairness measure composed simply as the difference between 
the classifier's Selection Rate on $\textit{dis}$ and $\textit{priv}$ groups, respectively.

$$
\text{Statistical Parity Difference} = \Delta(\text{Selection Rate}) = P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)
$$


### Stability Disparity

#### Label Stability Difference

Label Stability Difference measures the equality (or lack thereof) of [label stability](#label-stability) across groups. 
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups.

$$
\Delta U_{h, \text{stability}}(D^*)  = \mathbb{E}_{x* \in D^*_\text{dis}}[U_{\text{stability}}(x)] - \mathbb{E}_{x* \in D^*_\text{priv}}[U_{\text{stability}}(x)]
$$


### Uncertainty Disparity

#### Epistemic Uncertainty Difference

Epistemic Uncertainty Difference measures the equality (or lack thereof) of [epistemic (model) uncertainty](#epistemic-uncertainty) across groups.
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups.

$$
\Delta U_{h, \text{epistemic}}(D^*)  = \mathbb{E}_{x* \in D^*_\text{dis}}[U_{\text{epistemic}}(x)] -  \mathbb{E}_{x* \in D^*_\text{priv}}[U_{\text{epistemic}}(x)]
$$


#### Aleatoric Uncertainty Difference

Aleatoric Uncertainty Difference measures the equality (or lack thereof) of [aleatoric (data) uncertainty](#aleatoric-uncertainty) across groups.
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups.

$$
\Delta U_{h, \text{aleatoric}}(D^*)  = \mathbb{E}_{x* \in D^*_\text{dis}}[U_{\text{aleatoric}}(x)] -  \mathbb{E}_{x* \in D^*_\text{priv}}[U_{\text{aleatoric}}(x)]
$$


**References**

[^1]: Michael C. Darling and David J. Stracuzzi. “Toward Uncertainty Quantification for Supervised Classification”. In: 2018.

[^2]: Huiting Liu et al. “Model Stability with Continuous Data Updates”. In: arXiv preprint arXiv:2201.05692 (2022).

[^3]: Anique Tahir, Lu Cheng, and Huan Liu. 2023. Fairness through Aleatoric Uncertainty. In Proceedings of the 32nd ACM International Conference on
Information and Knowledge Management (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 2372–2381.

[^4]: Mahdi Milani Fard et al. “Launch and iterate: Reducing prediction churn”. In: Advances in Neural Information Processing Systems 29 (2016).

[^5]: Moritz Hardt, Eric Price, and Nati Srebro. “Equality of Opportunity in Supervised Learning”. In: Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain. Ed. by Daniel D. Lee et al. 2016, pp. 3315–3323.

[^6]: Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big Data 5, 2 (2017), 153–163. 2016

[^7]: Toon Calders and Sicco Verwer. Three naive bayes approaches for discrimination-free classification. Data Mining and Knowledge Discovery, 21(2):277–292, 2010.

[^8]: Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’15, pages 259–268, New York, NY, USA, 2015. ACM.

[^9]: Faisal Kamiran and Toon Calders. Classifying without discriminating. 2009 2nd International Conference on Computer, Control and Communication, IC4 2009, 2009.

[^10]: Toshihiro Kamishima, Shotaro Akaho, and Jun Sakuma. Fairness-aware learning through regularization approach. Proceedings - IEEE International Conference on Data Mining, ICDM, pages 643–650, 2011.

[^11]: Accuracy Score. Metrics and scoring: quantifying the quality of predictions. [Link](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)

[^12]: F1 score, scikit-learn documentation. [Link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)

[^13]: Feldman, Michael, et al. "Certifying and removing disparate impact." proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.
