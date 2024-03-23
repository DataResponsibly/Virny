# Performance Dimensions

Below, see the mathematical definition for each of the fairness metrics in the library.

## Overall Performance Dimensions

### Correctness

#### Accuracy

#### F1 Score


### Stability

#### Label Stability

Label Stability[^1] is defined as the normalized absolute difference between the number of times 
a sample is classified as positive or negative:

$$
S\left(p_\theta\right) = \frac{|\sum_{i=1}^{b} 1[p_{\theta_{i}}(x)==1] - \sum_{i=1}^{b} 1[p_{\theta_{i}}(x)==0]|}{b}
$$

where $x$ is an unseen test sample, and $p_{\theta_{i}}(x)$ is the prediction label of the $i^{\text{th}}$ model in the ensemble that has $b$ estimators.

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

Epistemic (model) uncertainty captures arbitrariness in outcomes due to uncertainty over the "true" model parameters. If we knew
the "best" model-type or hypothesis class to model each of our estimators in the bootstrap ($h_D$s) with, then, for different training datasets $D$,
we would fit the exact same model, and thereby have zero predictive variance. Following Tahir et al.[^3], we define
the epistemic uncertainty as a measure of predictive variance:

$$
U_{h, \text{epistemic}}(x) = \text{Var}_j[h_{D_j}(x)]
$$

where  $h_{D_j}(x)$ is the predicted probability of the positive class.


#### Aleatoric Uncertainty

The source of aleatoric uncertainty is the inherent stochasticity present in the data that, in general, cannot be reduced/mitigated. 
An intuitive way to think about aleatoric uncertainty is through an information-theoretic perspective: if there simply is not enough 
(target-specific) information in a given data point, then even the "best" model will not be able to make a confident prediction for it.
Following Tahir et al.[^3], we define the aleatoric uncertainty as the expected entropy for the prediction:

$$
U_{h, \text{aleatoric}}(x) = \mathbb{E}_j[h_{D_j}(x)log(h_{D_j}(x)) + (1-h_{D_j}(x))log(1-h_{D_j}(x))]
$$

where  $h_{D_j}(x)$ and 1- $h_{D_j}(x)$ are the predicted probabilities of the positive and negative class respectively.

### Representation

#### Selection Rate

#### Positive Rate


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

#### Accuracy Difference

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
for the disadvantaged group ($A=0$, which we call $\textit{\dis}$) and for the privileged group ($A=1$, which we call $\textit{\priv}$), respectively.

Equalized Odds Violation (False Positive):

$$
\Delta\text{FPR} = P(\hat{Y}=1|A=0,Y=0)- P(\hat{Y}=1|A=1,Y=0)
$$

Equalized Odds Violation (False Negative):

$$
\Delta\text{FNR} = P(\hat{Y}=0|A=0,Y=1)- P(\hat{Y}=0|A=1,Y=1)
$$


### Stability Disparity

#### Label Stability Difference


### Uncertainty Disparity

#### Epistemic Uncertainty Difference

#### Aleatoric Uncertainty Difference


### Representation Disparity

#### Disparate Impact

#### Statistical Parity Difference


**References**

[^1]: Michael C. Darling and David J. Stracuzzi. “Toward Uncertainty Quantification for Supervised Classification”. In: 2018.

[^2]: Huiting Liu et al. “Model Stability with Continuous Data Updates”. In: arXiv preprint arXiv:2201.05692 (2022).

[^3]: Anique Tahir, Lu Cheng, and Huan Liu. 2023. Fairness through Aleatoric Uncertainty. In Proceedings of the 32nd ACM International Conference on
Information and Knowledge Management (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 2372–2381.

[^4]: Mahdi Milani Fard et al. “Launch and iterate: Reducing prediction churn”. In: Advances in Neural Information Processing Systems 29 (2016).

[^5]: Moritz Hardt, Eric Price, and Nati Srebro. “Equality of Opportunity in Supervised Learning”. In: Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain. Ed. by Daniel D. Lee et al. 2016, pp. 3315–3323.
