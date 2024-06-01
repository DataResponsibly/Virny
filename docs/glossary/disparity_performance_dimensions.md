# Disparity Performance Dimensions

This page contains short descriptions and mathematical definitions for each disparity metric implemented in Virny.


## Notation

Let $Y$ be the true outcome, $X$ be the covariates or features, and $A$ be a set of sensitive attributes.
To start, we restrict our analysis to the binary classification of group membership, letting $A=0$ denote the disadvantaged group
and $A=1$ denote the privileged group. We are interested in constructing a classifier $\hat{Y} = f(X,A)$ that predicts $Y$ using
an appropriate loss function. In fair-ML, we apply additional constraints on the interaction between $\hat{Y}$, $Y$,
and $A$ to ensure that the classifier $\hat{Y}$ does not discriminate on the basis of sensitive attributes $A$.
Various fairness definitions are formalized as distinct constraints, and when the fairness constraint is not satisfied,
it is commonly known as the measure of model unfairness, which we will discuss next.

We will now rewrite influential fairness measures by expressing them as the difference or ratio between different
$\textit{base measures}$ on the disadvantage ($\textit{dis}$) and privileged ($\textit{priv}$) groups: $\Delta f = f_{dis} - f_{priv}$, $\mathcal{Q} f = f_{dis} / f_{priv}$.


## Error Disparity

### Equalized Odds

Hardt, Price, and Srebro[^1] state that a predictor $\hat{Y}$ satisfies $\textit{equalized odds}$ with respect
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


### Accuracy Difference

Accuracy Difference is a fairness notion that requires equal accuracy across groups.

$$
\Delta(\text{Accuracy}) = P(\hat{Y}=Y|A=0) - P(\hat{Y}=Y|A=1)
$$


## Representation Disparity

### Disparate Impact

Inspired by the 4/5th's rule in legal doctrines, Disparate Impact[^2][^3] has been formulated as a fairness metric:

$$
\text{Disparate Impact} = \mathcal{Q}(\text{Selection Rate}) = \frac{P(\hat{Y}=1|A=0)}{P(\hat{Y}=1|A=1)}
$$

$P(\hat{Y}=1)$ is simply the Selection Rate of the classifier, and so the measure of Disparate Impact is composed
as the ratio of the Selection Rate on the $\textit{dis}$ and $\textit{priv}$ groups, respectively.


### Statistical Parity Difference

Statistical Parity[^4][^5][^6][^7] is the fairness notion that asks if comparable proportions of samples from
each protected group receive the positive outcome:

$$
P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)
$$

Statistical Parity Difference (SPD) is a popular fairness measure composed simply as the difference between
the classifier's Selection Rate on $\textit{dis}$ and $\textit{priv}$ groups, respectively.

$$
\text{Statistical Parity Difference} = \Delta(\text{Selection Rate}) = P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)
$$


## Stability Disparity

### Label Stability Difference

Label Stability Difference measures the equality (or lack thereof) of _label stability_ across groups.
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups:

$$
\Delta U_{h, \text{stability}}(D)  = \mathbb{E}_{x \in D_\text{dis}}[U_{\text{stability}}(x)] - \mathbb{E}_{x^* \in D_\text{priv}}[U_{\text{stability}}(x^*)]
$$

where $D$ is a test set and $x$ and $x^*$ are $dis$ and $priv$ samples in $D$.


## Uncertainty Disparity

### Epistemic Uncertainty Difference

Epistemic Uncertainty Difference measures the equality (or lack thereof) of _epistemic (model) uncertainty_ across groups.
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups:

$$
\Delta U_{h, \text{epistemic}}(D)  = \mathbb{E}_{x \in D_\text{dis}}[U_{\text{epistemic}}(x)] -  \mathbb{E}_{x^* \in D_\text{priv}}[U_{\text{epistemic}}(x^*)]
$$

where $D$ is a test set and $x$ and $x^*$ are $dis$ and $priv$ samples in $D$.


### Aleatoric Uncertainty Difference

Aleatoric Uncertainty Difference measures the equality (or lack thereof) of _aleatoric (data) uncertainty_ across groups.
In practice, this metric is implemented as a difference between the averaged metric value for $\textit{dis}$ and $\textit{priv}$ groups:

$$
\Delta U_{h, \text{aleatoric}}(D)  = \mathbb{E}_{x \in D_\text{dis}}[U_{\text{aleatoric}}(x)] -  \mathbb{E}_{x^* \in D_\text{priv}}[U_{\text{aleatoric}}(x^*)]
$$

where $D$ is a test set and $x$ and $x^*$ are $dis$ and $priv$ samples in $D$.


**References**

[^1]: Moritz Hardt, Eric Price, and Nati Srebro. “Equality of Opportunity in Supervised Learning”. In: Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain. Ed. by Daniel D. Lee et al. 2016, pp. 3315–3323.

[^2]: Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big Data 5, 2 (2017), 153–163. 2016

[^3]: Feldman, Michael, et al. "Certifying and removing disparate impact." proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.

[^4]: Toon Calders and Sicco Verwer. Three naive bayes approaches for discrimination-free classification. Data Mining and Knowledge Discovery, 21(2):277–292, 2010.

[^5]: Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’15, pages 259–268, New York, NY, USA, 2015. ACM.

[^6]: Faisal Kamiran and Toon Calders. Classifying without discriminating. 2009 2nd International Conference on Computer, Control and Communication, IC4 2009, 2009.

[^7]: Toshihiro Kamishima, Shotaro Akaho, and Jun Sakuma. Fairness-aware learning through regularization approach. Proceedings - IEEE International Conference on Data Mining, ICDM, pages 643–650, 2011.
