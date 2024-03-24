# Measuring the Model Arbitrariness

Measuring only model accuracy is insufficient for reliable model development since an unstable or uncertain model can also lead 
to severe consequences in the production environment. Unfortunately, leading approaches to uncertainty quantification are 
tied to deep learning architectures and use Bayesian neural networks with methods like Monte Carlo Dropout[^1][^2].
However, deep learning models are not state-of-the-art on tabular datasets and are still outperformed by ensemble and 
tree-based models[^3]. Hence, model-agnostic uncertainty quantification is a major gap. 
We take a first step in this direction and propose a data-centric approach by bootstrapping over the training set[^4]. 

<figure>
    <img width="600" alt="data-centric-stability" src="https://github.com/DataResponsibly/Virny/assets/42843889/71d4b50f-b6e9-4a46-bfb6-c510e3f28be7">
    <figcaption>Figure 1. Data-centric evaluaton of model arbitrariness.</figcaption>
</figure>

Let $D(X,Y)$ be a (training) dataset of features and targets. Let $h(D)$ be the trained model, fit on the full dataset $D$, 
whose arbitrariness we are interested in quantifying.  We construct multiple training sets $D_i$ by sampling with replacement 
from the given training set $D$, and then fitting estimators $h_i(D_i)$ (with the same architecture and hyper-parameters as $h(D)$) 
on these bootstrapped training sets to construct an $\textit{approximating ensemble}$. At inference time, we use the approximating ensemble 
to construct a $\textit{predictive distribution}$ of class labels, with probabilities, for each test sample $x$ according to each model $h_i$.
We will then use this predictive distribution to compute different measures of variation/disagreement/arbitrariness between 
the predictions of members of the approximating ensemble, and approximate the variance of a single model trained on the full dataset.


**References**

[^1]: Yarin Gal and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

[^2]: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A Simple Way to Prevent Neural NetworksfromOverfitting.JournalofMachineLearningResearch15,56(2014),1929–1958.

[^3]: Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. 2022. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems (2022).

[^4]: Bradley Efron and Robert J Tibshirani. 1994. An introduction to the bootstrap. CRC press.
