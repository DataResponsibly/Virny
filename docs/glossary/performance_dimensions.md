# Performance Dimensions

Below, see the mathematical definition for each of the fairness metrics in the library.

## Overall Performance Dimensions

### Correctness

#### Accuracy

#### F1 Score


### Stability

#### Label Stability

#### Jitter


### Uncertainty

#### Aleatoric Uncertainty

```math
\frac{Pr(\hat{Y} = \text{pos_label} | D = \text{unprivileged})}
{Pr(\hat{Y} = \text{pos_label} | D = \text{privileged})}
```

```math
SE = \frac{\sigma}{\sqrt{n}}
```

#### Epistemic Uncertainty


### Representation

#### Selection Rate

#### Positive Rate


## Disparity Performance Dimensions

### Error Disparity

#### Accuracy Difference

#### Equalized Odds


### Stability Disparity

#### Label Stability Difference


### Uncertainty Disparity

#### Aleatoric Uncertainty Difference

#### Epistemic Uncertainty Difference


### Representation Disparity

#### Disparate Impact

#### Statistical Parity Difference
