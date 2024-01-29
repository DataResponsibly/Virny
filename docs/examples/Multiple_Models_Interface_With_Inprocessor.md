# Multiple Models Interface With Inprocessor

In this example, we are going to audit one inprocessor from AIF360 for stability and fairness, visualize metrics, and create an analysis report. For that, we will use `compute_metrics_with_config` interface that can compute metrics for multiple models. Thus, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Make group metrics composition

* Create metrics visualizations and an analysis report

## Import dependencies


```python
import os
from pprint import pprint
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
```


## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **config yaml** that is a file with configuration parameters for different user interfaces for metrics computation.

* A **dataset class** that is a wrapper above the user’s raw dataset that includes its descriptive attributes like a target column, numerical columns, categorical columns, etc. This class must be inherited from the BaseDataset class, which was created for user convenience.

* Finally, a **models config** that is a Python dictionary, where keys are model names and values are initialized models for analysis. This dictionary helps conduct audits for different analysis modes and analyze different types of models.


```python
DATASET_SPLIT_SEED = 42
MODELS_TUNING_SEED = 42
TEST_SET_FRACTION = 0.2
```

### Create a config object

`compute_metrics_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: str, a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: float, the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: int, the number of estimators for bootstrap to compute subgroup stability metrics.

* **sensitive_attributes_dct**: dict, a dictionary where keys are sensitive attribute names (including intersectional attributes), and values are disadvantaged values for these attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify disadvantaged values for intersectional groups since they will be derived from disadvantaged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.


```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: Law_School
bootstrap_fraction: 0.8
computation_mode: error_analysis
n_estimators: 30  # Better to input the higher number of estimators than 100; this is only for this use case example
sensitive_attributes_dct: {'male': '0.0', 'race': 'Non-White', 'male&race': None}
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)
```


```python
config = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', f'{config.dataset_name}_Metrics_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}')
```

### Preprocess the dataset and create a BaseFlowDataset class


```python
from virny.datasets.data_loaders import LawSchoolDataset

data_loader = LawSchoolDataset()
data_loader.X_data[data_loader.X_data.columns[:5]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>decile1b</th>
      <th>decile3</th>
      <th>lsat</th>
      <th>ugpa</th>
      <th>zfygpa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>44.0</td>
      <td>3.5</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>29.0</td>
      <td>3.5</td>
      <td>-0.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>7.0</td>
      <td>37.0</td>
      <td>3.4</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>7.0</td>
      <td>43.0</td>
      <td>3.3</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>41.0</td>
      <td>3.3</td>
      <td>-0.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
column_transformer = ColumnTransformer(transformers=[
    ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
    ('numerical_features', StandardScaler(), data_loader.numerical_columns),
])
```


```python
# Create a binary race column for in-processing since aif360 inprocessors use a sensitive attribute during their learning.
data_loader.X_data['race_binary'] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'White' else 0)

base_flow_dataset = preprocess_dataset(data_loader, column_transformer, TEST_SET_FRACTION, DATASET_SPLIT_SEED)
base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']
```

### Define an inprocessor and create a wrapper for it

To use inprocessors from AIF360 together with Virny, we need to create a wrapper to use it as a basic model in the _models_config_.

A wrapper should include the following methods:

* **fit(self, X, y)**: fits an inprocessor based on X and y pandas dataframes. Returns self.

* **predict_proba(self, X)**: predicts using the fitted inprocessor based on X features pandas dataframe. Returns probabilities for **ZERO** class. These probabilities will be used by Virny in the downstream metric computation.

* **predict(self, X)**: predicts using the fitted inprocessor based on X features pandas dataframe. Returns labels for each sample.

* **__copy__(self)** and **__deepcopy__(self, memo)**: methods, which will be used during copy.copy(inprocessor_wrapper) and copy.deepcopy(inprocessor_wrapper). Return a new instance of inprocessor's wrapper.

* **get_params(self)**: returns parameters of the wrapper. Alignment with sklearn API.


```python
import copy
import numpy as np

from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from virny.custom_classes.base_inprocessing_wrapper import BaseInprocessingWrapper
from virny.utils.postprocessing_intervention_utils import construct_binary_label_dataset_from_df


class ExpGradientReductionWrapper(BaseInprocessingWrapper):
    """
    A wrapper for fair inprocessors from aif360. The wrapper aligns fit, predict, and predict_proba methods
    to be compatible with sklearn models.

    Parameters
    ----------
    inprocessor
        An initialized inprocessor from aif360.
    sensitive_attr_for_intervention
        A sensitive attribute name to use in the fairness in-processing intervention.

    """

    def __init__(self, inprocessor, sensitive_attr_for_intervention):
        self.sensitive_attr_for_intervention = sensitive_attr_for_intervention
        self.inprocessor = inprocessor

    def fit(self, X, y):
        train_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                      y_sample=y,
                                                                      target_column='target',
                                                                      sensitive_attribute=self.sensitive_attr_for_intervention)
        # Fit an inprocessor
        self.inprocessor.fit(train_binary_dataset)
        return self

    def predict_proba(self, X):
        y_empty = np.zeros(X.shape[0])
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        # Set 1.0 since ExponentiatedGradientReduction can return probabilities slightly higher than 1.0.
        # This can cause Infinity values for entropy.
        test_dataset_pred.scores[test_dataset_pred.scores > 1.0] = 1.0
        # Return 1 - test_dataset_pred.scores since scores are probabilities for label 1, not for label 0
        return 1 - test_dataset_pred.scores

    def predict(self, X):
        y_empty = np.zeros(shape=X.shape[0])
        test_binary_dataset = construct_binary_label_dataset_from_df(X_sample=X,
                                                                     y_sample=y_empty,
                                                                     target_column='target',
                                                                     sensitive_attribute=self.sensitive_attr_for_intervention)
        test_dataset_pred = self.inprocessor.predict(test_binary_dataset)
        return test_dataset_pred.labels

    def __copy__(self):
        new_inprocessor = copy.copy(self.inprocessor)
        return ExpGradientReductionWrapper(inprocessor=new_inprocessor,
                                           sensitive_attr_for_intervention=self.sensitive_attr_for_intervention)

    def __deepcopy__(self, memo):
        new_inprocessor = copy.deepcopy(self.inprocessor)
        return ExpGradientReductionWrapper(inprocessor=new_inprocessor,
                                           sensitive_attr_for_intervention=self.sensitive_attr_for_intervention)

    def get_params(self):
        return {'sensitive_attr_for_intervention': self.sensitive_attr_for_intervention}

```


```python
# Define a name of a sensitive attribute for the in-processing intervention.
# Note that in the above wrapper, 1 is used as a favorable label, and 0 is used as an unfavorable label.
sensitive_attr_for_intervention = 'race_binary'

# Define an inprocessor
estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
inprocessor = ExponentiatedGradientReduction(estimator=estimator,
                                             constraints='DemographicParity',
                                             drop_prot_attr=True)

models_config = {
    'ExponentiatedGradientReduction': ExpGradientReductionWrapper(inprocessor=inprocessor, 
                                                                  sensitive_attr_for_intervention=sensitive_attr_for_intervention)
}
```

## Subgroup Metrics Computation

After the variables are input to a user interface, the interface uses subgroup analyzers to compute different sets of metrics for each privileged and disadvantaged subgroup. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_dct = compute_metrics_with_config(dataset=base_flow_dataset,
                                          config=config,
                                          models_config=models_config,
                                          save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                          notebook_logs_stdout=True)
```


    Analyze multiple models:   0%|          | 0/1 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/30 [00:00<?, ?it/s]


Look at several columns in top rows of computed metrics


```python
sample_model_metrics_df = metrics_dct[list(models_config.keys())[0]]
sample_model_metrics_df[sample_model_metrics_df.columns[:6]].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>overall</th>
      <th>male_priv</th>
      <th>male_priv_correct</th>
      <th>male_priv_incorrect</th>
      <th>male_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean_Prediction</td>
      <td>0.023388</td>
      <td>0.020879</td>
      <td>0.014551</td>
      <td>0.086595</td>
      <td>0.026703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Overall_Uncertainty</td>
      <td>0.022109</td>
      <td>0.019718</td>
      <td>0.013374</td>
      <td>0.085599</td>
      <td>0.025269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aleatoric_Uncertainty</td>
      <td>0.008804</td>
      <td>0.007275</td>
      <td>0.005080</td>
      <td>0.030067</td>
      <td>0.010825</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Statistical_Bias</td>
      <td>0.098350</td>
      <td>0.089518</td>
      <td>0.004390</td>
      <td>0.973543</td>
      <td>0.110021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IQR</td>
      <td>0.010310</td>
      <td>0.009667</td>
      <td>0.007016</td>
      <td>0.037194</td>
      <td>0.011161</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Std</td>
      <td>0.009641</td>
      <td>0.008739</td>
      <td>0.005772</td>
      <td>0.039553</td>
      <td>0.010832</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Epistemic_Uncertainty</td>
      <td>0.013305</td>
      <td>0.012443</td>
      <td>0.008294</td>
      <td>0.055532</td>
      <td>0.014444</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Label_Stability</td>
      <td>0.987516</td>
      <td>0.988232</td>
      <td>0.991728</td>
      <td>0.951923</td>
      <td>0.986570</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jitter</td>
      <td>0.009299</td>
      <td>0.008656</td>
      <td>0.006096</td>
      <td>0.035234</td>
      <td>0.010149</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TPR</td>
      <td>0.990612</td>
      <td>0.991163</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.989861</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TNR</td>
      <td>0.138889</td>
      <td>0.133028</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.144860</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PPV</td>
      <td>0.908487</td>
      <td>0.918534</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.895129</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FNR</td>
      <td>0.009388</td>
      <td>0.008837</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.010139</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FPR</td>
      <td>0.861111</td>
      <td>0.866972</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.855140</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Accuracy</td>
      <td>0.902163</td>
      <td>0.912162</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.888951</td>
    </tr>
    <tr>
      <th>15</th>
      <td>F1</td>
      <td>0.947774</td>
      <td>0.953468</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.940114</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Selection-Rate</td>
      <td>0.977163</td>
      <td>0.979730</td>
      <td>0.986574</td>
      <td>0.908654</td>
      <td>0.973772</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Positive-Rate</td>
      <td>1.090397</td>
      <td>1.079070</td>
      <td>1.000000</td>
      <td>9.947368</td>
      <td>1.105830</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sample_Size</td>
      <td>4160.000000</td>
      <td>2368.000000</td>
      <td>2160.000000</td>
      <td>208.000000</td>
      <td>1792.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Group Metrics Composition

**Metrics Composer** is responsible for this second stage of the model audit. Currently, it computes our custom group fairness and stability metrics, but extending it for new group metrics is very simple. We noticed that more and more group metrics have appeared during the last decade, but most of them are based on the same subgroup metrics. Hence, such a separation of subgroup and group metrics computation allows one to experiment with different combinations of subgroup metrics and avoid subgroup metrics recomputation for a new set of grouped metrics.


```python
models_metrics_dct = read_model_metric_dfs(SAVE_RESULTS_DIR_PATH, model_names=list(models_config.keys()))
```


```python
metrics_composer = MetricsComposer(models_metrics_dct, config.sensitive_attributes_dct)
```

Compute composed metrics


```python
models_composed_metrics_df = metrics_composer.compose_metrics()
```


```python
models_composed_metrics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>male</th>
      <th>race</th>
      <th>male&amp;race</th>
      <th>Model_Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy_Difference</td>
      <td>-0.023211</td>
      <td>-0.180454</td>
      <td>-0.160337</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aleatoric_Uncertainty_Difference</td>
      <td>0.003550</td>
      <td>0.030174</td>
      <td>0.032971</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aleatoric_Uncertainty_Ratio</td>
      <td>1.487968</td>
      <td>8.174811</td>
      <td>6.327559</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Epistemic_Uncertainty_Difference</td>
      <td>0.002002</td>
      <td>0.010810</td>
      <td>0.015361</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Epistemic_Uncertainty_Ratio</td>
      <td>1.160858</td>
      <td>1.927304</td>
      <td>2.270925</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Equalized_Odds_FNR</td>
      <td>0.001302</td>
      <td>0.001559</td>
      <td>0.003110</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Equalized_Odds_FPR</td>
      <td>-0.011832</td>
      <td>0.082345</td>
      <td>0.057266</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>7</th>
      <td>IQR_Difference</td>
      <td>0.001494</td>
      <td>0.018301</td>
      <td>0.021933</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jitter_Difference</td>
      <td>0.001493</td>
      <td>0.015634</td>
      <td>0.017956</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Label_Stability_Ratio</td>
      <td>0.998318</td>
      <td>0.979427</td>
      <td>0.976888</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Label_Stability_Difference</td>
      <td>-0.001662</td>
      <td>-0.020380</td>
      <td>-0.022865</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overall_Uncertainty_Difference</td>
      <td>0.005552</td>
      <td>0.040984</td>
      <td>0.048332</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overall_Uncertainty_Ratio</td>
      <td>1.281547</td>
      <td>3.583619</td>
      <td>3.644668</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Disparate_Impact</td>
      <td>1.024800</td>
      <td>1.248497</td>
      <td>1.215937</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Statistical_Parity_Difference</td>
      <td>-0.005957</td>
      <td>-0.010275</td>
      <td>-0.011401</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Std_Difference</td>
      <td>0.002092</td>
      <td>0.011726</td>
      <td>0.015143</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Std_Ratio</td>
      <td>1.239415</td>
      <td>2.493035</td>
      <td>2.794353</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Equalized_Odds_TNR</td>
      <td>0.011832</td>
      <td>-0.082345</td>
      <td>-0.057266</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Equalized_Odds_TPR</td>
      <td>-0.001302</td>
      <td>-0.001559</td>
      <td>-0.003110</td>
      <td>ExponentiatedGradientReduction</td>
    </tr>
  </tbody>
</table>
</div>



## Metrics Visualization and Reporting

**Metric Visualizer** allows us to build static visualizations for the computed metrics. It unifies different preprocessing methods for the computed metrics and creates various data formats required for visualizations. Hence, users can simply call methods of the _MetricsVisualizer_ class and get custom plots for diverse metric analysis.


```python
visualizer = MetricsVisualizer(models_metrics_dct, models_composed_metrics_df, config.dataset_name,
                               model_names=list(models_config.keys()),
                               sensitive_attributes_dct=config.sensitive_attributes_dct)
```


```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Accuracy', 'F1', 'TPR', 'TNR', 'PPV', 'Selection-Rate'],
    plot_title="Accuracy Metrics"
)
```





<div id="altair-viz-88507ce88f374bfc8d4eeffaae67738f"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-88507ce88f374bfc8d4eeffaae67738f") {
      outputDiv = document.getElementById("altair-viz-88507ce88f374bfc8d4eeffaae67738f");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-8e1abd75ad4d8436dab939cadfc9eb86"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Accuracy", "F1", "TPR", "TNR", "PPV", "Selection-Rate"], "title": "Accuracy Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-8e1abd75ad4d8436dab939cadfc9eb86": [{"overall": 0.9021634615384616, "metric": "Accuracy", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.9477736430129604, "metric": "F1", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.9084870848708488, "metric": "PPV", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.9771634615384616, "metric": "Selection-Rate", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.1388888888888889, "metric": "TNR", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.9906115879828328, "metric": "TPR", "model_name": "ExponentiatedGradientReduction"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Aleatoric_Uncertainty', 'Epistemic_Uncertainty', 'Std', 'IQR', 'Jitter'],
    plot_title="Stability and Uncertainty Metrics"
)
```





<div id="altair-viz-4b9935d05e4044f3b5cfdd1f45c67c9c"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-4b9935d05e4044f3b5cfdd1f45c67c9c") {
      outputDiv = document.getElementById("altair-viz-4b9935d05e4044f3b5cfdd1f45c67c9c");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-a874bf725903e4febd75fa4668d469cb"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Aleatoric_Uncertainty", "Epistemic_Uncertainty", "Std", "IQR", "Jitter"], "title": "Stability and Uncertainty Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-a874bf725903e4febd75fa4668d469cb": [{"overall": 0.008804261223772, "metric": "Aleatoric_Uncertainty", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.0133050928217114, "metric": "Epistemic_Uncertainty", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.0103101389257501, "metric": "IQR", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.0092987400530503, "metric": "Jitter", "model_name": "ExponentiatedGradientReduction"}, {"overall": 0.009640551539234, "metric": "Std", "model_name": "ExponentiatedGradientReduction"}]}}, {"mode": "vega-lite"});
</script>
