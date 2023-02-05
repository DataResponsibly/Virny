# One Run Multiple Models Interface Usage

In this example, we are going to audit 4 models for stability and fairness, visualize metrics, and create an analysis report. To get better analysis accuracy, we will use `run_metrics_computation_with_config` interface that will make multiple runs per model. For that, we will need to do next steps:

* Initialize input variables

* Compute subgroup metrics

* Make group metrics composition

* Create metrics visualizations and an analysis report

## Import dependencies


```python
import os
import pandas as pd
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from virny.user_interfaces.metrics_computation_interfaces import run_metrics_computation_with_config
from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.custom_classes.base_dataset import BaseDataset
from virny.configs.constants import ReportType
```

## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **dataset class** that is a wrapper above the user’s raw dataset that includes its descriptive attributes like a target column, numerical columns, categorical columns, etc. This class must be inherited from the BaseDataset class, which was created for user convenience.

* A **config yaml** that is a file with configuration parameters for different user interfaces for metrics computation.

* Finally, a **models config** that is a Python dictionary, where keys are model names and values are initialized models for analysis. This dictionary helps conduct audits of multiple models for one or multiple runs and analyze different types of models.

### Create a Dataset class

Based on the BaseDataset class, your **dataset class** should include the following attributes:
* **Obligatory attributes**: dataset, target, features, numerical_columns, categorical_columns

* **Optional attributes**: X_data, y_data, columns_with_nulls

For more details, please refer to the library documentation.


```python
class CompasDataset(BaseDataset):
    """
    Dataset class for COMPAS dataset that contains sensitive attributes among feature columns.

    Parameters
    ----------
    dataset_path
        Path to a dataset file

    """
    def __init__(self, dataset_path: str):
        df = pd.read_csv(dataset_path)

        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        target = 'recidivism'
        numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                               'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
```


```python
dataset = CompasDataset(dataset_path=os.path.join('virny', 'data', 'COMPAS.csv'))
dataset.X_data[dataset.X_data.columns[:6]].head()
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
      <th>age</th>
      <th>juv_fel_count</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>priors_count</th>
      <th>race</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>0.0</td>
      <td>-2.340451</td>
      <td>1.0</td>
      <td>-15.010999</td>
      <td>African-American</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>Caucasian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>Caucasian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6.000000</td>
      <td>African-American</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.513697</td>
      <td>Caucasian</td>
    </tr>
  </tbody>
</table>
</div>



### Create a config object

`run_metrics_computation_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: a name of your dataset; it will be used to name files with metrics.

* **test_set_fraction**: the fraction from the whole dataset in the range [0.0 - 1.0] to create a test set.

* **bootstrap_fraction**: the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: the number of estimators for bootstrap to compute subgroup variance metrics.

* **runs_seed_lst**: a list of seeds for each run; the number of runs is derived based on the length of this list. For example, if your runs_seed_lst is [100, 200], this means that for the first run, the interface will use 100 seed, and the code logic will increment this seed for each model (101 for the first model in models_config, 102 for the second model, etc.).

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_compas_config.yaml')
config_yaml_content = """
dataset_name: COMPAS
test_set_fraction: 0.2
bootstrap_fraction: 0.8
n_estimators: 100
sensitive_attributes_dct: {'sex': 0, 'race': 'Caucasian', 'sex&race': None}
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)
```


```python
config = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results',
                                     f'{config.dataset_name}_Metrics_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}')
```

### Create a models config

**models_config** is a Python dictionary, where keys are model names and values are initialized models for analysis


```python
models_config = {
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                     max_depth=20,
                                                     max_features=0.6,
                                                     min_samples_split=0.1),
    'LogisticRegression': LogisticRegression(C=1,
                                             max_iter=50,
                                             penalty='l2',
                                             solver='newton-cg'),
#     'RandomForestClassifier': RandomForestClassifier(max_depth=4,
#                                                      max_features=0.6,
#                                                      min_samples_leaf=1,
#                                                      n_estimators=50),
#     'XGBClassifier': XGBClassifier(learning_rate=0.1,
#                                    max_depth=5,
#                                    n_estimators=20),
}
```

## Subgroup Metrics Computation

After the variables are input to a user interface, the interface creates a **generic pipeline** based on the input dataset class to hide preprocessing complexity and provide handy attributes and methods for different types of model analysis. Later this generic pipeline is used in subgroup analyzers that compute different sets of metrics. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Statistical Bias Analyzer**, but it is easily extensible to any other analyzers. When the variance and bias analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
subgroup_metrics_dct = run_metrics_computation_with_config(dataset, config, models_config, SAVE_RESULTS_DIR_PATH)
```

A lot of progress logs ...



Look at several columns in top rows of computed metrics


```python
sample_model_metrics_df = subgroup_metrics_dct[list(models_config.keys())[0]]
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
      <th>sex_priv</th>
      <th>sex_dis</th>
      <th>race_priv</th>
      <th>race_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.520361</td>
      <td>0.584853</td>
      <td>0.502702</td>
      <td>0.600694</td>
      <td>0.471180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.095816</td>
      <td>0.108234</td>
      <td>0.092415</td>
      <td>0.092157</td>
      <td>0.098056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.124709</td>
      <td>0.146045</td>
      <td>0.118866</td>
      <td>0.120969</td>
      <td>0.126999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.259691</td>
      <td>0.000000</td>
      <td>0.255675</td>
      <td>0.000000</td>
      <td>0.269280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.166510</td>
      <td>0.180292</td>
      <td>0.162736</td>
      <td>0.158829</td>
      <td>0.171212</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.679943</td>
      <td>0.686300</td>
      <td>0.678203</td>
      <td>0.707905</td>
      <td>0.662824</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.765833</td>
      <td>0.735771</td>
      <td>0.774065</td>
      <td>0.770623</td>
      <td>0.762901</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.670732</td>
      <td>0.552632</td>
      <td>0.692308</td>
      <td>0.544828</td>
      <td>0.723343</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.730496</td>
      <td>0.794702</td>
      <td>0.707022</td>
      <td>0.843750</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.684647</td>
      <td>0.575342</td>
      <td>0.704156</td>
      <td>0.663866</td>
      <td>0.691460</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.329268</td>
      <td>0.447368</td>
      <td>0.307692</td>
      <td>0.455172</td>
      <td>0.276657</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.269504</td>
      <td>0.205298</td>
      <td>0.292978</td>
      <td>0.156250</td>
      <td>0.363636</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.702652</td>
      <td>0.713656</td>
      <td>0.699638</td>
      <td>0.735661</td>
      <td>0.682443</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.677618</td>
      <td>0.563758</td>
      <td>0.698182</td>
      <td>0.598485</td>
      <td>0.707042</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.456439</td>
      <td>0.321586</td>
      <td>0.493366</td>
      <td>0.296758</td>
      <td>0.554198</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>0.979675</td>
      <td>0.960526</td>
      <td>0.983173</td>
      <td>0.820690</td>
      <td>1.046110</td>
    </tr>
  </tbody>
</table>
</div>



## Group Metrics Composition

**Metrics Composer** is responsible for this second stage of the model audit. Currently, it computes our custom group statistical bias and variance metrics, but extending it for new group metrics is very simple. We noticed that more and more group metrics have appeared during the last decade, but most of them are based on the same subgroup metrics. Hence, such a separation of subgroup and group metrics computation allows one to experiment with different combinations of subgroup metrics and avoid subgroup metrics recomputation for a new set of grouped metrics.


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

## Metrics Visualization and Reporting

**Metrics Visualizer** provides metrics visualization and reporting functionality. It unifies different preprocessing methods for result metrics and creates various data formats required for visualizations. Hence, users can simply call methods of the Metrics Visualizer class and get custom plots for diverse metrics analysis. Additionally, these plots could be collected in an HTML report with comments for user convenience and future reference.


```python
visualizer = MetricsVisualizer(models_metrics_dct, models_composed_metrics_df, config.dataset_name,
                               model_names=list(models_config.keys()),
                               sensitive_attributes_dct=config.sensitive_attributes_dct)
```


```python
visualizer.visualize_overall_metrics(
    metrics_names=['TPR', 'PPV', 'Accuracy', 'F1', 'Selection-Rate',
                   'Per_Sample_Accuracy', 'Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    x_label="Overall Metrics"
)
```


    
![png](One_Run_Multiple_Models_Interface_Use_Case_files/One_Run_Multiple_Models_Interface_Use_Case_34_0.png)
    


Below is an example of an interactive plot. It requires that you run the below cell in Jupyter in the browser.

You can use this plot to compare any pair of bias and variance metrics for all models.


```python
# visualizer.create_bias_variance_interactive_bar_chart()
```

Create an analysis report. It includes correspondent visualizations and explanations for your result metrics.


```python
visualizer.create_html_report(report_type=ReportType.ONE_RUN_MULTIPLE_MODELS,
                              dataset_name=config.dataset_name,
                              report_save_path=os.path.join(ROOT_DIR, "results", "reports"))
```


App saved to ./docs/examples/results/reports/COMPAS_Metrics_Report_20230202__105644.html
