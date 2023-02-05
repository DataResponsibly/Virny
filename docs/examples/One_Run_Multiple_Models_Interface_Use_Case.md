# One Run Multiple Models Interface Usage

In this example, we are going to audit 5 models for stability and fairness, visualize metrics, and create an analysis report. We will use `run_metrics_computation_with_config` interface that will execute the auditing pipeline for all defined models. For that, we will need to do the next steps:

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
from sklearn.neighbors import KNeighborsClassifier

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
import numpy as np
from folktables import ACSDataSource, ACSEmployment


def optimize_data_loading(data, categorical):
    """
    Optimizing the dataset size by downcasting categorical columns
    """
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data


class ACSEmploymentDataset(BaseDataset):
    def __init__(self, state, year, root_dir=os.path.join(os.getcwd(), 'virny', 'data'),
                 with_nulls=False, optimize=True, subsample=None):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if subsample is not None:
            acs_data = acs_data.sample(subsample)

        dataset = acs_data
        features = ACSEmployment.features
        target = ACSEmployment.target
        categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC','SCHL']
        numerical_columns = ['AGEP']

        if with_nulls is True:
            X_data = acs_data[features]
        else:
            X_data = acs_data[features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize:
            X_data = optimize_data_loading(X_data, categorical_columns)

        optimized_X_data = X_data[categorical_columns].astype('str')
        for col in numerical_columns:
            optimized_X_data[col] = X_data[col]
        y_data = acs_data[target].apply(lambda x: int(x == 1))

        columns_with_nulls = optimized_X_data.columns[optimized_X_data.isna().any().to_list()].to_list()

        super().__init__(
            pandas_df=dataset,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=optimized_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )
```


```python
dataset = ACSEmploymentDataset(state=['GA'], year=2018, root_dir=os.path.join('virny', 'datasets'), with_nulls=False, subsample=20_000)
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
      <th>MAR</th>
      <th>MIL</th>
      <th>ESP</th>
      <th>MIG</th>
      <th>DREM</th>
      <th>NATIVITY</th>
      <th>DIS</th>
      <th>DEAR</th>
      <th>DEYE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12327</th>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>99831</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42838</th>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13915</th>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37946</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
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

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_compas_config.yaml')
config_yaml_content = \
"""
dataset_name: Folktables_GA_2018
test_set_fraction: 0.2
bootstrap_fraction: 0.8
n_estimators: 20  # Better to input the higher number of estimators than 100; this is only for this use case example
sensitive_attributes_dct: {'SEX': '1', 'RAC1P': '1', 'SEX&RAC1P': None}
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
                                                     max_depth=10,
                                                     max_features=0.6,
                                                     min_samples_split=0.1),
    'LogisticRegression': LogisticRegression(C=1,
                                             max_iter=150,
                                             penalty='l2',
                                             solver='lbfgs'),
    'KNeighborsClassifier': KNeighborsClassifier(metric='minkowski',
                                                 n_neighbors=25,
                                                 weights='uniform'),
    'RandomForestClassifier': RandomForestClassifier(max_depth=10,
                                                     max_features=0.6,
                                                     min_samples_leaf=4,
                                                     n_estimators=100),
    'XGBClassifier': XGBClassifier(reg_lambda=1,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   n_estimators=100),
}
```

## Subgroup Metrics Computation

After the variables are input to a user interface, the interface creates a **generic pipeline** based on the input dataset class to hide preprocessing complexity and provide handy attributes and methods for different types of model analysis. Later this generic pipeline is used in subgroup analyzers that compute different sets of metrics. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Statistical Bias Analyzer**, but it is easily extensible to any other analyzers. When the variance and bias analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
subgroup_metrics_dct = run_metrics_computation_with_config(dataset, config, models_config, SAVE_RESULTS_DIR_PATH, run_seed=100)
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
      <th>SEX_priv</th>
      <th>SEX_dis</th>
      <th>RAC1P_priv</th>
      <th>RAC1P_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.553023</td>
      <td>0.528359</td>
      <td>0.576554</td>
      <td>0.542431</td>
      <td>0.573288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.052534</td>
      <td>0.053111</td>
      <td>0.051984</td>
      <td>0.052990</td>
      <td>0.051662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.059290</td>
      <td>0.058500</td>
      <td>0.060044</td>
      <td>0.059339</td>
      <td>0.059197</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.039426</td>
      <td>0.036432</td>
      <td>0.042283</td>
      <td>0.037529</td>
      <td>0.043056</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.813500</td>
      <td>0.850563</td>
      <td>0.778139</td>
      <td>0.810316</td>
      <td>0.819592</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.945100</td>
      <td>0.948643</td>
      <td>0.941720</td>
      <td>0.948306</td>
      <td>0.938966</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.858539</td>
      <td>0.866375</td>
      <td>0.849879</td>
      <td>0.852694</td>
      <td>0.871143</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.792127</td>
      <td>0.854808</td>
      <td>0.738739</td>
      <td>0.787352</td>
      <td>0.800487</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.760571</td>
      <td>0.839703</td>
      <td>0.687561</td>
      <td>0.768006</td>
      <td>0.745342</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.141461</td>
      <td>0.133625</td>
      <td>0.150121</td>
      <td>0.147306</td>
      <td>0.128857</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.207873</td>
      <td>0.145192</td>
      <td>0.261261</td>
      <td>0.212648</td>
      <td>0.199513</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.821000</td>
      <td>0.860215</td>
      <td>0.783586</td>
      <td>0.816901</td>
      <td>0.828842</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.806591</td>
      <td>0.852830</td>
      <td>0.760152</td>
      <td>0.808137</td>
      <td>0.803347</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.490750</td>
      <td>0.482335</td>
      <td>0.498779</td>
      <td>0.502094</td>
      <td>0.469046</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>1.128810</td>
      <td>1.031763</td>
      <td>1.236077</td>
      <td>1.110269</td>
      <td>1.168784</td>
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
visualizer.create_overall_metrics_bar_char(
    metrics_names=['TPR', 'PPV', 'Accuracy', 'F1', 'Selection-Rate', 'Positive-Rate'],
    metrics_title="Bias Metrics"
)
```





<div id="altair-viz-d7e87c5eab8946e69bc91c7cb154cee9"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-d7e87c5eab8946e69bc91c7cb154cee9") {
      outputDiv = document.getElementById("altair-viz-d7e87c5eab8946e69bc91c7cb154cee9");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-575e14539fc486b57643da4b324fb490"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Bias Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-575e14539fc486b57643da4b324fb490": [{"overall": 0.821, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8065910318746623, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.760570555272542, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 1.1288096607245544, "metric": "Positive-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.49075, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8585393904542841, "metric": "TPR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.81425, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.7980429464528405, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.7637877211238293, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 1.0939100739897554, "metric": "Positive-Rate", "model_name": "LogisticRegression"}, {"overall": 0.4805, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.8355150825270348, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.802, "metric": "Accuracy", "model_name": "KNeighborsClassifier"}, {"overall": 0.7942857142857143, "metric": "F1", "model_name": "KNeighborsClassifier"}, {"overall": 0.7298329355608592, "metric": "PPV", "model_name": "KNeighborsClassifier"}, {"overall": 1.1937321937321936, "metric": "Positive-Rate", "model_name": "KNeighborsClassifier"}, {"overall": 0.52375, "metric": "Selection-Rate", "model_name": "KNeighborsClassifier"}, {"overall": 0.8712250712250712, "metric": "TPR", "model_name": "KNeighborsClassifier"}, {"overall": 0.8245, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.816710182767624, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.7746409113422487, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 1.1148536720044175, "metric": "Positive-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.50475, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.8636112644947542, "metric": "TPR", "model_name": "RandomForestClassifier"}, {"overall": 0.82325, "metric": "Accuracy", "model_name": "XGBClassifier"}, {"overall": 0.8114163777007202, "metric": "F1", "model_name": "XGBClassifier"}, {"overall": 0.7674066599394551, "metric": "PPV", "model_name": "XGBClassifier"}, {"overall": 1.121675155631013, "metric": "Positive-Rate", "model_name": "XGBClassifier"}, {"overall": 0.4955, "metric": "Selection-Rate", "model_name": "XGBClassifier"}, {"overall": 0.8607809847198642, "metric": "TPR", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    metrics_title="Variance Metrics"
)
```





<div id="altair-viz-2ad8d8fe739b47b0ac593e6d58faaa4a"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-2ad8d8fe739b47b0ac593e6d58faaa4a") {
      outputDiv = document.getElementById("altair-viz-2ad8d8fe739b47b0ac593e6d58faaa4a");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-0468f7fb0b18f8eeab582006a4c557b9"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Variance Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-0468f7fb0b18f8eeab582006a4c557b9": [{"overall": 0.9407097641418828, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9605736842105264, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9451, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9474656966154313, "metric": "Std", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9706374366487484, "metric": "IQR", "model_name": "LogisticRegression"}, {"overall": 0.9681723684210527, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.956325, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.9774882126070372, "metric": "Std", "model_name": "LogisticRegression"}, {"overall": 0.9127925, "metric": "IQR", "model_name": "KNeighborsClassifier"}, {"overall": 0.9092486842105264, "metric": "Jitter", "model_name": "KNeighborsClassifier"}, {"overall": 0.877325, "metric": "Label_Stability", "model_name": "KNeighborsClassifier"}, {"overall": 0.931247713110665, "metric": "Std", "model_name": "KNeighborsClassifier"}, {"overall": 0.9586320605682214, "metric": "IQR", "model_name": "RandomForestClassifier"}, {"overall": 0.9681065789473685, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.957225, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}, {"overall": 0.9668485221316792, "metric": "Std", "model_name": "RandomForestClassifier"}, {"overall": 0.9723685530498624, "metric": "IQR", "model_name": "XGBClassifier"}, {"overall": 0.977603947368421, "metric": "Jitter", "model_name": "XGBClassifier"}, {"overall": 0.969275, "metric": "Label_Stability", "model_name": "XGBClassifier"}, {"overall": 0.9780136942863465, "metric": "Std", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_model_rank_heatmaps(
    metrics_lst=[
        # Group statistical bias metrics
        'Equalized_Odds_TPR',
        'Equalized_Odds_FPR',
        'Disparate_Impact',
        'Statistical_Parity_Difference',
        'Accuracy_Parity',
        # Group variance metrics
        'Label_Stability_Ratio',
        'IQR_Parity',
        'Std_Parity',
        'Std_Ratio',
        'Jitter_Parity',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
)
```


    
![png](One_Run_Multiple_Models_Interface_Use_Case_files/One_Run_Multiple_Models_Interface_Use_Case_36_0.png)
    



    
![png](One_Run_Multiple_Models_Interface_Use_Case_files/One_Run_Multiple_Models_Interface_Use_Case_36_1.png)
    


Create an analysis report. It includes correspondent visualizations and details about your result metrics.


```python
visualizer.create_html_report(report_type=ReportType.ONE_RUN_MULTIPLE_MODELS,
                              report_save_path=os.path.join(ROOT_DIR, "results", "reports"))
```


App saved to ./docs/examples/results/reports/Folktables_GA_2018_Metrics_Report_20230205__165240.html



```python

```
