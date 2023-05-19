# Single Run Multiple Models Interface Usage

In this example, we are going to audit 5 models for stability and fairness, visualize metrics, and create an analysis report. We will use `run_metrics_computation_with_config` interface that will execute the auditing pipeline for all defined models. For that, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Make group metrics composition

* Create metrics visualizations and an analysis report

## Import dependencies


```python
import os
import pathlib
import pandas as pd
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from virny.user_interfaces.metrics_computation_interfaces import run_metrics_computation_with_config
from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.datasets.base import BaseDataLoader
from virny.configs.constants import ReportType
```

## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **dataset class** that is a wrapper above the user’s raw dataset that includes its descriptive attributes like a target column, numerical columns, categorical columns, etc. This class must be inherited from the BaseDataset class, which was created for user convenience.

* A **config yaml** that is a file with configuration parameters for different user interfaces for metrics computation.

* Finally, a **models config** that is a Python dictionary, where keys are model names and values are initialized models for analysis. This dictionary helps conduct audits of multiple models for one or multiple runs and analyze different types of models.


```python
DATASET_SPLIT_SEED = 42
TEST_SET_FRACTION = 0.2
```

### Create a config object

`run_metrics_computation_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: the number of estimators for bootstrap to compute subgroup stability metrics.

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
age_range = [i for i in range(30, 41)]
config_yaml_content = \
f"""
dataset_name: Folktables_GA_2018
bootstrap_fraction: 0.8
n_estimators: 20  # Better to input the higher number of estimators than 100; this is only for this use case example
sensitive_attributes_dct: {{'SEX': '1', 'RAC1P': '1', 'AGEP': {age_range}, 'SEX & RAC1P & AGEP': None}}
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)
```


```python
config = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', f'{config.dataset_name}_Metrics_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}')
```

### Preprocess the dataset and create a BaseFlowDataset class

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


class ACSEmploymentDataset(BaseDataLoader):
    def __init__(self, state, year, root_dir=None, with_nulls=False, optimize=True, subsample=None):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_dir = pathlib.Path(__file__).parent if root_dir is None else root_dir
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_dir
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
            full_df=optimized_X_data,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=optimized_X_data,
            y_data=y_data,
            columns_with_nulls=columns_with_nulls,
        )
```


```python
data_loader = ACSEmploymentDataset(state=['GA'], year=2018, root_dir=os.path.join('virny', 'datasets'), with_nulls=False, subsample=20_000)
data_loader.X_data[data_loader.X_data.columns[:9]].head()
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
      <th>8341</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3461</th>
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
      <th>16117</th>
      <td>1</td>
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
      <th>30054</th>
      <td>5</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31512</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
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
base_flow_dataset = preprocess_dataset(data_loader, column_transformer, TEST_SET_FRACTION, DATASET_SPLIT_SEED)
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

After the variables are input to a user interface, the interface uses subgroup analyzers to compute different sets of metrics for each privileged and disprivileged subgroup. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
subgroup_metrics_dct = run_metrics_computation_with_config(base_flow_dataset, config, models_config, SAVE_RESULTS_DIR_PATH, run_seed=100)
```

A lot of logs...


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
      <td>0.550842</td>
      <td>0.527410</td>
      <td>0.573109</td>
      <td>0.546446</td>
      <td>0.559491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.058951</td>
      <td>0.063070</td>
      <td>0.055037</td>
      <td>0.056289</td>
      <td>0.064189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.076933</td>
      <td>0.083406</td>
      <td>0.070782</td>
      <td>0.074419</td>
      <td>0.081879</td>
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
      <td>0.054613</td>
      <td>0.053455</td>
      <td>0.055714</td>
      <td>0.049903</td>
      <td>0.063880</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.806900</td>
      <td>0.838969</td>
      <td>0.776426</td>
      <td>0.811859</td>
      <td>0.797144</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.924200</td>
      <td>0.922473</td>
      <td>0.925841</td>
      <td>0.930128</td>
      <td>0.912537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.833240</td>
      <td>0.836423</td>
      <td>0.829909</td>
      <td>0.819124</td>
      <td>0.866171</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.801994</td>
      <td>0.859496</td>
      <td>0.751489</td>
      <td>0.819613</td>
      <td>0.771605</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.773692</td>
      <td>0.841009</td>
      <td>0.713445</td>
      <td>0.803125</td>
      <td>0.715822</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.166760</td>
      <td>0.163577</td>
      <td>0.170091</td>
      <td>0.180876</td>
      <td>0.133829</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.198006</td>
      <td>0.140504</td>
      <td>0.248511</td>
      <td>0.180387</td>
      <td>0.228395</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.816000</td>
      <td>0.848640</td>
      <td>0.784983</td>
      <td>0.819382</td>
      <td>0.809347</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.802363</td>
      <td>0.838710</td>
      <td>0.767282</td>
      <td>0.811045</td>
      <td>0.783852</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.482750</td>
      <td>0.467932</td>
      <td>0.496831</td>
      <td>0.482655</td>
      <td>0.482938</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>1.076966</td>
      <td>0.994547</td>
      <td>1.163242</td>
      <td>1.019920</td>
      <td>1.210037</td>
    </tr>
  </tbody>
</table>
</div>



## Group Metrics Composition


```python
models_metrics_dct = read_model_metric_dfs(SAVE_RESULTS_DIR_PATH, model_names=list(models_config.keys()))
```

**Metrics Composer** is responsible for this second stage of the model audit. Currently, it computes our custom group fairness and stability metrics, but extending it for new group metrics is very simple. We noticed that more and more group metrics have appeared during the last decade, but most of them are based on the same subgroup metrics. Hence, such a separation of subgroup and group metrics computation allows one to experiment with different combinations of subgroup metrics and avoid subgroup metrics recomputation for a new set of grouped metrics.


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
    metrics_title="Error Metrics"
)
```





<div id="altair-viz-460ccc61131b4c098dc606910c221dab"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-460ccc61131b4c098dc606910c221dab") {
      outputDiv = document.getElementById("altair-viz-460ccc61131b4c098dc606910c221dab");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-87cbff61ec39e58009b395aa04042b88"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Error Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-87cbff61ec39e58009b395aa04042b88": [{"overall": 0.816, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8023630504833512, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.7736923873640601, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 1.0769659788064696, "metric": "Positive-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.48275, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8332403792526492, "metric": "TPR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.80625, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.7941567065073041, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.7581135902636917, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 1.0998326826547686, "metric": "Positive-Rate", "model_name": "LogisticRegression"}, {"overall": 0.493, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.833798103736754, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.7995, "metric": "Accuracy", "model_name": "KNeighborsClassifier"}, {"overall": 0.796239837398374, "metric": "F1", "model_name": "KNeighborsClassifier"}, {"overall": 0.731217918805413, "metric": "PPV", "model_name": "KNeighborsClassifier"}, {"overall": 1.1952035694366985, "metric": "Positive-Rate", "model_name": "KNeighborsClassifier"}, {"overall": 0.53575, "metric": "Selection-Rate", "model_name": "KNeighborsClassifier"}, {"overall": 0.8739542665923034, "metric": "TPR", "model_name": "KNeighborsClassifier"}, {"overall": 0.8235, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.8157620041753654, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.7665522314860226, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 1.1372002230897935, "metric": "Positive-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.50975, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.871723368655884, "metric": "TPR", "model_name": "RandomForestClassifier"}, {"overall": 0.824, "metric": "Accuracy", "model_name": "XGBClassifier"}, {"overall": 0.8157068062827225, "metric": "F1", "model_name": "XGBClassifier"}, {"overall": 0.7686235816477553, "metric": "PPV", "model_name": "XGBClassifier"}, {"overall": 1.130507529280535, "metric": "Positive-Rate", "model_name": "XGBClassifier"}, {"overall": 0.50675, "metric": "Selection-Rate", "model_name": "XGBClassifier"}, {"overall": 0.8689347462353597, "metric": "TPR", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    metrics_title="Variance Metrics"
)
```





<div id="altair-viz-0cd1bcfdbb48406e97938866c29c07cd"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0cd1bcfdbb48406e97938866c29c07cd") {
      outputDiv = document.getElementById("altair-viz-0cd1bcfdbb48406e97938866c29c07cd");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-0677c045a64519b5d1edef4633bf201a"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Variance Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-0677c045a64519b5d1edef4633bf201a": [{"overall": 0.9230673119963418, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9453868421052632, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9242, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9410487574275062, "metric": "Std", "model_name": "DecisionTreeClassifier"}, {"overall": 0.971266848404931, "metric": "IQR", "model_name": "LogisticRegression"}, {"overall": 0.968453947368421, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.956925, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.9774634882432649, "metric": "Std", "model_name": "LogisticRegression"}, {"overall": 0.9128025, "metric": "IQR", "model_name": "KNeighborsClassifier"}, {"overall": 0.9124986842105264, "metric": "Jitter", "model_name": "KNeighborsClassifier"}, {"overall": 0.882375, "metric": "Label_Stability", "model_name": "KNeighborsClassifier"}, {"overall": 0.9314640215093075, "metric": "Std", "model_name": "KNeighborsClassifier"}, {"overall": 0.9592260197764773, "metric": "IQR", "model_name": "RandomForestClassifier"}, {"overall": 0.9679039473684211, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.956925, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}, {"overall": 0.9674595365660505, "metric": "Std", "model_name": "RandomForestClassifier"}, {"overall": 0.9706980982869864, "metric": "IQR", "model_name": "XGBClassifier"}, {"overall": 0.9799368421052632, "metric": "Jitter", "model_name": "XGBClassifier"}, {"overall": 0.9734, "metric": "Label_Stability", "model_name": "XGBClassifier"}, {"overall": 0.9767059218138457, "metric": "Std", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_model_rank_heatmaps(
    metrics_lst=[
        # Group fairness metrics
        'Equalized_Odds_TPR',
        'Equalized_Odds_FPR',
        'Disparate_Impact',
        'Statistical_Parity_Difference',
        'Accuracy_Parity',
        # Group stability metrics
        'Label_Stability_Ratio',
        'IQR_Parity',
        'Std_Parity',
        'Std_Ratio',
        'Jitter_Parity',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
)
```


    
![png](Single_Run_Multiple_Models_Interface_Use_Case_files/Single_Run_Multiple_Models_Interface_Use_Case_39_0.png)



Create an analysis report. It includes correspondent visualizations and details about your result metrics.


```python
visualizer.create_html_report(report_type=ReportType.ONE_RUN_MULTIPLE_MODELS,
                              report_save_path=os.path.join(ROOT_DIR, "results", "reports"))
```


App saved to ./docs/examples/results/reports/Folktables_GA_2018_Metrics_Report_20230519__211628.html
