# Single Run Single Model Interface Usage

In this example, we are going to audit 1 model for stability and fairness, visualize metrics, and create an analysis report. We will use `compute_model_metrics_with_config` interface that will conduct the auditing pipeline for this model. For that, we will need to do next steps:

* Initialize input variables

* Compute subgroup metrics

* Make group metrics composition

* Create metrics visualizations and an analysis report

## Import dependencies


```python
import os
import pandas as pd
from datetime import datetime, timezone

from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from virny.user_interfaces.metrics_computation_interfaces import compute_model_metrics_with_config
from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.datasets.base import BaseDataLoader
from virny.preprocessing.basic_preprocessing import preprocess_dataset
```

## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **dataset class** that is a wrapper above the user’s raw dataset that includes its descriptive attributes like a target column, numerical columns, categorical columns, etc. This class must be inherited from the BaseDataset class, which was created for user convenience.

* A **config yaml** that is a file with configuration parameters for different user interfaces for metrics computation.

* Finally, a **models config** that is a Python dictionary, where keys are model names and values are initialized models for analysis. This dictionary helps conduct audits of multiple models for one or multiple runs and analyze different types of models.


```python
TEST_SET_FRACTION = 0.2
DATASET_SPLIT_SEED = 42
```

### Create a Dataset class

Based on the BaseDataset class, your **dataset class** should include the following attributes:

* **Obligatory attributes**: dataset, target, features, numerical_columns, categorical_columns

* **Optional attributes**: X_data, y_data, columns_with_nulls

For more details, please refer to the library documentation.


```python
class CompasDataset(BaseDataLoader):
    """
    Dataset class for COMPAS dataset that contains sensitive attributes among feature columns.

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset

    """
    def __init__(self, dataset_path, subsample_size=None):
        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size)

        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        target = 'recidivism'
        numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                               'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
```


```python
data_loader = CompasDataset(dataset_path=os.path.join('virny', 'datasets', 'COMPAS.csv'))
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
      <th>age</th>
      <th>juv_fel_count</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>priors_count</th>
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
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.513697</td>
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

### Create a config object

`compute_model_metrics_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: the number of estimators for bootstrap to compute subgroup stability metrics.

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: COMPAS
bootstrap_fraction: 0.8
n_estimators: 100
sensitive_attributes_dct: {'sex': 0, 'age': 25, 'race': 'Caucasian', 'sex&race': None}
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
}
```

## Subgroup Metrics Computation

After the variables are input to a user interface, the interface uses subgroup analyzers to compute different sets of metrics for each privileged and disprivileged subgroup. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_df = compute_model_metrics_with_config(models_config['DecisionTreeClassifier'], 'DecisionTreeClassifier', base_flow_dataset,
                                               config, SAVE_RESULTS_DIR_PATH, save_results=True, verbose=0)
```


    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


Look at several columns in top rows of computed metrics


```python
metrics_df[metrics_df.columns[:6]].head(20)
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
      <th>age_priv</th>
      <th>age_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.524564</td>
      <td>0.622992</td>
      <td>0.499986</td>
      <td>0.471650</td>
      <td>0.527084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.091872</td>
      <td>0.101903</td>
      <td>0.089368</td>
      <td>0.087381</td>
      <td>0.092086</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.120359</td>
      <td>0.133644</td>
      <td>0.117041</td>
      <td>0.110779</td>
      <td>0.120815</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.250195</td>
      <td>0.230957</td>
      <td>0.254999</td>
      <td>0.000000</td>
      <td>0.252084</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.159485</td>
      <td>0.146778</td>
      <td>0.162659</td>
      <td>0.133039</td>
      <td>0.160745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.676591</td>
      <td>0.690616</td>
      <td>0.673089</td>
      <td>0.676667</td>
      <td>0.676587</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.778447</td>
      <td>0.796019</td>
      <td>0.774059</td>
      <td>0.816667</td>
      <td>0.776627</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.626327</td>
      <td>0.306667</td>
      <td>0.686869</td>
      <td>0.625000</td>
      <td>0.626398</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.731624</td>
      <td>0.882353</td>
      <td>0.685969</td>
      <td>0.625000</td>
      <td>0.736185</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.652655</td>
      <td>0.589744</td>
      <td>0.658596</td>
      <td>0.625000</td>
      <td>0.654206</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.373673</td>
      <td>0.693333</td>
      <td>0.313131</td>
      <td>0.375000</td>
      <td>0.373602</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.268376</td>
      <td>0.117647</td>
      <td>0.314031</td>
      <td>0.375000</td>
      <td>0.263815</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.684659</td>
      <td>0.677725</td>
      <td>0.686391</td>
      <td>0.625000</td>
      <td>0.687500</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.639220</td>
      <td>0.403509</td>
      <td>0.672435</td>
      <td>0.625000</td>
      <td>0.640000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.428030</td>
      <td>0.184834</td>
      <td>0.488757</td>
      <td>0.500000</td>
      <td>0.424603</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>0.959660</td>
      <td>0.520000</td>
      <td>1.042929</td>
      <td>1.000000</td>
      <td>0.957494</td>
    </tr>
  </tbody>
</table>
</div>



## Group Metrics Composition

**Metrics Composer** is responsible for this second stage of the model audit. Currently, it computes our custom group fairness and stability metrics, but extending it for new group metrics is very simple. We noticed that more and more group metrics have appeared during the last decade, but most of them are based on the same subgroup metrics. Hence, such a separation of subgroup and group metrics computation allows one to experiment with different combinations of subgroup metrics and avoid subgroup metrics recomputation for a new set of grouped metrics.


```python
models_metrics_dct = read_model_metric_dfs(SAVE_RESULTS_DIR_PATH, model_names=['DecisionTreeClassifier'])
```


```python
metrics_composer = MetricsComposer(models_metrics_dct, config.sensitive_attributes_dct)
```

Compute composed metrics


```python
models_composed_metrics_df = metrics_composer.compose_metrics()
```


```python
models_composed_metrics_df.head(20)
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
      <th>sex</th>
      <th>age</th>
      <th>race</th>
      <th>sex&amp;race</th>
      <th>Model_Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equalized_Odds_TPR</td>
      <td>0.380202</td>
      <td>0.001398</td>
      <td>0.267700</td>
      <td>0.576180</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Equalized_Odds_FPR</td>
      <td>0.196384</td>
      <td>-0.111185</td>
      <td>0.142322</td>
      <td>0.285020</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Equalized_Odds_FNR</td>
      <td>-0.380202</td>
      <td>-0.001398</td>
      <td>-0.267700</td>
      <td>-0.576180</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Disparate_Impact</td>
      <td>2.005633</td>
      <td>0.957494</td>
      <td>1.314176</td>
      <td>2.864909</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Statistical_Parity_Difference</td>
      <td>0.522929</td>
      <td>-0.042506</td>
      <td>0.247921</td>
      <td>0.713053</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Accuracy_Parity</td>
      <td>0.008665</td>
      <td>0.062500</td>
      <td>0.009730</td>
      <td>0.038441</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability_Ratio</td>
      <td>0.972413</td>
      <td>0.950972</td>
      <td>1.017002</td>
      <td>0.987639</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>7</th>
      <td>IQR_Parity</td>
      <td>-0.016603</td>
      <td>0.010036</td>
      <td>-0.005627</td>
      <td>-0.021093</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Std_Parity</td>
      <td>-0.012536</td>
      <td>0.004706</td>
      <td>-0.001453</td>
      <td>-0.011836</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Std_Ratio</td>
      <td>0.876985</td>
      <td>1.053852</td>
      <td>0.984339</td>
      <td>0.882427</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Jitter_Parity</td>
      <td>0.015881</td>
      <td>0.027706</td>
      <td>-0.005745</td>
      <td>0.008915</td>
      <td>DecisionTreeClassifier</td>
    </tr>
  </tbody>
</table>
</div>



## Metrics Visualization

**Metrics Visualizer** provides metrics visualization and reporting functionality. It unifies different preprocessing methods for result metrics and creates various data formats required for visualizations. Hence, users can simply call methods of the Metrics Visualizer class and get custom plots for diverse metrics analysis.


```python
visualizer = MetricsVisualizer(models_metrics_dct, models_composed_metrics_df, config.dataset_name,
                               model_names=['DecisionTreeClassifier'],
                               sensitive_attributes_dct=config.sensitive_attributes_dct)
```


```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['TPR', 'PPV', 'Accuracy', 'F1', 'Selection-Rate', 'Positive-Rate'],
    metrics_title="Error Metrics"
)
```





<div id="altair-viz-04be312eed21479ab2afc70b43b5857d"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-04be312eed21479ab2afc70b43b5857d") {
      outputDiv = document.getElementById("altair-viz-04be312eed21479ab2afc70b43b5857d");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-02ce93f37d51ccd007459beeee9fdeab"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Error Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-02ce93f37d51ccd007459beeee9fdeab": [{"overall": 0.6846590909090909, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6392199349945829, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6526548672566371, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9596602972399152, "metric": "Positive-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.428030303030303, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6263269639065817, "metric": "TPR", "model_name": "DecisionTreeClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    metrics_title="Variance Metrics"
)
```





<div id="altair-viz-8882130ab0d9432e85bbdc06429fdd5d"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-8882130ab0d9432e85bbdc06429fdd5d") {
      outputDiv = document.getElementById("altair-viz-8882130ab0d9432e85bbdc06429fdd5d");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-c09ca468560a457675234299eb7aca53"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Variance Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-c09ca468560a457675234299eb7aca53": [{"overall": 0.8796414525570231, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8405146158555247, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.7784469696969698, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9081275471369314, "metric": "Std", "model_name": "DecisionTreeClassifier"}]}}, {"mode": "vega-lite"});
</script>
