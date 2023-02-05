# Multiple Runs Interface Usage

In this example, we are going to audit 4 models for stability and fairness, visualize metrics, and create an analysis report. To get better analysis accuracy, we will use `compute_metrics_multiple_runs` interface that will make multiple runs per model. For that, we will need to do the next steps:

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

from virny.user_interfaces.metrics_computation_interfaces import compute_metrics_multiple_runs
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
class CompasWithoutSensitiveAttrsDataset(BaseDataset):
    """
    Dataset class for COMPAS dataset that does not contain sensitive attributes among feature columns
     to test blind classifiers

    Parameters
    ----------
    dataset_path
        Path to a dataset file

    """
    def __init__(self, dataset_path: str):
        # Read a dataset
        df = pd.read_csv(dataset_path)

        # Initial data types transformation
        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        # Define params
        target = 'recidivism'
        numerical_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count']
        categorical_columns = ['age_cat_25 - 45', 'age_cat_Greater than 45','age_cat_Less than 25',
                                    'c_charge_degree_F', 'c_charge_degree_M']
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
```


```python
dataset = CompasWithoutSensitiveAttrsDataset(dataset_path=os.path.join('virny', 'datasets', 'COMPAS.csv'))
dataset.X_data[dataset.X_data.columns[:5]].head()
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
      <th>juv_fel_count</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>priors_count</th>
      <th>age_cat_25 - 45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-2.340451</td>
      <td>1.0</td>
      <td>-15.010999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.513697</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Create a config object

`compute_metrics_multiple_runs` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: a name of your dataset; it will be used to name files with metrics.

* **test_set_fraction**: the fraction from the whole dataset in the range [0.0 - 1.0] to create a test set.

* **bootstrap_fraction**: the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: the number of estimators for bootstrap to compute subgroup variance metrics.

* **runs_seed_lst**: a list of seeds for each run; the number of runs is derived based on the length of this list. For example, if your runs_seed_lst is [100, 200], this means that for the first run, the interface will use 100 seed, and the code logic will increment this seed for each model (101 for the first model in models_config, 102 for the second model, etc.).

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_compas_config.yaml')
config_yaml_content = \
"""dataset_name: COMPAS_Without_Sensitive_Attributes
test_set_fraction: 0.2
bootstrap_fraction: 0.8
n_estimators: 50  # Better to input the higher number of estimators than 100; this is only for this use case example
runs_seed_lst: [100, 200, 300, 400, 500]
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
    'RandomForestClassifier': RandomForestClassifier(max_depth=4,
                                                     max_features=0.6,
                                                     min_samples_leaf=1,
                                                     n_estimators=50),
    'XGBClassifier': XGBClassifier(learning_rate=0.1,
                                   max_depth=5,
                                   n_estimators=20),
}
```

## Subgroup Metrics Computation

After the variables are input to a user interface, the interface creates a **generic pipeline** based on the input dataset class to hide preprocessing complexity and provide handy attributes and methods for different types of model analysis. Later this generic pipeline is used in subgroup analyzers that compute different sets of metrics. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Statistical Bias Analyzer**, but it is easily extensible to any other analyzers. When the variance and bias analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
multiple_run_metrics_dct = compute_metrics_multiple_runs(dataset, config, models_config, SAVE_RESULTS_DIR_PATH, debug_mode=False)
```

A lot of progress logs ...



Look at several columns in top rows of computed metrics


```python
sample_model_metrics_df = multiple_run_metrics_dct[list(models_config.keys())[0]]
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
      <td>0.525182</td>
      <td>0.558977</td>
      <td>0.517335</td>
      <td>0.588704</td>
      <td>0.482060</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.071942</td>
      <td>0.078615</td>
      <td>0.070393</td>
      <td>0.071601</td>
      <td>0.072175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.089570</td>
      <td>0.096166</td>
      <td>0.088039</td>
      <td>0.091559</td>
      <td>0.088220</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.193735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.116624</td>
      <td>0.129681</td>
      <td>0.113592</td>
      <td>0.108870</td>
      <td>0.121888</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.661913</td>
      <td>0.681508</td>
      <td>0.657363</td>
      <td>0.662061</td>
      <td>0.661812</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.843598</td>
      <td>0.826131</td>
      <td>0.847655</td>
      <td>0.851710</td>
      <td>0.838092</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.632444</td>
      <td>0.571429</td>
      <td>0.642686</td>
      <td>0.477987</td>
      <td>0.707317</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.725835</td>
      <td>0.751938</td>
      <td>0.718182</td>
      <td>0.791045</td>
      <td>0.667774</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.663793</td>
      <td>0.555556</td>
      <td>0.683673</td>
      <td>0.575758</td>
      <td>0.698795</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.367556</td>
      <td>0.428571</td>
      <td>0.357314</td>
      <td>0.522013</td>
      <td>0.292683</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.274165</td>
      <td>0.248062</td>
      <td>0.281818</td>
      <td>0.208955</td>
      <td>0.332226</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.682765</td>
      <td>0.688442</td>
      <td>0.681447</td>
      <td>0.674473</td>
      <td>0.688394</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.647739</td>
      <td>0.563380</td>
      <td>0.662546</td>
      <td>0.522337</td>
      <td>0.703030</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.439394</td>
      <td>0.361809</td>
      <td>0.457410</td>
      <td>0.309133</td>
      <td>0.527822</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>0.952772</td>
      <td>1.028571</td>
      <td>0.940048</td>
      <td>0.830189</td>
      <td>1.012195</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.524371</td>
      <td>0.560791</td>
      <td>0.515861</td>
      <td>0.591471</td>
      <td>0.481953</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.069567</td>
      <td>0.077170</td>
      <td>0.067791</td>
      <td>0.070342</td>
      <td>0.069077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.085142</td>
      <td>0.092375</td>
      <td>0.083452</td>
      <td>0.087349</td>
      <td>0.083748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.222123</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
visualizer.create_boxes_and_whiskers_for_models_multiple_runs(metrics_lst=['Std', 'IQR', 'Jitter', 'FNR','FPR'])
```


    
![png](Multiple_Runs_Interface_Use_Case_files/Multiple_Runs_Interface_Use_Case_34_0.png)
    



```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['TPR', 'PPV', 'Accuracy', 'F1', 'Selection-Rate', 'Positive-Rate'],
    metrics_title="Bias Metrics"
)
```





<div id="altair-viz-cd6765fe259844c4ae70e3afd8c6920b"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-cd6765fe259844c4ae70e3afd8c6920b") {
      outputDiv = document.getElementById("altair-viz-cd6765fe259844c4ae70e3afd8c6920b");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-fd7b2ee95de43239fadf41b8dcaf5e01"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Bias Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-fd7b2ee95de43239fadf41b8dcaf5e01": [{"overall": 0.6801136363636363, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6504402880373518, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6627372557361907, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9644934963129117, "metric": "Positive-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.4492424242424242, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6388269611776994, "metric": "TPR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6600378787878788, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.6199889298320485, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.653641138239119, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 0.9033205755153567, "metric": "Positive-Rate", "model_name": "LogisticRegression"}, {"overall": 0.4246212121212121, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.5899478361231674, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.6676136363636364, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.6251645861816336, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.6827393347950576, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 0.8496145297124584, "metric": "Positive-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.40738636363636366, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.5779741009599183, "metric": "TPR", "model_name": "RandomForestClassifier"}, {"overall": 0.675, "metric": "Accuracy", "model_name": "XGBClassifier"}, {"overall": 0.6472314731167365, "metric": "F1", "model_name": "XGBClassifier"}, {"overall": 0.6713400255768536, "metric": "PPV", "model_name": "XGBClassifier"}, {"overall": 0.9309059679873537, "metric": "Positive-Rate", "model_name": "XGBClassifier"}, {"overall": 0.4441287878787879, "metric": "Selection-Rate", "model_name": "XGBClassifier"}, {"overall": 0.6248878586439098, "metric": "TPR", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    metrics_title="Variance Metrics"
)
```





<div id="altair-viz-da6526e6fc604e92b32e6629a0c90200"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-da6526e6fc604e92b32e6629a0c90200") {
      outputDiv = document.getElementById("altair-viz-da6526e6fc604e92b32e6629a0c90200");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-bfeaedb2591c451cc64e705317b3cfd3"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Variance Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-bfeaedb2591c451cc64e705317b3cfd3": [{"overall": 0.9129468994841241, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8792411873840446, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8411060606060605, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.926668761176367, "metric": "Std", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9723001277107443, "metric": "IQR", "model_name": "LogisticRegression"}, {"overall": 0.9548531230674089, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.9378484848484849, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.9792936156402017, "metric": "Std", "model_name": "LogisticRegression"}, {"overall": 0.9480487270117715, "metric": "IQR", "model_name": "RandomForestClassifier"}, {"overall": 0.9355575139146568, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.9119696969696969, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}, {"overall": 0.9606283310736627, "metric": "Std", "model_name": "RandomForestClassifier"}, {"overall": 0.9385613391147645, "metric": "IQR", "model_name": "XGBClassifier"}, {"overall": 0.9057578849721707, "metric": "Jitter", "model_name": "XGBClassifier"}, {"overall": 0.8686969696969697, "metric": "Label_Stability", "model_name": "XGBClassifier"}, {"overall": 0.9529354810714722, "metric": "Std", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
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


    
![png](Multiple_Runs_Interface_Use_Case_files/Multiple_Runs_Interface_Use_Case_37_0.png)
    



    
![png](Multiple_Runs_Interface_Use_Case_files/Multiple_Runs_Interface_Use_Case_37_1.png)
    


Create an analysis report. It includes correspondent visualizations and details about your result metrics.


```python
visualizer.create_html_report(report_type=ReportType.MULTIPLE_RUNS_MULTIPLE_MODELS,
                              report_save_path=os.path.join(ROOT_DIR, "results", "reports"))
```


App saved to ./docs/examples/results/reports/COMPAS_Without_Sensitive_Attributes_Metrics_Report_20230205__171832.html

