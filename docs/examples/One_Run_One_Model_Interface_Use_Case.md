# One Run One Model Interface Usage

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

from virny.user_interfaces.metrics_computation_interfaces import compute_model_metrics_with_config
from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.custom_classes.base_dataset import BaseDataset
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
dataset = CompasDataset(dataset_path=os.path.join('virny', 'datasets', 'COMPAS.csv'))
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



### Create a config object

`compute_model_metrics_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: a name of your dataset; it will be used to name files with metrics.

* **test_set_fraction**: the fraction from the whole dataset in the range [0.0 - 1.0] to create a test set.

* **bootstrap_fraction**: the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: the number of estimators for bootstrap to compute subgroup variance metrics.

* **sensitive_attributes_dct**: a dictionary where keys are sensitive attribute names (including attribute intersections), and values are privileged values for these attributes. Currently, the library supports only intersections among two sensitive attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify privileged values for intersectional groups since they will be derived from privileged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.



```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_compas_config.yaml')
config_yaml_content = """
dataset_name: COMPAS
test_set_fraction: 0.2
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

After the variables are input to a user interface, the interface creates a **generic pipeline** based on the input dataset class to hide preprocessing complexity and provide handy attributes and methods for different types of model analysis. Later this generic pipeline is used in subgroup analyzers that compute different sets of metrics. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Statistical Bias Analyzer**, but it is easily extensible to any other analyzers. When the variance and bias analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_df = compute_model_metrics_with_config(models_config['DecisionTreeClassifier'], 'DecisionTreeClassifier', dataset,
                                               config, SAVE_RESULTS_DIR_PATH, save_results=True, debug_mode=False)
```

A lot of progress logs ...


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
      <td>0.528258</td>
      <td>0.603018</td>
      <td>0.510248</td>
      <td>0.464100</td>
      <td>0.531648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.093973</td>
      <td>0.109636</td>
      <td>0.090200</td>
      <td>0.079892</td>
      <td>0.094717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.117446</td>
      <td>0.141692</td>
      <td>0.111605</td>
      <td>0.102105</td>
      <td>0.118256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.252574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.159417</td>
      <td>0.183520</td>
      <td>0.153611</td>
      <td>0.172838</td>
      <td>0.158708</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.667992</td>
      <td>0.693415</td>
      <td>0.661868</td>
      <td>0.656604</td>
      <td>0.668594</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.778523</td>
      <td>0.740390</td>
      <td>0.787709</td>
      <td>0.715472</td>
      <td>0.781854</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.607069</td>
      <td>0.440000</td>
      <td>0.637931</td>
      <td>0.548387</td>
      <td>0.611111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.739130</td>
      <td>0.838462</td>
      <td>0.710112</td>
      <td>0.772727</td>
      <td>0.737794</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.660633</td>
      <td>0.611111</td>
      <td>0.667526</td>
      <td>0.772727</td>
      <td>0.654762</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.392931</td>
      <td>0.560000</td>
      <td>0.362069</td>
      <td>0.451613</td>
      <td>0.388889</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.260870</td>
      <td>0.161538</td>
      <td>0.289888</td>
      <td>0.227273</td>
      <td>0.262206</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.678977</td>
      <td>0.692683</td>
      <td>0.675676</td>
      <td>0.641509</td>
      <td>0.680957</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.632719</td>
      <td>0.511628</td>
      <td>0.652393</td>
      <td>0.641509</td>
      <td>0.632184</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.418561</td>
      <td>0.263415</td>
      <td>0.455934</td>
      <td>0.415094</td>
      <td>0.418744</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>0.918919</td>
      <td>0.720000</td>
      <td>0.955665</td>
      <td>0.709677</td>
      <td>0.933333</td>
    </tr>
  </tbody>
</table>
</div>



## Group Metrics Composition

**Metrics Composer** is responsible for this second stage of the model audit. Currently, it computes our custom group statistical bias and variance metrics, but extending it for new group metrics is very simple. We noticed that more and more group metrics have appeared during the last decade, but most of them are based on the same subgroup metrics. Hence, such a separation of subgroup and group metrics computation allows one to experiment with different combinations of subgroup metrics and avoid subgroup metrics recomputation for a new set of grouped metrics.


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
      <td>0.197931</td>
      <td>0.062724</td>
      <td>0.186038</td>
      <td>0.349172</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Equalized_Odds_FPR</td>
      <td>0.128349</td>
      <td>0.034933</td>
      <td>0.120236</td>
      <td>0.230252</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Disparate_Impact</td>
      <td>1.327313</td>
      <td>1.315152</td>
      <td>1.187412</td>
      <td>1.730978</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Statistical_Parity_Difference</td>
      <td>0.235665</td>
      <td>0.223656</td>
      <td>0.153337</td>
      <td>0.417702</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Accuracy_Parity</td>
      <td>-0.017007</td>
      <td>0.039448</td>
      <td>-0.007362</td>
      <td>-0.029665</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Label_Stability_Ratio</td>
      <td>1.063910</td>
      <td>1.092782</td>
      <td>1.015365</td>
      <td>1.083336</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>6</th>
      <td>IQR_Parity</td>
      <td>-0.030087</td>
      <td>0.016152</td>
      <td>-0.005684</td>
      <td>-0.033944</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Std_Parity</td>
      <td>-0.019436</td>
      <td>0.014826</td>
      <td>-0.005017</td>
      <td>-0.023656</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Std_Ratio</td>
      <td>0.822726</td>
      <td>1.185571</td>
      <td>0.948220</td>
      <td>0.790095</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jitter_Parity</td>
      <td>-0.029909</td>
      <td>-0.014130</td>
      <td>-0.006402</td>
      <td>-0.035366</td>
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
    metrics_title="Bias Metrics"
)
```





<div id="altair-viz-700c5ea7038849069a023253a2ef717c"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-700c5ea7038849069a023253a2ef717c") {
      outputDiv = document.getElementById("altair-viz-700c5ea7038849069a023253a2ef717c");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-16219b4152ce0a8377e649d7a3ed1f91"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Bias Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-16219b4152ce0a8377e649d7a3ed1f91": [{"overall": 0.6789772727272727, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6327193932827736, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6606334841628959, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 0.918918918918919, "metric": "Positive-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.4185606060606061, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6070686070686071, "metric": "TPR", "model_name": "DecisionTreeClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metrics_names=['Label_Stability'],
    reversed_metrics_names=['Std', 'IQR', 'Jitter'],
    metrics_title="Variance Metrics"
)
```





<div id="altair-viz-b977dd0dc13d463990210dacff0fce6c"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-b977dd0dc13d463990210dacff0fce6c") {
      outputDiv = document.getElementById("altair-viz-b977dd0dc13d463990210dacff0fce6c");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 14, "titleFontSize": 18}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 14, "labelPadding": 10, "titleFontSize": 18}}, "data": {"name": "data-2bdca5438465147db1f00c21f79ff68c"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 13, "title": "Model Name", "titleFontSize": 13}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "title": "Variance Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-2bdca5438465147db1f00c21f79ff68c": [{"overall": 0.8825543491634812, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8405827211509027, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.7785227272727273, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9060266424497246, "metric": "Std", "model_name": "DecisionTreeClassifier"}]}}, {"mode": "vega-lite"});
</script>

