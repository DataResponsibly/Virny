```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
```


```python
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
```


```python
cur_folder_name = os.getcwd().split('/')[-1]
if cur_folder_name != "Virny":
    os.chdir("../..")

print('Current location: ', os.getcwd())
```

    Current location:  /Users/denys_herasymuk/UCU/4course_2term/Bachelor_Thesis/Code/Virny


# Multiple Models Interface With Error Analysis

In this example, we are going to conduct a deep performance profiling for 4 models. The only difference with the multiple models interface tutorial is the use of an `error_analysis` computation mode. This mode measures subgroup metrics also for correct and incorrect predictions. For example, when a default computation mode measures metrics for sex_priv and sex_dis, an `error_analysis` mode measures metrics for (sex_priv, sex_priv_correct, sex_priv_incorrect) and (sex_dis, sex_dis_correct, sex_dis_incorrect). Therefore, a user can analyze how a model is certain about its incorrect predictions.

For that, we will use `compute_metrics_with_config` interface that can compute metrics for multiple models. Thus, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Perform disparity metrics composition using the Metric Composer

* Create static visualizations using the Metric Visualizer

## Import dependencies


```python
import os
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs, create_models_config_from_tuned_params_df
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config
from virny.datasets import CompasWithoutSensitiveAttrsDataset
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
```

    WARNING:root:No module named 'tempeh': LawSchoolGPADataset will be unavailable. To install, run:
    pip install 'aif360[LawSchoolGPA]'


## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **config yaml** that is a file with configuration parameters for different user interfaces for metric computation.

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

* **computation_mode**: str, 'default' or 'error_analysis'. Name of the computation mode. When a default computation mode measures metrics for sex_priv and sex_dis, an `error_analysis` mode measures metrics for (sex_priv, sex_priv_correct, sex_priv_incorrect) and (sex_dis, sex_dis_correct, sex_dis_incorrect). Therefore, a user can analyze how a model is certain about its incorrect predictions.

* **sensitive_attributes_dct**: dict, a dictionary where keys are sensitive attribute names (including intersectional attributes), and values are disadvantaged values for these attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify disadvantaged values for intersectional groups since they will be derived from disadvantaged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.

Note that disadvantaged value in a sensitive attribute dictionary must be **the same as in the original dataset**. For example, when distinct values of the _sex_ column in the original dataset are 'F' and 'M', and after pre-processing they became 0 and 1 respectively, you still need to set a disadvantaged value as 'F' or 'M' in the sensitive attribute dictionary.


```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: COMPAS_Without_Sensitive_Attributes
bootstrap_fraction: 0.8
random_state: 42
n_estimators: 50  # Better to input the higher number of estimators than 100; this is only for this use case example
computation_mode: error_analysis
sensitive_attributes_dct: {'sex': 1, 'race': 'African-American', 'sex&race': None}
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
data_loader = CompasWithoutSensitiveAttrsDataset()
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




```python
column_transformer = ColumnTransformer(transformers=[
    ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
    ('numerical_features', StandardScaler(), data_loader.numerical_columns),
])
```


```python
base_flow_dataset = preprocess_dataset(data_loader=data_loader,
                                       column_transformer=column_transformer,
                                       sensitive_attributes_dct=config.sensitive_attributes_dct,
                                       test_set_fraction=TEST_SET_FRACTION,
                                       dataset_split_seed=DATASET_SPLIT_SEED)
```

### Create a models config for metrics computation

**models_config** is a Python dictionary, where keys are model names and values are initialized models for analysis


```python
models_config = {
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                     max_depth=20,
                                                     max_features=0.6,
                                                     min_samples_split=0.1),
    'LogisticRegression': LogisticRegression(penalty='l2',
                                             C=0.1,
                                             max_iter=250),
    'RandomForestClassifier': RandomForestClassifier(max_depth=4,
                                                     max_features=0.6,
                                                     min_samples_leaf=1,
                                                     n_estimators=50),
    'XGBClassifier': XGBClassifier(learning_rate=0.1,
                                   n_estimators=200,
                                   max_depth=5,
                                   verbosity=0)
}
```

## Subgroup Metric Computation

After that we need to input the _BaseFlowDataset_ object, models config, and config yaml to a metric computation interface and execute it. The interface uses subgroup analyzers to compute different sets of metrics for each privileged and disadvantaged group. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metric computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_dct = compute_metrics_with_config(base_flow_dataset, config, models_config, SAVE_RESULTS_DIR_PATH, notebook_logs_stdout=True)
```


    Analyze multiple models:   0%|          | 0/4 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]


Look at several columns in top rows of computed metrics. Note that now we have metrics also for `*_correct` and `*_incorrect` subgroups.


```python
sample_model_metrics_df = metrics_dct[list(models_config.keys())[0]]
sample_model_metrics_df[sample_model_metrics_df.columns[:5]].head(20)
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
      <th>sex_priv_correct</th>
      <th>sex_priv_incorrect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean_Prediction</td>
      <td>0.521995</td>
      <td>0.577845</td>
      <td>0.601574</td>
      <td>0.525711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.072253</td>
      <td>0.075557</td>
      <td>0.068902</td>
      <td>0.090176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overall_Uncertainty</td>
      <td>0.887763</td>
      <td>0.896446</td>
      <td>0.875769</td>
      <td>0.941874</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aleatoric_Uncertainty</td>
      <td>0.860720</td>
      <td>0.866208</td>
      <td>0.850550</td>
      <td>0.900606</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IQR</td>
      <td>0.082323</td>
      <td>0.084506</td>
      <td>0.074594</td>
      <td>0.106282</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Statistical_Bias</td>
      <td>0.416585</td>
      <td>0.411295</td>
      <td>0.320042</td>
      <td>0.611776</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Epistemic_Uncertainty</td>
      <td>0.027043</td>
      <td>0.030239</td>
      <td>0.025219</td>
      <td>0.041268</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jitter</td>
      <td>0.110380</td>
      <td>0.130717</td>
      <td>0.099423</td>
      <td>0.199468</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Label_Stability</td>
      <td>0.859508</td>
      <td>0.825403</td>
      <td>0.871448</td>
      <td>0.724242</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TPR</td>
      <td>0.651805</td>
      <td>0.466667</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TNR</td>
      <td>0.733333</td>
      <td>0.808824</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PPV</td>
      <td>0.663067</td>
      <td>0.573770</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FNR</td>
      <td>0.348195</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FPR</td>
      <td>0.266667</td>
      <td>0.191176</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Accuracy</td>
      <td>0.696970</td>
      <td>0.687204</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>F1</td>
      <td>0.657388</td>
      <td>0.514706</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Selection-Rate</td>
      <td>0.438447</td>
      <td>0.289100</td>
      <td>0.241379</td>
      <td>0.393939</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Positive-Rate</td>
      <td>0.983015</td>
      <td>0.813333</td>
      <td>1.000000</td>
      <td>0.650000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sample_Size</td>
      <td>1056.000000</td>
      <td>211.000000</td>
      <td>145.000000</td>
      <td>66.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Disparity Metric Composition

To compose disparity metrics, the Metric Composer should be applied. **Metric Composer** is responsible for the second stage of the model audit. Currently, it computes our custom error disparity, stability disparity, and uncertainty disparity metrics, but extending it for new disparity metrics is very simple. We noticed that more and more disparity metrics have appeared during the last decade, but most of them are based on the same group specific metrics. Hence, such a separation of group specific and disparity metrics computation allows us to experiment with different combinations of group specific metrics and avoid group metrics recomputation for a new set of disparity metrics.


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

## Metric Visualization

**Metric Visualizer** allows us to build static visualizations for the computed metrics. It unifies different preprocessing methods for the computed metrics and creates various data formats required for visualizations. Hence, users can simply call methods of the MetricsVisualizer class and get custom plots for diverse metric analysis.


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





<div id="altair-viz-3416c4f16d5a4cb1a078edf5cf83739a"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-3416c4f16d5a4cb1a078edf5cf83739a") {
      outputDiv = document.getElementById("altair-viz-3416c4f16d5a4cb1a078edf5cf83739a");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-4377aaea72661c7a7f79f075a0cf02b3"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Accuracy", "F1", "TPR", "TNR", "PPV", "Selection-Rate"], "title": "Accuracy Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-4377aaea72661c7a7f79f075a0cf02b3": [{"overall": 0.696969696969697, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6573875802997858, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6630669546436285, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 0.4384469696969697, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.7333333333333333, "metric": "TNR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6518046709129511, "metric": "TPR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6837121212121212, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.6369565217391304, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.6525612472160356, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 0.4251893939393939, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.7333333333333333, "metric": "TNR", "model_name": "LogisticRegression"}, {"overall": 0.6220806794055201, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.7007575757575758, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.6594827586206896, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.6695842450765864, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 0.4327651515151515, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.7418803418803419, "metric": "TNR", "model_name": "RandomForestClassifier"}, {"overall": 0.6496815286624203, "metric": "TPR", "model_name": "RandomForestClassifier"}, {"overall": 0.7073863636363636, "metric": "Accuracy", "model_name": "XGBClassifier"}, {"overall": 0.6680988184747583, "metric": "F1", "model_name": "XGBClassifier"}, {"overall": 0.6760869565217391, "metric": "PPV", "model_name": "XGBClassifier"}, {"overall": 0.4356060606060606, "metric": "Selection-Rate", "model_name": "XGBClassifier"}, {"overall": 0.7452991452991453, "metric": "TNR", "model_name": "XGBClassifier"}, {"overall": 0.6602972399150743, "metric": "TPR", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Aleatoric_Uncertainty', 'Overall_Uncertainty', 'Label_Stability', 'Std', 'IQR', 'Jitter'],
    plot_title="Stability and Uncertainty Metrics"
)
```





<div id="altair-viz-4eb6bce4207946df99654e2b95bae3ce"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-4eb6bce4207946df99654e2b95bae3ce") {
      outputDiv = document.getElementById("altair-viz-4eb6bce4207946df99654e2b95bae3ce");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-3614bc049170c97b2d93016ad918ed87"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Aleatoric_Uncertainty", "Overall_Uncertainty", "Label_Stability", "Std", "IQR", "Jitter"], "title": "Stability and Uncertainty Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-3614bc049170c97b2d93016ad918ed87": [{"overall": 0.8607199637290075, "metric": "Aleatoric_Uncertainty", "model_name": "DecisionTreeClassifier"}, {"overall": 0.0823234508592164, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.1103795609152753, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8595075757575757, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8877630638614109, "metric": "Overall_Uncertainty", "model_name": "DecisionTreeClassifier"}, {"overall": 0.0722530200070569, "metric": "Std", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9098846506898132, "metric": "Aleatoric_Uncertainty", "model_name": "LogisticRegression"}, {"overall": 0.0305957160788724, "metric": "IQR", "model_name": "LogisticRegression"}, {"overall": 0.0497835497835497, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.9365151515151516, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.9127038364381584, "metric": "Overall_Uncertainty", "model_name": "LogisticRegression"}, {"overall": 0.0225548750141018, "metric": "Std", "model_name": "LogisticRegression"}, {"overall": 0.90655470177347, "metric": "Aleatoric_Uncertainty", "model_name": "RandomForestClassifier"}, {"overall": 0.0526181463140505, "metric": "IQR", "model_name": "RandomForestClassifier"}, {"overall": 0.0670949288806432, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.9121969696969696, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}, {"overall": 0.9122376701069465, "metric": "Overall_Uncertainty", "model_name": "RandomForestClassifier"}, {"overall": 0.0393746870091122, "metric": "Std", "model_name": "RandomForestClassifier"}, {"overall": 0.8269357681274414, "metric": "Aleatoric_Uncertainty", "model_name": "XGBClassifier"}, {"overall": 0.0908706365153193, "metric": "IQR", "model_name": "XGBClassifier"}, {"overall": 0.1086966604823747, "metric": "Jitter", "model_name": "XGBClassifier"}, {"overall": 0.8509848484848486, "metric": "Label_Stability", "model_name": "XGBClassifier"}, {"overall": 0.855431345573202, "metric": "Overall_Uncertainty", "model_name": "XGBClassifier"}, {"overall": 0.0696154683828353, "metric": "Std", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metric_heatmap(
    model_names=list(models_metrics_dct.keys()),
    metrics_lst=visualizer.all_accuracy_metrics + visualizer.all_stability_metrics,
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_With_Error_Analysis_files/Multiple_Models_Interface_With_Error_Analysis_38_0.png)
    



```python
visualizer.create_disparity_metric_heatmap(
    model_names=list(models_metrics_dct.keys()),
    metrics_lst=[
        # Error disparity metrics
        'Equalized_Odds_TPR',
        'Equalized_Odds_FPR',
        'Disparate_Impact',
        # Stability disparity metrics
        'Label_Stability_Difference',
        'Aleatoric_Uncertainty_Difference',
        'Std_Ratio',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_With_Error_Analysis_files/Multiple_Models_Interface_With_Error_Analysis_39_0.png)
    



```python

```
