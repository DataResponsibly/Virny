# Multiple Models Interface With Postprocessor

In this example, we are going to conduct a deep performance profiling for 2 models. The only difference with a tutorial about the multiple model interface is that we input an additional argument _postprocessor_ to the `compute_metrics_with_config` interface and set _postprocessing_sensitive_attribute_ in the config yaml. Thus, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Perform disparity metrics composition using the Metric Composer

* Create static visualizations using the Metric Visualizer

## Import dependencies


```python
import os
from pprint import pprint
from datetime import datetime, timezone

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.postprocessing import EqOddsPostprocessing

from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs, create_models_config_from_tuned_params_df
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.utils.model_tuning_utils import tune_ML_models
```

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


```python
models_params_for_tuning = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=MODELS_TUNING_SEED),
        'params': {
            'penalty': ['l2'],
            'C' : [0.0001, 0.1, 1, 100],
            'solver': ['newton-cg', 'lbfgs'],
            'max_iter': [250],
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=MODELS_TUNING_SEED),
        'params': {
            "max_depth": [6, 10],
            "min_samples_leaf": [1],
            "n_estimators": [50, 100],
            "max_features": [0.6]
        }
    },
}
```

### Create a config object

`compute_metrics_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: str, a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: float, the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: int, the number of estimators for bootstrap to compute subgroup stability metrics.

* **computation_mode**: must be set as 'postprocessing_intervention' to enable postprocessing.

* **sensitive_attributes_dct**: dict, a dictionary where keys are sensitive attribute names (including intersectional attributes), and values are disadvantaged values for these attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify disadvantaged values for intersectional groups since they will be derived from disadvantaged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.

* **postprocessing_sensitive_attribute**: str, a name of a sensitive attribute to use for postprocessing.

Note that disadvantaged value in a sensitive attribute dictionary must be **the same as in the original dataset**. For example, when distinct values of the _sex_ column in the original dataset are 'F' and 'M', and after pre-processing they became 0 and 1 respectively, you still need to set a disadvantaged value as 'F' or 'M' in the sensitive attribute dictionary.


```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: Law_School
bootstrap_fraction: 0.8
computation_mode: postprocessing_intervention
n_estimators: 50  # Better to input the higher number of estimators than 100; this is only for this use case example
sensitive_attributes_dct: {'male': '0.0', 'race': 'Non-White', 'male&race': None}
postprocessing_sensitive_attribute: 'race_binary'
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)
```


```python
config = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', f'{config.dataset_name}_Metrics_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}')
```

### Preprocess the dataset, create a BaseFlowDataset class, and define a postprocessor


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
# Create a binary race column for postprocessing since aif360 postprocessors can postprocess a dataset only based on binary columns.
data_loader.X_data['race_binary'] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'White' else 0)

base_flow_dataset = preprocess_dataset(data_loader, column_transformer, TEST_SET_FRACTION, DATASET_SPLIT_SEED)
base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']
```


```python
# Define a postprocessor
privileged_groups = [{'race_binary': 1}]
unprivileged_groups = [{'race_binary': 0}]
postprocessor = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                     unprivileged_groups=unprivileged_groups,
                                     seed=None)
```

### Tune models and create a models config for metrics computation


```python
tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset, config.dataset_name, n_folds=3)
tuned_params_df
```

    2023/12/26, 17:00:54: Tuning LogisticRegression...
    2023/12/26, 17:00:55: Tuning for LogisticRegression is finished [F1 score = 0.6563618630035558, Accuracy = 0.8987258083904316]
    
    2023/12/26, 17:00:55: Tuning RandomForestClassifier...
    2023/12/26, 17:00:58: Tuning for RandomForestClassifier is finished [F1 score = 0.6538551003755212, Accuracy = 0.8980646712345234]





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
      <th>Dataset_Name</th>
      <th>Model_Name</th>
      <th>F1_Score</th>
      <th>Accuracy_Score</th>
      <th>Model_Best_Params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Law_School</td>
      <td>LogisticRegression</td>
      <td>0.656362</td>
      <td>0.898726</td>
      <td>{'C': 100, 'max_iter': 250, 'penalty': 'l2', '...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Law_School</td>
      <td>RandomForestClassifier</td>
      <td>0.653855</td>
      <td>0.898065</td>
      <td>{'max_depth': 10, 'max_features': 0.6, 'min_sa...</td>
    </tr>
  </tbody>
</table>
</div>




```python
now = datetime.now(timezone.utc)
date_time_str = now.strftime("%Y%m%d__%H%M%S")
tuned_df_path = os.path.join(ROOT_DIR, 'results', 'models_tuning', f'tuning_results_{config.dataset_name}_{date_time_str}.csv')
tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
```

Create models_config from the saved tuned_params_df for higher reliability


```python
models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_df_path)
pprint(models_config)
```

    {'LogisticRegression': LogisticRegression(C=100, max_iter=250, random_state=42, solver='newton-cg'),
     'RandomForestClassifier': RandomForestClassifier(max_depth=10, max_features=0.6, n_estimators=50,
                           random_state=42)}


## Subgroup Metric Computation

After the variables are input to a user interface, the interface uses subgroup analyzers to compute different sets of metrics for each privileged and disadvantaged subgroup. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metrics computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_dct = compute_metrics_with_config(dataset=base_flow_dataset,
                                          config=config,
                                          models_config=models_config,
                                          save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                          postprocessor=postprocessor,
                                          notebook_logs_stdout=True)
```


    Analyze multiple models:   0%|          | 0/2 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]


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
      <th>male_dis</th>
      <th>race_priv</th>
      <th>race_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Label_Stability</td>
      <td>0.952250</td>
      <td>0.955963</td>
      <td>0.947344</td>
      <td>0.960170</td>
      <td>0.908202</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jitter</td>
      <td>0.041991</td>
      <td>0.039177</td>
      <td>0.045709</td>
      <td>0.036896</td>
      <td>0.070329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPR</td>
      <td>0.995172</td>
      <td>0.995349</td>
      <td>0.994930</td>
      <td>0.994484</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TNR</td>
      <td>0.078704</td>
      <td>0.073394</td>
      <td>0.084112</td>
      <td>0.129278</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PPV</td>
      <td>0.903116</td>
      <td>0.913749</td>
      <td>0.889015</td>
      <td>0.934082</td>
      <td>0.733438</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FNR</td>
      <td>0.004828</td>
      <td>0.004651</td>
      <td>0.005070</td>
      <td>0.005516</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FPR</td>
      <td>0.921296</td>
      <td>0.926606</td>
      <td>0.915888</td>
      <td>0.870722</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Accuracy</td>
      <td>0.900000</td>
      <td>0.910473</td>
      <td>0.886161</td>
      <td>0.929949</td>
      <td>0.733438</td>
    </tr>
    <tr>
      <th>8</th>
      <td>F1</td>
      <td>0.946912</td>
      <td>0.952805</td>
      <td>0.938995</td>
      <td>0.963337</td>
      <td>0.846224</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Selection-Rate</td>
      <td>0.987500</td>
      <td>0.989020</td>
      <td>0.985491</td>
      <td>0.985252</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Positive-Rate</td>
      <td>1.101931</td>
      <td>1.089302</td>
      <td>1.119138</td>
      <td>1.064664</td>
      <td>1.363441</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sample_Size</td>
      <td>4160.000000</td>
      <td>2368.000000</td>
      <td>1792.000000</td>
      <td>3526.000000</td>
      <td>634.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Disparity Metric Composition

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
      <td>Accuracy_Parity</td>
      <td>-0.024312</td>
      <td>-0.196510</td>
      <td>-0.174444</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Equalized_Odds_FNR</td>
      <td>0.000419</td>
      <td>-0.005516</td>
      <td>-0.005166</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Equalized_Odds_FPR</td>
      <td>-0.010718</td>
      <td>0.129278</td>
      <td>0.098266</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jitter_Parity</td>
      <td>0.006532</td>
      <td>0.033433</td>
      <td>0.048724</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Label_Stability_Ratio</td>
      <td>0.990984</td>
      <td>0.945876</td>
      <td>0.925362</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Label_Stability_Difference</td>
      <td>-0.008619</td>
      <td>-0.051968</td>
      <td>-0.071497</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Disparate_Impact</td>
      <td>1.027390</td>
      <td>1.280630</td>
      <td>1.247212</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Statistical_Parity_Difference</td>
      <td>-0.003529</td>
      <td>0.014748</td>
      <td>0.013577</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Equalized_Odds_TNR</td>
      <td>0.010718</td>
      <td>-0.129278</td>
      <td>-0.098266</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Equalized_Odds_TPR</td>
      <td>-0.000419</td>
      <td>0.005516</td>
      <td>0.005166</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Accuracy_Parity</td>
      <td>-0.013167</td>
      <td>-0.150202</td>
      <td>-0.112374</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Equalized_Odds_FNR</td>
      <td>-0.000343</td>
      <td>0.012914</td>
      <td>0.011227</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Equalized_Odds_FPR</td>
      <td>-0.090757</td>
      <td>-0.112471</td>
      <td>-0.175091</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Jitter_Parity</td>
      <td>0.010870</td>
      <td>0.093276</td>
      <td>0.102008</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Label_Stability_Ratio</td>
      <td>0.981777</td>
      <td>0.848091</td>
      <td>0.829891</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Label_Stability_Difference</td>
      <td>-0.017342</td>
      <td>-0.146832</td>
      <td>-0.162814</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Disparate_Impact</td>
      <td>1.017189</td>
      <td>1.186318</td>
      <td>1.139418</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Statistical_Parity_Difference</td>
      <td>-0.013257</td>
      <td>-0.059066</td>
      <td>-0.072735</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Equalized_Odds_TNR</td>
      <td>0.090757</td>
      <td>0.112471</td>
      <td>0.175091</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Equalized_Odds_TPR</td>
      <td>0.000343</td>
      <td>-0.012914</td>
      <td>-0.011227</td>
      <td>RandomForestClassifier</td>
    </tr>
  </tbody>
</table>
</div>



## Metric Visualization

**Metrics Visualizer** provides metrics visualization and reporting functionality. It unifies different preprocessing methods for result metrics and creates various data formats required for visualizations. Hence, users can simply call methods of the Metrics Visualizer class and get custom plots for diverse metrics analysis. Additionally, these plots could be collected in an HTML report with comments for user convenience and future reference.


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





<div id="altair-viz-206fcd1109334a94abe1ad0b228714dc"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-206fcd1109334a94abe1ad0b228714dc") {
      outputDiv = document.getElementById("altair-viz-206fcd1109334a94abe1ad0b228714dc");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-596e8936162d820be1993bc97bef62a9"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Accuracy", "F1", "TPR", "TNR", "PPV", "Selection-Rate"], "title": "Accuracy Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-596e8936162d820be1993bc97bef62a9": [{"overall": 0.9, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.9469116896375702, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.9031158714703018, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 0.9875, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.0787037037037037, "metric": "TNR", "model_name": "LogisticRegression"}, {"overall": 0.9951716738197424, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.9064903846153848, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.950134598128445, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.909894426712497, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 0.9790865384615384, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.1504629629629629, "metric": "TNR", "model_name": "RandomForestClassifier"}, {"overall": 0.994098712446352, "metric": "TPR", "model_name": "RandomForestClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Label_Stability', 'Jitter'],
    plot_title="Stability Metrics"
)
```





<div id="altair-viz-e30fad52c50d4bcea903c428de2fffa2"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-e30fad52c50d4bcea903c428de2fffa2") {
      outputDiv = document.getElementById("altair-viz-e30fad52c50d4bcea903c428de2fffa2");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-25573cb0410b4f32ff4ee7854694185a"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Label_Stability", "Jitter"], "title": "Stability Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-25573cb0410b4f32ff4ee7854694185a": [{"overall": 0.041990973312402, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.95225, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.0411363814756671, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.9442019230769232, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metric_heatmap(
    model_names=list(models_params_for_tuning.keys()),
    metrics_lst=['Accuracy', 'F1', 'TNR', 'TPR', 'FNR', 'FPR', 'Label_Stability', 'Jitter'],
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_With_Postprocessor_files/Multiple_Models_Interface_With_Postprocessor_42_0.png)
    



```python
visualizer.create_disparity_metric_heatmap(
    model_names=list(models_params_for_tuning.keys()),
    metrics_lst=[
        # Error disparity metrics
        'Equalized_Odds_TPR',
        'Equalized_Odds_FPR',
        'Disparate_Impact',
        # Stability disparity metrics
        'Label_Stability_Difference',
        'Jitter_Parity',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_With_Postprocessor_files/Multiple_Models_Interface_With_Postprocessor_43_0.png)
