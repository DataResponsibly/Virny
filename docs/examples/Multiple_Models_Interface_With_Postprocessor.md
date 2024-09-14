# Multiple Models Interface With Postprocessor

In this example, we are going to audit 2 models together with a postprocessor from AIF360, visualize metrics, and create an analysis report. For that, we will use `compute_metrics_with_config` interface that can compute metrics for multiple models. Thus, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Perform disparity metrics composition using the Metric Composer

* Create static visualizations using the Metric Visualizer

## Import dependencies


```python
import os
from pprint import pprint
from datetime import datetime, timezone

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

* **computation_mode**: str, 'default' or 'error_analysis'. Name of the computation mode. When a default computation mode measures metrics for sex_priv and sex_dis, an `error_analysis` mode measures metrics for (sex_priv, sex_priv_correct, sex_priv_incorrect) and (sex_dis, sex_dis_correct, sex_dis_incorrect). Therefore, a user can analyze how a model is certain about its incorrect predictions.

* **random_state**: int, a seed to control the randomness of the whole model evaluation pipeline.

* **n_estimators**: int, the number of estimators for bootstrap to compute subgroup stability metrics.

* **sensitive_attributes_dct**: dict, a dictionary where keys are sensitive attribute names (including intersectional attributes), and values are disadvantaged values for these attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify disadvantaged values for intersectional groups since they will be derived from disadvantaged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.

* **postprocessing_sensitive_attribute**: str, a name of a sensitive attribute to use for postprocessing.

Note that disadvantaged value in a sensitive attribute dictionary must be **the same as in the original dataset**. For example, when distinct values of the _sex_ column in the original dataset are 'F' and 'M', and after pre-processing they became 0 and 1 respectively, you still need to set a disadvantaged value as 'F' or 'M' in the sensitive attribute dictionary.


```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: Law_School
bootstrap_fraction: 0.8
computation_mode: error_analysis
random_state: 42
n_estimators: 50  # Better to input the higher number of estimators than 100; this is only for this use case example
sensitive_attributes_dct: {'male': '0', 'race': 'Non-White', 'male&race': None}
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
from virny.datasets import LawSchoolDataset

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
    ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse_output=False), data_loader.categorical_columns),
    ('numerical_features', StandardScaler(), data_loader.numerical_columns),
])
```


```python
# Create a binary race column for postprocessing since aif360 postprocessors can postprocess a dataset only based on binary columns.
data_loader.X_data['race_binary'] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'White' else 0)

base_flow_dataset = preprocess_dataset(data_loader=data_loader,
                                       column_transformer=column_transformer,
                                       sensitive_attributes_dct=config.sensitive_attributes_dct,
                                       test_set_fraction=TEST_SET_FRACTION,
                                       dataset_split_seed=DATASET_SPLIT_SEED)
base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']
```


```python
# Define a postprocessor
privileged_groups = [{'race_binary': 1}]
unprivileged_groups = [{'race_binary': 0}]
postprocessor = EqOddsPostprocessing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    seed=None  # Set postprocessor's seed to None to avoid similar predictions during the bootstrap
)
```

### Tune models and create a models config for metrics computation


```python
tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset, config.dataset_name, n_folds=3)
tuned_params_df
```

    2024/06/02, 00:35:52: Tuning LogisticRegression...
    2024/06/02, 00:35:54: Tuning for LogisticRegression is finished [F1 score = 0.6563618630035558, Accuracy = 0.8987258083904316]
    
    2024/06/02, 00:35:54: Tuning RandomForestClassifier...
    2024/06/02, 00:35:56: Tuning for RandomForestClassifier is finished [F1 score = 0.6538551003755212, Accuracy = 0.8980646712345234]





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


    Enabled a postprocessing mode



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]


    Enabled a postprocessing mode



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
      <th>male_priv_correct</th>
      <th>male_priv_incorrect</th>
      <th>male_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jitter</td>
      <td>0.044141</td>
      <td>0.040939</td>
      <td>0.035644</td>
      <td>0.094502</td>
      <td>0.048374</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Label_Stability</td>
      <td>0.949913</td>
      <td>0.953970</td>
      <td>0.961893</td>
      <td>0.873803</td>
      <td>0.944554</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPR</td>
      <td>0.994903</td>
      <td>0.994884</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.994930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TNR</td>
      <td>0.078704</td>
      <td>0.073394</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.084112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PPV</td>
      <td>0.903092</td>
      <td>0.913712</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.889015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FNR</td>
      <td>0.005097</td>
      <td>0.005116</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.005070</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FPR</td>
      <td>0.921296</td>
      <td>0.926606</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.915888</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Accuracy</td>
      <td>0.899760</td>
      <td>0.910051</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.886161</td>
    </tr>
    <tr>
      <th>8</th>
      <td>F1</td>
      <td>0.946777</td>
      <td>0.952572</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.938995</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Selection-Rate</td>
      <td>0.987260</td>
      <td>0.988598</td>
      <td>0.992575</td>
      <td>0.948357</td>
      <td>0.985491</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sample_Size</td>
      <td>4160.000000</td>
      <td>2368.000000</td>
      <td>2155.000000</td>
      <td>213.000000</td>
      <td>1792.000000</td>
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
      <td>Accuracy_Difference</td>
      <td>-0.023890</td>
      <td>-0.196227</td>
      <td>-0.174183</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Equalized_Odds_FNR</td>
      <td>-0.000047</td>
      <td>-0.005823</td>
      <td>-0.005454</td>
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
      <td>Jitter_Difference</td>
      <td>0.007435</td>
      <td>0.034351</td>
      <td>0.049795</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Label_Stability_Ratio</td>
      <td>0.990130</td>
      <td>0.943974</td>
      <td>0.924259</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Label_Stability_Difference</td>
      <td>-0.009416</td>
      <td>-0.053678</td>
      <td>-0.072383</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Statistical_Parity_Difference</td>
      <td>-0.003107</td>
      <td>0.015031</td>
      <td>0.013838</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Disparate_Impact</td>
      <td>0.996857</td>
      <td>1.015261</td>
      <td>1.014032</td>
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
      <td>0.000047</td>
      <td>0.005823</td>
      <td>0.005454</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Accuracy_Difference</td>
      <td>-0.020693</td>
      <td>-0.158407</td>
      <td>-0.134267</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Equalized_Odds_FNR</td>
      <td>0.004134</td>
      <td>0.020908</td>
      <td>0.029136</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Equalized_Odds_FPR</td>
      <td>-0.058218</td>
      <td>-0.104439</td>
      <td>-0.140207</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Jitter_Difference</td>
      <td>0.009800</td>
      <td>0.093877</td>
      <td>0.101423</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Label_Stability_Ratio</td>
      <td>0.981678</td>
      <td>0.844858</td>
      <td>0.825698</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Label_Stability_Difference</td>
      <td>-0.017405</td>
      <td>-0.149755</td>
      <td>-0.166575</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Statistical_Parity_Difference</td>
      <td>-0.013514</td>
      <td>-0.061529</td>
      <td>-0.076446</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Disparate_Impact</td>
      <td>0.986242</td>
      <td>0.937586</td>
      <td>0.922193</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Equalized_Odds_TNR</td>
      <td>0.058218</td>
      <td>0.104439</td>
      <td>0.140207</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Equalized_Odds_TPR</td>
      <td>-0.004134</td>
      <td>-0.020908</td>
      <td>-0.029136</td>
      <td>RandomForestClassifier</td>
    </tr>
  </tbody>
</table>
</div>



## Metric Visualization

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





<div id="altair-viz-2b16e75ab1794ba3934c83da650c9449"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-2b16e75ab1794ba3934c83da650c9449") {
      outputDiv = document.getElementById("altair-viz-2b16e75ab1794ba3934c83da650c9449");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-e375c8a67e434d2b83591104119a1e44"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Accuracy", "F1", "TPR", "TNR", "PPV", "Selection-Rate"], "title": "Accuracy Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-e375c8a67e434d2b83591104119a1e44": [{"overall": 0.8997596153846154, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.9467772814294833, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.90309228147066, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 0.9872596153846154, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.0787037037037037, "metric": "TNR", "model_name": "LogisticRegression"}, {"overall": 0.9949034334763948, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.9024038461538462, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.9478818998716304, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.9089118660758247, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 0.9764423076923076, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.1435185185185185, "metric": "TNR", "model_name": "RandomForestClassifier"}, {"overall": 0.990343347639485, "metric": "TPR", "model_name": "RandomForestClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Label_Stability', 'Jitter'],
    plot_title="Stability Metrics"
)
```





<div id="altair-viz-868f4cc36a9749e6afd91ebd75521c0f"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-868f4cc36a9749e6afd91ebd75521c0f") {
      outputDiv = document.getElementById("altair-viz-868f4cc36a9749e6afd91ebd75521c0f");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-aef3c30529bc9bb1916dccf6536b6639"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Label_Stability", "Jitter"], "title": "Stability Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-aef3c30529bc9bb1916dccf6536b6639": [{"overall": 0.0441414835164834, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.9499134615384616, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.0416481554160125, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.9424519230769232, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}]}}, {"mode": "vega-lite"});
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
        'Jitter_Difference',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_With_Postprocessor_files/Multiple_Models_Interface_With_Postprocessor_43_0.png)
