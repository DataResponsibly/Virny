```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



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

    Current location:  /home/denys_herasymuk/UCU/4course_2term/Bachelor_Thesis/Code/Virny


# Benchmark

## Import dependencies


```python
import os
import pandas as pd

from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from docs.examples.benchmark_utils import clear_directory, create_averaged_dfs_dict, populate_benchmark_report
from virny.datasets.data_loaders import CompasWithoutSensitiveAttrsDataset, ACSEmploymentDataset
from virny.user_interfaces.metrics_computation_interfaces import compute_metrics_multiple_runs
from virny.utils.custom_initializers import create_config_obj, read_model_metric_dfs
```

## COMPAS_Without_Sensitive_Attributes Dataset

## Initialize Input Variables

### Create a models config


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

### Create a Dataset class


```python
dataset = CompasWithoutSensitiveAttrsDataset(dataset_path=os.path.join('virny', 'datasets', 'COMPAS.csv'))
dataset.X_data.head()
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
      <th>age_cat_Greater than 45</th>
      <th>age_cat_Less than 25</th>
      <th>c_charge_degree_F</th>
      <th>c_charge_degree_M</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.513697</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Create a config object


```python
ROOT_DIR = os.path.join('docs', 'examples')
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: COMPAS_Without_Sensitive_Attributes
test_set_fraction: 0.2
bootstrap_fraction: 0.8
n_estimators: 100
runs_seed_lst: [100, 200, 300]
#runs_seed_lst: [100, 200]
sensitive_attributes_dct: {'sex': 0, 'race': 'Caucasian', 'sex&race': None}
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)
```


```python
config = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH1 = os.path.join(ROOT_DIR, 'results', 'benchmark', 'benchmark_results', config.dataset_name)
STANDARD_RESULTS_DIR_PATH1 = os.path.join(ROOT_DIR, 'results', 'benchmark', 'standard_results', config.dataset_name)
```


```python
clear_directory(SAVE_RESULTS_DIR_PATH1)
```

    Directory is cleared


## Subgroup Metrics Computation


```python
multiple_run_metrics_dct = compute_metrics_multiple_runs(dataset, config, models_config, SAVE_RESULTS_DIR_PATH1, verbose=1)
```


    Multiple runs progress:   0%|          | 0/3 [00:00<?, ?it/s]



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  101
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:21:53 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:21:54 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:21:56 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  102
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:22:03 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:22:07 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:22:09 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  103
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:22:16 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:22:26 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:22:29 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  104
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:22:35 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:22:39 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:22:42 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  201
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:22:48 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:22:48 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:22:50 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  202
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:22:56 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:22:58 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:23:01 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  203
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:23:07 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:23:15 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:23:17 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  204
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:23:23 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:23:26 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:23:28 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  301
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:23:35 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:23:35 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:23:38 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  302
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:23:46 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:23:50 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:23:52 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  303
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:23:59 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:24:08 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:24:10 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  304
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-02-04 15:24:17 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:24:22 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:24:25 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



```python
sample_model_metrics_df = multiple_run_metrics_dct[list(models_config.keys())[0]]
sample_model_metrics_df.head(20)
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
      <th>sex&amp;race_priv</th>
      <th>sex&amp;race_dis</th>
      <th>Model_Seed</th>
      <th>Model_Name</th>
      <th>Run_Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.527155</td>
      <td>0.559299</td>
      <td>0.519691</td>
      <td>0.592431</td>
      <td>0.482842</td>
      <td>0.590330</td>
      <td>0.471730</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.072303</td>
      <td>0.077301</td>
      <td>0.071142</td>
      <td>0.070465</td>
      <td>0.073550</td>
      <td>0.087271</td>
      <td>0.074440</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.091349</td>
      <td>0.097852</td>
      <td>0.089839</td>
      <td>0.092370</td>
      <td>0.090656</td>
      <td>0.112774</td>
      <td>0.091649</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.276445</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.268387</td>
      <td>0.267615</td>
      <td>0.265161</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.162999</td>
      <td>0.183252</td>
      <td>0.158296</td>
      <td>0.143923</td>
      <td>0.175948</td>
      <td>0.177904</td>
      <td>0.173475</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.653400</td>
      <td>0.663618</td>
      <td>0.651027</td>
      <td>0.656370</td>
      <td>0.651383</td>
      <td>0.643750</td>
      <td>0.645386</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.759943</td>
      <td>0.724221</td>
      <td>0.768238</td>
      <td>0.781499</td>
      <td>0.745310</td>
      <td>0.728409</td>
      <td>0.750541</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.593429</td>
      <td>0.542857</td>
      <td>0.601918</td>
      <td>0.427673</td>
      <td>0.673780</td>
      <td>0.440000</td>
      <td>0.685512</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.743409</td>
      <td>0.759690</td>
      <td>0.738636</td>
      <td>0.805970</td>
      <td>0.687708</td>
      <td>0.730159</td>
      <td>0.659574</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.664368</td>
      <td>0.550725</td>
      <td>0.685792</td>
      <td>0.566667</td>
      <td>0.701587</td>
      <td>0.392857</td>
      <td>0.708029</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.406571</td>
      <td>0.457143</td>
      <td>0.398082</td>
      <td>0.572327</td>
      <td>0.326220</td>
      <td>0.560000</td>
      <td>0.314488</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.256591</td>
      <td>0.240310</td>
      <td>0.261364</td>
      <td>0.194030</td>
      <td>0.312292</td>
      <td>0.269841</td>
      <td>0.340426</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.674242</td>
      <td>0.683417</td>
      <td>0.672112</td>
      <td>0.665105</td>
      <td>0.680445</td>
      <td>0.647727</td>
      <td>0.673745</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.626898</td>
      <td>0.546763</td>
      <td>0.641124</td>
      <td>0.487455</td>
      <td>0.687403</td>
      <td>0.415094</td>
      <td>0.696589</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.411932</td>
      <td>0.346734</td>
      <td>0.427071</td>
      <td>0.281030</td>
      <td>0.500795</td>
      <td>0.318182</td>
      <td>0.528958</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>0.893224</td>
      <td>0.985714</td>
      <td>0.877698</td>
      <td>0.754717</td>
      <td>0.960366</td>
      <td>1.120000</td>
      <td>0.968198</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.523108</td>
      <td>0.556546</td>
      <td>0.515295</td>
      <td>0.588047</td>
      <td>0.482056</td>
      <td>0.597748</td>
      <td>0.473006</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.072069</td>
      <td>0.079373</td>
      <td>0.070363</td>
      <td>0.069654</td>
      <td>0.073596</td>
      <td>0.084972</td>
      <td>0.073286</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.089271</td>
      <td>0.100294</td>
      <td>0.086696</td>
      <td>0.089385</td>
      <td>0.089199</td>
      <td>0.108994</td>
      <td>0.088269</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.217344</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
  </tbody>
</table>
</div>



## Create a Benchmark Report


```python
models_metrics_dct = read_model_metric_dfs(SAVE_RESULTS_DIR_PATH1, model_names=models_config.keys())
standard_models_metrics_dct = read_model_metric_dfs(STANDARD_RESULTS_DIR_PATH1, model_names=models_config.keys())

avg_models_metrics_dct = create_averaged_dfs_dict(models_metrics_dct)
avg_standard_models_metrics_dct = create_averaged_dfs_dict(standard_models_metrics_dct)
```


```python
report_df = pd.DataFrame()
report_df = populate_benchmark_report(report_df, avg_models_metrics_dct, avg_standard_models_metrics_dct,
                                      config.dataset_name, config.sensitive_attributes_dct)
```


```python
report_df
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
      <th>Dataset</th>
      <th>Model</th>
      <th>Subgroup</th>
      <th>Delta%_Accuracy</th>
      <th>Delta%_Entropy</th>
      <th>Delta%_F1</th>
      <th>Delta%_FNR</th>
      <th>Delta%_FPR</th>
      <th>Delta%_IQR</th>
      <th>Delta%_Jitter</th>
      <th>Delta%_Label_Stability</th>
      <th>Delta%_Mean</th>
      <th>Delta%_PPV</th>
      <th>Delta%_Per_Sample_Accuracy</th>
      <th>Delta%_Positive-Rate</th>
      <th>Delta%_Selection-Rate</th>
      <th>Delta%_Std</th>
      <th>Delta%_TNR</th>
      <th>Delta%_TPR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>overall</td>
      <td>0.095</td>
      <td>0.775</td>
      <td>0.146</td>
      <td>-0.201</td>
      <td>0.008</td>
      <td>0.016</td>
      <td>0.676</td>
      <td>-1.284</td>
      <td>0.049</td>
      <td>0.076</td>
      <td>-0.145</td>
      <td>0.192</td>
      <td>0.095</td>
      <td>0.124</td>
      <td>-0.008</td>
      <td>0.201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex_priv</td>
      <td>0.006</td>
      <td>0.538</td>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.021</td>
      <td>0.479</td>
      <td>-1.167</td>
      <td>0.108</td>
      <td>0.015</td>
      <td>0.035</td>
      <td>-0.049</td>
      <td>-0.006</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex_dis</td>
      <td>0.117</td>
      <td>0.829</td>
      <td>0.167</td>
      <td>-0.236</td>
      <td>0.009</td>
      <td>0.024</td>
      <td>0.723</td>
      <td>-1.312</td>
      <td>0.035</td>
      <td>0.079</td>
      <td>-0.187</td>
      <td>0.228</td>
      <td>0.117</td>
      <td>0.126</td>
      <td>-0.009</td>
      <td>0.236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>race_priv</td>
      <td>-0.002</td>
      <td>0.496</td>
      <td>0.125</td>
      <td>-0.210</td>
      <td>0.134</td>
      <td>0.016</td>
      <td>0.537</td>
      <td>-1.032</td>
      <td>0.041</td>
      <td>-0.024</td>
      <td>-0.033</td>
      <td>0.409</td>
      <td>0.158</td>
      <td>0.051</td>
      <td>-0.134</td>
      <td>0.210</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>race_dis</td>
      <td>0.158</td>
      <td>1.012</td>
      <td>0.164</td>
      <td>-0.197</td>
      <td>-0.101</td>
      <td>0.018</td>
      <td>0.769</td>
      <td>-1.454</td>
      <td>0.055</td>
      <td>0.129</td>
      <td>-0.216</td>
      <td>0.089</td>
      <td>0.052</td>
      <td>0.172</td>
      <td>0.101</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>5</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.000</td>
      <td>7.311</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.125</td>
      <td>0.504</td>
      <td>-1.236</td>
      <td>0.242</td>
      <td>0.000</td>
      <td>0.097</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.082</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>0.190</td>
      <td>1.083</td>
      <td>0.186</td>
      <td>-0.227</td>
      <td>-0.133</td>
      <td>0.050</td>
      <td>0.833</td>
      <td>-1.526</td>
      <td>0.067</td>
      <td>0.140</td>
      <td>-0.257</td>
      <td>0.109</td>
      <td>0.065</td>
      <td>0.177</td>
      <td>0.133</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>7</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>overall</td>
      <td>0.063</td>
      <td>0.061</td>
      <td>0.231</td>
      <td>-0.465</td>
      <td>0.301</td>
      <td>0.099</td>
      <td>0.269</td>
      <td>-0.311</td>
      <td>-0.016</td>
      <td>-0.096</td>
      <td>-0.023</td>
      <td>0.796</td>
      <td>0.379</td>
      <td>0.084</td>
      <td>-0.301</td>
      <td>0.465</td>
    </tr>
    <tr>
      <th>8</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex_priv</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.185</td>
      <td>-0.392</td>
      <td>0.258</td>
      <td>0.044</td>
      <td>0.248</td>
      <td>-0.379</td>
      <td>-0.020</td>
      <td>-0.083</td>
      <td>-0.074</td>
      <td>0.784</td>
      <td>0.312</td>
      <td>0.069</td>
      <td>-0.258</td>
      <td>0.392</td>
    </tr>
    <tr>
      <th>9</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex_dis</td>
      <td>0.077</td>
      <td>0.067</td>
      <td>0.236</td>
      <td>-0.478</td>
      <td>0.316</td>
      <td>0.111</td>
      <td>0.274</td>
      <td>-0.294</td>
      <td>-0.015</td>
      <td>-0.105</td>
      <td>-0.012</td>
      <td>0.801</td>
      <td>0.397</td>
      <td>0.087</td>
      <td>-0.316</td>
      <td>0.478</td>
    </tr>
    <tr>
      <th>10</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>race_priv</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.189</td>
      <td>-0.198</td>
      <td>0.002</td>
      <td>0.075</td>
      <td>0.209</td>
      <td>-0.246</td>
      <td>-0.030</td>
      <td>0.094</td>
      <td>-0.008</td>
      <td>0.179</td>
      <td>0.075</td>
      <td>0.085</td>
      <td>-0.002</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>11</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>race_dis</td>
      <td>0.053</td>
      <td>0.062</td>
      <td>0.226</td>
      <td>-0.602</td>
      <td>0.565</td>
      <td>0.114</td>
      <td>0.309</td>
      <td>-0.354</td>
      <td>-0.006</td>
      <td>-0.168</td>
      <td>-0.033</td>
      <td>1.105</td>
      <td>0.585</td>
      <td>0.083</td>
      <td>-0.565</td>
      <td>0.602</td>
    </tr>
    <tr>
      <th>12</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex&amp;race_priv</td>
      <td>0.337</td>
      <td>0.000</td>
      <td>0.919</td>
      <td>-0.952</td>
      <td>0.000</td>
      <td>0.067</td>
      <td>0.234</td>
      <td>-0.504</td>
      <td>-0.038</td>
      <td>0.649</td>
      <td>-0.024</td>
      <td>0.952</td>
      <td>0.337</td>
      <td>0.084</td>
      <td>0.000</td>
      <td>0.952</td>
    </tr>
    <tr>
      <th>13</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex&amp;race_dis</td>
      <td>0.129</td>
      <td>0.084</td>
      <td>0.297</td>
      <td>-0.706</td>
      <td>0.580</td>
      <td>0.132</td>
      <td>0.318</td>
      <td>-0.369</td>
      <td>-0.007</td>
      <td>-0.141</td>
      <td>-0.015</td>
      <td>1.180</td>
      <td>0.651</td>
      <td>0.088</td>
      <td>-0.580</td>
      <td>0.706</td>
    </tr>
    <tr>
      <th>14</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>overall</td>
      <td>0.158</td>
      <td>0.026</td>
      <td>0.207</td>
      <td>-0.266</td>
      <td>-0.063</td>
      <td>-0.026</td>
      <td>-0.235</td>
      <td>0.211</td>
      <td>0.009</td>
      <td>0.150</td>
      <td>0.004</td>
      <td>0.202</td>
      <td>0.095</td>
      <td>-0.060</td>
      <td>0.063</td>
      <td>0.266</td>
    </tr>
    <tr>
      <th>15</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex_priv</td>
      <td>0.541</td>
      <td>0.209</td>
      <td>0.767</td>
      <td>-0.606</td>
      <td>-0.513</td>
      <td>-0.053</td>
      <td>0.088</td>
      <td>-0.293</td>
      <td>-0.008</td>
      <td>0.943</td>
      <td>-0.030</td>
      <td>-0.606</td>
      <td>-0.180</td>
      <td>-0.040</td>
      <td>0.513</td>
      <td>0.606</td>
    </tr>
    <tr>
      <th>16</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex_dis</td>
      <td>0.078</td>
      <td>-3.899</td>
      <td>0.129</td>
      <td>-0.224</td>
      <td>0.073</td>
      <td>-0.018</td>
      <td>-0.310</td>
      <td>0.328</td>
      <td>0.014</td>
      <td>0.031</td>
      <td>0.011</td>
      <td>0.299</td>
      <td>0.152</td>
      <td>-0.064</td>
      <td>-0.073</td>
      <td>0.224</td>
    </tr>
    <tr>
      <th>17</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>race_priv</td>
      <td>0.156</td>
      <td>-3.823</td>
      <td>0.235</td>
      <td>-0.208</td>
      <td>-0.134</td>
      <td>-0.046</td>
      <td>-0.319</td>
      <td>0.304</td>
      <td>0.063</td>
      <td>0.286</td>
      <td>-0.112</td>
      <td>0.021</td>
      <td>0.000</td>
      <td>-0.065</td>
      <td>0.134</td>
      <td>0.208</td>
    </tr>
    <tr>
      <th>18</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>race_dis</td>
      <td>0.159</td>
      <td>-0.211</td>
      <td>0.187</td>
      <td>-0.292</td>
      <td>0.000</td>
      <td>-0.013</td>
      <td>-0.178</td>
      <td>0.148</td>
      <td>-0.027</td>
      <td>0.084</td>
      <td>0.081</td>
      <td>0.292</td>
      <td>0.159</td>
      <td>-0.056</td>
      <td>0.000</td>
      <td>0.292</td>
    </tr>
    <tr>
      <th>19</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.000</td>
      <td>-0.060</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.089</td>
      <td>0.014</td>
      <td>-0.120</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>-0.186</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.059</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>0.000</td>
      <td>-4.128</td>
      <td>0.057</td>
      <td>-0.214</td>
      <td>0.294</td>
      <td>-0.009</td>
      <td>-0.247</td>
      <td>0.269</td>
      <td>-0.023</td>
      <td>-0.092</td>
      <td>0.073</td>
      <td>0.427</td>
      <td>0.247</td>
      <td>-0.063</td>
      <td>-0.294</td>
      <td>0.214</td>
    </tr>
    <tr>
      <th>21</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>overall</td>
      <td>-0.126</td>
      <td>-0.391</td>
      <td>-0.348</td>
      <td>0.724</td>
      <td>-0.425</td>
      <td>-0.085</td>
      <td>-0.160</td>
      <td>0.022</td>
      <td>0.057</td>
      <td>0.090</td>
      <td>-0.029</td>
      <td>-1.184</td>
      <td>-0.568</td>
      <td>-0.011</td>
      <td>0.425</td>
      <td>-0.724</td>
    </tr>
    <tr>
      <th>22</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex_priv</td>
      <td>-0.679</td>
      <td>-0.633</td>
      <td>-1.411</td>
      <td>1.989</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.359</td>
      <td>0.627</td>
      <td>-0.008</td>
      <td>-0.831</td>
      <td>0.128</td>
      <td>-1.989</td>
      <td>-0.679</td>
      <td>0.012</td>
      <td>0.000</td>
      <td>-1.989</td>
    </tr>
    <tr>
      <th>23</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex_dis</td>
      <td>0.000</td>
      <td>-0.047</td>
      <td>-0.182</td>
      <td>0.529</td>
      <td>-0.555</td>
      <td>-0.115</td>
      <td>-0.110</td>
      <td>-0.128</td>
      <td>0.073</td>
      <td>0.229</td>
      <td>-0.066</td>
      <td>-1.058</td>
      <td>-0.541</td>
      <td>-0.016</td>
      <td>0.555</td>
      <td>-0.529</td>
    </tr>
    <tr>
      <th>24</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>race_priv</td>
      <td>0.074</td>
      <td>-0.372</td>
      <td>-0.392</td>
      <td>0.812</td>
      <td>-0.645</td>
      <td>-0.107</td>
      <td>-0.387</td>
      <td>0.554</td>
      <td>0.045</td>
      <td>0.262</td>
      <td>0.119</td>
      <td>-1.827</td>
      <td>-0.714</td>
      <td>-0.023</td>
      <td>0.645</td>
      <td>-0.812</td>
    </tr>
    <tr>
      <th>25</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>race_dis</td>
      <td>-0.264</td>
      <td>0.064</td>
      <td>-0.369</td>
      <td>0.687</td>
      <td>-0.236</td>
      <td>-0.069</td>
      <td>-0.007</td>
      <td>-0.332</td>
      <td>0.063</td>
      <td>-0.040</td>
      <td>-0.131</td>
      <td>-0.879</td>
      <td>-0.476</td>
      <td>-0.001</td>
      <td>0.236</td>
      <td>-0.687</td>
    </tr>
    <tr>
      <th>26</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.009</td>
      <td>-7.021</td>
      <td>-1.371</td>
      <td>2.894</td>
      <td>-1.715</td>
      <td>-0.051</td>
      <td>-0.394</td>
      <td>0.983</td>
      <td>0.047</td>
      <td>0.917</td>
      <td>0.295</td>
      <td>-6.052</td>
      <td>-2.201</td>
      <td>-0.007</td>
      <td>1.715</td>
      <td>-2.894</td>
    </tr>
    <tr>
      <th>27</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>-0.067</td>
      <td>-5.041</td>
      <td>-0.216</td>
      <td>0.649</td>
      <td>-0.758</td>
      <td>-0.108</td>
      <td>0.068</td>
      <td>-0.488</td>
      <td>0.090</td>
      <td>0.250</td>
      <td>-0.155</td>
      <td>-1.183</td>
      <td>-0.694</td>
      <td>-0.007</td>
      <td>0.758</td>
      <td>-0.649</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Folktables [GA, 2018] Dataset


```python
dataset2 = ACSEmploymentDataset(state=['GA'], year=2018, root_dir=os.path.join('virny', 'datasets'), with_nulls=False, subsample=20_000)
dataset2.X_data.head()
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
      <th>SEX</th>
      <th>RAC1P</th>
      <th>RELP</th>
      <th>CIT</th>
      <th>ANC</th>
      <th>SCHL</th>
      <th>AGEP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71506</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>61</td>
    </tr>
    <tr>
      <th>64254</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>21</td>
      <td>79</td>
    </tr>
    <tr>
      <th>96328</th>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>69</td>
    </tr>
    <tr>
      <th>47767</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>67</td>
    </tr>
    <tr>
      <th>46198</th>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>14</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
models_config2 = {
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                     max_depth=10,
                                                     max_features=0.6,
                                                     min_samples_split=0.1),
    'LogisticRegression': LogisticRegression(C=1,
                                             max_iter=150,
                                             penalty='l2',
                                             solver='lbfgs'),
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


```python
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = """
dataset_name: Folktables_GA_2018
test_set_fraction: 0.2
bootstrap_fraction: 0.8
n_estimators: 100
runs_seed_lst: [100, 200, 300]
#runs_seed_lst: [100, 200, 300, 400, 500, 600]
sensitive_attributes_dct: {'SEX': '1', 'RAC1P': '1', 'SEX&RAC1P': None}
"""

with open(config_yaml_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml_content)

config2 = create_config_obj(config_yaml_path=config_yaml_path)
SAVE_RESULTS_DIR_PATH2 = os.path.join(ROOT_DIR, 'results', 'benchmark', 'benchmark_results', config2.dataset_name)
STANDARD_RESULTS_DIR_PATH2 = os.path.join(ROOT_DIR, 'results', 'benchmark', 'standard_results', config2.dataset_name)
```


```python
clear_directory(SAVE_RESULTS_DIR_PATH2)
```

    Directory is cleared



```python
multiple_run_metrics_dct2 = compute_metrics_multiple_runs(dataset2, config2, models_config2, SAVE_RESULTS_DIR_PATH2, verbose=1)
```


    Multiple runs progress:   0%|          | 0/3 [00:00<?, ?it/s]



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  101
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:24:35 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:24:39 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:24:51 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  102
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:25:15 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:26:01 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:26:10 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  103
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:26:32 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:29:04 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:29:12 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  104
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:29:37 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:32:08 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:32:17 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  201
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:32:41 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:32:44 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:32:52 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  202
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:33:15 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:33:58 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:34:07 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  203
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:34:30 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:37:00 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:37:09 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  204
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:37:31 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:39:04 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:39:13 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



    Analyze models in one run:   0%|          | 0/4 [00:00<?, ?it/s]


    ##############################  [Model 1 / 4] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  301
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:39:40 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:39:44 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:39:53 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 2 / 4] Analyze LogisticRegression  ##############################
    Model random_state:  302
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:40:19 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:41:20 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:41:34 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 3 / 4] Analyze RandomForestClassifier  ##############################
    Model random_state:  303
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:42:01 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:44:36 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:44:45 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    
    ##############################  [Model 4 / 4] Analyze XGBClassifier  ##############################
    Model random_state:  304
    Baseline X_train shape:  (16000, 16)
    Baseline X_test shape:  (4000, 16)
    
    


    2023-02-04 15:45:08 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap



    Classifiers testing by bootstrap:   0%|          | 0/100 [00:00<?, ?it/s]


    
    


    2023-02-04 15:47:02 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-02-04 15:47:12 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    
    
    



```python
sample_model_metrics_df = multiple_run_metrics_dct2[list(models_config2.keys())[0]]
sample_model_metrics_df.head(20)
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
      <th>SEX&amp;RAC1P_priv</th>
      <th>SEX&amp;RAC1P_dis</th>
      <th>Model_Seed</th>
      <th>Model_Name</th>
      <th>Run_Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.553779</td>
      <td>0.531211</td>
      <td>0.574568</td>
      <td>0.547523</td>
      <td>0.566169</td>
      <td>0.507491</td>
      <td>0.552915</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.053819</td>
      <td>0.057550</td>
      <td>0.050381</td>
      <td>0.053637</td>
      <td>0.054179</td>
      <td>0.056413</td>
      <td>0.049336</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.065751</td>
      <td>0.070943</td>
      <td>0.060969</td>
      <td>0.067190</td>
      <td>0.062902</td>
      <td>0.071738</td>
      <td>0.057622</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jitter</td>
      <td>0.049987</td>
      <td>0.038030</td>
      <td>0.061001</td>
      <td>0.046342</td>
      <td>0.057205</td>
      <td>0.034722</td>
      <td>0.067273</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.827892</td>
      <td>0.855620</td>
      <td>0.802349</td>
      <td>0.829120</td>
      <td>0.825462</td>
      <td>0.866804</td>
      <td>0.820314</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Label_Stability</td>
      <td>0.928835</td>
      <td>0.944442</td>
      <td>0.914457</td>
      <td>0.934612</td>
      <td>0.917392</td>
      <td>0.949725</td>
      <td>0.904290</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TPR</td>
      <td>0.866066</td>
      <td>0.872928</td>
      <td>0.858945</td>
      <td>0.857734</td>
      <td>0.883803</td>
      <td>0.876855</td>
      <td>0.899110</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TNR</td>
      <td>0.810166</td>
      <td>0.856861</td>
      <td>0.771074</td>
      <td>0.816425</td>
      <td>0.798450</td>
      <td>0.884858</td>
      <td>0.787342</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPV</td>
      <td>0.784804</td>
      <td>0.844920</td>
      <td>0.730019</td>
      <td>0.795856</td>
      <td>0.762918</td>
      <td>0.890060</td>
      <td>0.782946</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FNR</td>
      <td>0.133934</td>
      <td>0.127072</td>
      <td>0.141055</td>
      <td>0.142266</td>
      <td>0.116197</td>
      <td>0.123145</td>
      <td>0.100890</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FPR</td>
      <td>0.189834</td>
      <td>0.143139</td>
      <td>0.228926</td>
      <td>0.183575</td>
      <td>0.201550</td>
      <td>0.115142</td>
      <td>0.212658</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Accuracy</td>
      <td>0.835000</td>
      <td>0.864442</td>
      <td>0.807877</td>
      <td>0.835214</td>
      <td>0.834575</td>
      <td>0.880734</td>
      <td>0.838798</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F1</td>
      <td>0.823435</td>
      <td>0.858696</td>
      <td>0.789252</td>
      <td>0.825637</td>
      <td>0.818923</td>
      <td>0.883408</td>
      <td>0.837017</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Selection-Rate</td>
      <td>0.490250</td>
      <td>0.487487</td>
      <td>0.492795</td>
      <td>0.490218</td>
      <td>0.490313</td>
      <td>0.507645</td>
      <td>0.528689</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Positive-Rate</td>
      <td>1.103545</td>
      <td>1.033149</td>
      <td>1.176606</td>
      <td>1.077750</td>
      <td>1.158451</td>
      <td>0.985163</td>
      <td>1.148368</td>
      <td>101</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>0.553985</td>
      <td>0.533772</td>
      <td>0.572457</td>
      <td>0.547070</td>
      <td>0.567484</td>
      <td>0.510165</td>
      <td>0.554548</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Std</td>
      <td>0.050968</td>
      <td>0.052652</td>
      <td>0.049429</td>
      <td>0.050290</td>
      <td>0.052291</td>
      <td>0.051924</td>
      <td>0.050709</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IQR</td>
      <td>0.061006</td>
      <td>0.065531</td>
      <td>0.056871</td>
      <td>0.061177</td>
      <td>0.060673</td>
      <td>0.065779</td>
      <td>0.056998</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>201</td>
      <td>DecisionTreeClassifier</td>
      <td>Run_2</td>
    </tr>
  </tbody>
</table>
</div>




```python
models_metrics_dct2 = read_model_metric_dfs(SAVE_RESULTS_DIR_PATH2, model_names=models_config2.keys())
standard_models_metrics_dct2 = read_model_metric_dfs(STANDARD_RESULTS_DIR_PATH2, model_names=models_config2.keys())

avg_models_metrics_dct2 = create_averaged_dfs_dict(models_metrics_dct2)
avg_standard_models_metrics_dct2 = create_averaged_dfs_dict(standard_models_metrics_dct2)
```

## Create a Benchmark Report


```python
report_df = populate_benchmark_report(report_df, avg_models_metrics_dct2, avg_standard_models_metrics_dct2,
                                      config2.dataset_name, config2.sensitive_attributes_dct)
```


```python
filename = f'benchmark_report_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}'
report_df.to_csv(os.path.join(ROOT_DIR, 'results', 'benchmark', 'benchmark_reports', f'{filename}.csv'), index=False)
```


```python
report_df
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
      <th>Dataset</th>
      <th>Model</th>
      <th>Subgroup</th>
      <th>Delta%_Accuracy</th>
      <th>Delta%_Entropy</th>
      <th>Delta%_F1</th>
      <th>Delta%_FNR</th>
      <th>Delta%_FPR</th>
      <th>Delta%_IQR</th>
      <th>Delta%_Jitter</th>
      <th>Delta%_Label_Stability</th>
      <th>Delta%_Mean</th>
      <th>Delta%_PPV</th>
      <th>Delta%_Per_Sample_Accuracy</th>
      <th>Delta%_Positive-Rate</th>
      <th>Delta%_Selection-Rate</th>
      <th>Delta%_Std</th>
      <th>Delta%_TNR</th>
      <th>Delta%_TPR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>overall</td>
      <td>0.095</td>
      <td>0.775</td>
      <td>0.146</td>
      <td>-0.201</td>
      <td>0.008</td>
      <td>0.016</td>
      <td>0.676</td>
      <td>-1.284</td>
      <td>0.049</td>
      <td>0.076</td>
      <td>-0.145</td>
      <td>0.192</td>
      <td>0.095</td>
      <td>0.124</td>
      <td>-0.008</td>
      <td>0.201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex_priv</td>
      <td>0.006</td>
      <td>0.538</td>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.021</td>
      <td>0.479</td>
      <td>-1.167</td>
      <td>0.108</td>
      <td>0.015</td>
      <td>0.035</td>
      <td>-0.049</td>
      <td>-0.006</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex_dis</td>
      <td>0.117</td>
      <td>0.829</td>
      <td>0.167</td>
      <td>-0.236</td>
      <td>0.009</td>
      <td>0.024</td>
      <td>0.723</td>
      <td>-1.312</td>
      <td>0.035</td>
      <td>0.079</td>
      <td>-0.187</td>
      <td>0.228</td>
      <td>0.117</td>
      <td>0.126</td>
      <td>-0.009</td>
      <td>0.236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>race_priv</td>
      <td>-0.002</td>
      <td>0.496</td>
      <td>0.125</td>
      <td>-0.210</td>
      <td>0.134</td>
      <td>0.016</td>
      <td>0.537</td>
      <td>-1.032</td>
      <td>0.041</td>
      <td>-0.024</td>
      <td>-0.033</td>
      <td>0.409</td>
      <td>0.158</td>
      <td>0.051</td>
      <td>-0.134</td>
      <td>0.210</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>race_dis</td>
      <td>0.158</td>
      <td>1.012</td>
      <td>0.164</td>
      <td>-0.197</td>
      <td>-0.101</td>
      <td>0.018</td>
      <td>0.769</td>
      <td>-1.454</td>
      <td>0.055</td>
      <td>0.129</td>
      <td>-0.216</td>
      <td>0.089</td>
      <td>0.052</td>
      <td>0.172</td>
      <td>0.101</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>5</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.000</td>
      <td>7.311</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.125</td>
      <td>0.504</td>
      <td>-1.236</td>
      <td>0.242</td>
      <td>0.000</td>
      <td>0.097</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.082</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>0.190</td>
      <td>1.083</td>
      <td>0.186</td>
      <td>-0.227</td>
      <td>-0.133</td>
      <td>0.050</td>
      <td>0.833</td>
      <td>-1.526</td>
      <td>0.067</td>
      <td>0.140</td>
      <td>-0.257</td>
      <td>0.109</td>
      <td>0.065</td>
      <td>0.177</td>
      <td>0.133</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>7</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>overall</td>
      <td>0.063</td>
      <td>0.061</td>
      <td>0.231</td>
      <td>-0.465</td>
      <td>0.301</td>
      <td>0.099</td>
      <td>0.269</td>
      <td>-0.311</td>
      <td>-0.016</td>
      <td>-0.096</td>
      <td>-0.023</td>
      <td>0.796</td>
      <td>0.379</td>
      <td>0.084</td>
      <td>-0.301</td>
      <td>0.465</td>
    </tr>
    <tr>
      <th>8</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex_priv</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.185</td>
      <td>-0.392</td>
      <td>0.258</td>
      <td>0.044</td>
      <td>0.248</td>
      <td>-0.379</td>
      <td>-0.020</td>
      <td>-0.083</td>
      <td>-0.074</td>
      <td>0.784</td>
      <td>0.312</td>
      <td>0.069</td>
      <td>-0.258</td>
      <td>0.392</td>
    </tr>
    <tr>
      <th>9</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex_dis</td>
      <td>0.077</td>
      <td>0.067</td>
      <td>0.236</td>
      <td>-0.478</td>
      <td>0.316</td>
      <td>0.111</td>
      <td>0.274</td>
      <td>-0.294</td>
      <td>-0.015</td>
      <td>-0.105</td>
      <td>-0.012</td>
      <td>0.801</td>
      <td>0.397</td>
      <td>0.087</td>
      <td>-0.316</td>
      <td>0.478</td>
    </tr>
    <tr>
      <th>10</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>race_priv</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.189</td>
      <td>-0.198</td>
      <td>0.002</td>
      <td>0.075</td>
      <td>0.209</td>
      <td>-0.246</td>
      <td>-0.030</td>
      <td>0.094</td>
      <td>-0.008</td>
      <td>0.179</td>
      <td>0.075</td>
      <td>0.085</td>
      <td>-0.002</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>11</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>race_dis</td>
      <td>0.053</td>
      <td>0.062</td>
      <td>0.226</td>
      <td>-0.602</td>
      <td>0.565</td>
      <td>0.114</td>
      <td>0.309</td>
      <td>-0.354</td>
      <td>-0.006</td>
      <td>-0.168</td>
      <td>-0.033</td>
      <td>1.105</td>
      <td>0.585</td>
      <td>0.083</td>
      <td>-0.565</td>
      <td>0.602</td>
    </tr>
    <tr>
      <th>12</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex&amp;race_priv</td>
      <td>0.337</td>
      <td>0.000</td>
      <td>0.919</td>
      <td>-0.952</td>
      <td>0.000</td>
      <td>0.067</td>
      <td>0.234</td>
      <td>-0.504</td>
      <td>-0.038</td>
      <td>0.649</td>
      <td>-0.024</td>
      <td>0.952</td>
      <td>0.337</td>
      <td>0.084</td>
      <td>0.000</td>
      <td>0.952</td>
    </tr>
    <tr>
      <th>13</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>sex&amp;race_dis</td>
      <td>0.129</td>
      <td>0.084</td>
      <td>0.297</td>
      <td>-0.706</td>
      <td>0.580</td>
      <td>0.132</td>
      <td>0.318</td>
      <td>-0.369</td>
      <td>-0.007</td>
      <td>-0.141</td>
      <td>-0.015</td>
      <td>1.180</td>
      <td>0.651</td>
      <td>0.088</td>
      <td>-0.580</td>
      <td>0.706</td>
    </tr>
    <tr>
      <th>14</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>overall</td>
      <td>0.158</td>
      <td>0.026</td>
      <td>0.207</td>
      <td>-0.266</td>
      <td>-0.063</td>
      <td>-0.026</td>
      <td>-0.235</td>
      <td>0.211</td>
      <td>0.009</td>
      <td>0.150</td>
      <td>0.004</td>
      <td>0.202</td>
      <td>0.095</td>
      <td>-0.060</td>
      <td>0.063</td>
      <td>0.266</td>
    </tr>
    <tr>
      <th>15</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex_priv</td>
      <td>0.541</td>
      <td>0.209</td>
      <td>0.767</td>
      <td>-0.606</td>
      <td>-0.513</td>
      <td>-0.053</td>
      <td>0.088</td>
      <td>-0.293</td>
      <td>-0.008</td>
      <td>0.943</td>
      <td>-0.030</td>
      <td>-0.606</td>
      <td>-0.180</td>
      <td>-0.040</td>
      <td>0.513</td>
      <td>0.606</td>
    </tr>
    <tr>
      <th>16</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex_dis</td>
      <td>0.078</td>
      <td>-3.899</td>
      <td>0.129</td>
      <td>-0.224</td>
      <td>0.073</td>
      <td>-0.018</td>
      <td>-0.310</td>
      <td>0.328</td>
      <td>0.014</td>
      <td>0.031</td>
      <td>0.011</td>
      <td>0.299</td>
      <td>0.152</td>
      <td>-0.064</td>
      <td>-0.073</td>
      <td>0.224</td>
    </tr>
    <tr>
      <th>17</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>race_priv</td>
      <td>0.156</td>
      <td>-3.823</td>
      <td>0.235</td>
      <td>-0.208</td>
      <td>-0.134</td>
      <td>-0.046</td>
      <td>-0.319</td>
      <td>0.304</td>
      <td>0.063</td>
      <td>0.286</td>
      <td>-0.112</td>
      <td>0.021</td>
      <td>0.000</td>
      <td>-0.065</td>
      <td>0.134</td>
      <td>0.208</td>
    </tr>
    <tr>
      <th>18</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>race_dis</td>
      <td>0.159</td>
      <td>-0.211</td>
      <td>0.187</td>
      <td>-0.292</td>
      <td>0.000</td>
      <td>-0.013</td>
      <td>-0.178</td>
      <td>0.148</td>
      <td>-0.027</td>
      <td>0.084</td>
      <td>0.081</td>
      <td>0.292</td>
      <td>0.159</td>
      <td>-0.056</td>
      <td>0.000</td>
      <td>0.292</td>
    </tr>
    <tr>
      <th>19</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.000</td>
      <td>-0.060</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.089</td>
      <td>0.014</td>
      <td>-0.120</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>-0.186</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-0.059</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>0.000</td>
      <td>-4.128</td>
      <td>0.057</td>
      <td>-0.214</td>
      <td>0.294</td>
      <td>-0.009</td>
      <td>-0.247</td>
      <td>0.269</td>
      <td>-0.023</td>
      <td>-0.092</td>
      <td>0.073</td>
      <td>0.427</td>
      <td>0.247</td>
      <td>-0.063</td>
      <td>-0.294</td>
      <td>0.214</td>
    </tr>
    <tr>
      <th>21</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>overall</td>
      <td>-0.126</td>
      <td>-0.391</td>
      <td>-0.348</td>
      <td>0.724</td>
      <td>-0.425</td>
      <td>-0.085</td>
      <td>-0.160</td>
      <td>0.022</td>
      <td>0.057</td>
      <td>0.090</td>
      <td>-0.029</td>
      <td>-1.184</td>
      <td>-0.568</td>
      <td>-0.011</td>
      <td>0.425</td>
      <td>-0.724</td>
    </tr>
    <tr>
      <th>22</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex_priv</td>
      <td>-0.679</td>
      <td>-0.633</td>
      <td>-1.411</td>
      <td>1.989</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>-0.359</td>
      <td>0.627</td>
      <td>-0.008</td>
      <td>-0.831</td>
      <td>0.128</td>
      <td>-1.989</td>
      <td>-0.679</td>
      <td>0.012</td>
      <td>0.000</td>
      <td>-1.989</td>
    </tr>
    <tr>
      <th>23</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex_dis</td>
      <td>0.000</td>
      <td>-0.047</td>
      <td>-0.182</td>
      <td>0.529</td>
      <td>-0.555</td>
      <td>-0.115</td>
      <td>-0.110</td>
      <td>-0.128</td>
      <td>0.073</td>
      <td>0.229</td>
      <td>-0.066</td>
      <td>-1.058</td>
      <td>-0.541</td>
      <td>-0.016</td>
      <td>0.555</td>
      <td>-0.529</td>
    </tr>
    <tr>
      <th>24</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>race_priv</td>
      <td>0.074</td>
      <td>-0.372</td>
      <td>-0.392</td>
      <td>0.812</td>
      <td>-0.645</td>
      <td>-0.107</td>
      <td>-0.387</td>
      <td>0.554</td>
      <td>0.045</td>
      <td>0.262</td>
      <td>0.119</td>
      <td>-1.827</td>
      <td>-0.714</td>
      <td>-0.023</td>
      <td>0.645</td>
      <td>-0.812</td>
    </tr>
    <tr>
      <th>25</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>race_dis</td>
      <td>-0.264</td>
      <td>0.064</td>
      <td>-0.369</td>
      <td>0.687</td>
      <td>-0.236</td>
      <td>-0.069</td>
      <td>-0.007</td>
      <td>-0.332</td>
      <td>0.063</td>
      <td>-0.040</td>
      <td>-0.131</td>
      <td>-0.879</td>
      <td>-0.476</td>
      <td>-0.001</td>
      <td>0.236</td>
      <td>-0.687</td>
    </tr>
    <tr>
      <th>26</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex&amp;race_priv</td>
      <td>0.009</td>
      <td>-7.021</td>
      <td>-1.371</td>
      <td>2.894</td>
      <td>-1.715</td>
      <td>-0.051</td>
      <td>-0.394</td>
      <td>0.983</td>
      <td>0.047</td>
      <td>0.917</td>
      <td>0.295</td>
      <td>-6.052</td>
      <td>-2.201</td>
      <td>-0.007</td>
      <td>1.715</td>
      <td>-2.894</td>
    </tr>
    <tr>
      <th>27</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>sex&amp;race_dis</td>
      <td>-0.067</td>
      <td>-5.041</td>
      <td>-0.216</td>
      <td>0.649</td>
      <td>-0.758</td>
      <td>-0.108</td>
      <td>0.068</td>
      <td>-0.488</td>
      <td>0.090</td>
      <td>0.250</td>
      <td>-0.155</td>
      <td>-1.183</td>
      <td>-0.694</td>
      <td>-0.007</td>
      <td>0.758</td>
      <td>-0.649</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>overall</td>
      <td>0.008</td>
      <td>2.534</td>
      <td>-0.015</td>
      <td>0.773</td>
      <td>-0.599</td>
      <td>0.571</td>
      <td>0.246</td>
      <td>-0.534</td>
      <td>0.005</td>
      <td>0.607</td>
      <td>-0.136</td>
      <td>-1.862</td>
      <td>-0.450</td>
      <td>0.312</td>
      <td>0.599</td>
      <td>-0.773</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>SEX_priv</td>
      <td>-0.404</td>
      <td>1.982</td>
      <td>-0.608</td>
      <td>1.580</td>
      <td>-0.648</td>
      <td>0.420</td>
      <td>0.626</td>
      <td>-1.257</td>
      <td>-0.009</td>
      <td>0.337</td>
      <td>-0.694</td>
      <td>-2.285</td>
      <td>-1.123</td>
      <td>0.265</td>
      <td>0.648</td>
      <td>-1.580</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>SEX_dis</td>
      <td>0.343</td>
      <td>0.000</td>
      <td>0.575</td>
      <td>-0.072</td>
      <td>-0.422</td>
      <td>0.704</td>
      <td>-0.095</td>
      <td>0.123</td>
      <td>0.049</td>
      <td>0.913</td>
      <td>0.335</td>
      <td>-1.445</td>
      <td>0.183</td>
      <td>0.354</td>
      <td>0.422</td>
      <td>0.072</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>RAC1P_priv</td>
      <td>0.055</td>
      <td>2.503</td>
      <td>0.109</td>
      <td>0.751</td>
      <td>-0.710</td>
      <td>0.508</td>
      <td>0.273</td>
      <td>-0.559</td>
      <td>-0.304</td>
      <td>0.865</td>
      <td>-0.138</td>
      <td>-2.095</td>
      <td>-0.319</td>
      <td>0.330</td>
      <td>0.710</td>
      <td>-0.751</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>RAC1P_dis</td>
      <td>-0.053</td>
      <td>0.000</td>
      <td>-0.215</td>
      <td>0.836</td>
      <td>-0.473</td>
      <td>0.696</td>
      <td>0.167</td>
      <td>-0.447</td>
      <td>0.558</td>
      <td>0.223</td>
      <td>-0.104</td>
      <td>-1.540</td>
      <td>-0.698</td>
      <td>0.275</td>
      <td>0.473</td>
      <td>-0.836</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>SEX&amp;RAC1P_priv</td>
      <td>-0.678</td>
      <td>2.032</td>
      <td>-0.921</td>
      <td>1.774</td>
      <td>-0.453</td>
      <td>0.384</td>
      <td>0.752</td>
      <td>-1.375</td>
      <td>-0.078</td>
      <td>-0.017</td>
      <td>-0.812</td>
      <td>-2.001</td>
      <td>-1.416</td>
      <td>0.352</td>
      <td>0.453</td>
      <td>-1.774</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Folktables_GA_2018</td>
      <td>DecisionTreeClassifier</td>
      <td>SEX&amp;RAC1P_dis</td>
      <td>-0.484</td>
      <td>0.000</td>
      <td>-0.870</td>
      <td>0.519</td>
      <td>0.278</td>
      <td>0.840</td>
      <td>0.089</td>
      <td>-0.076</td>
      <td>1.145</td>
      <td>-1.151</td>
      <td>-0.040</td>
      <td>0.979</td>
      <td>-0.845</td>
      <td>0.433</td>
      <td>-0.278</td>
      <td>-0.519</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>overall</td>
      <td>0.317</td>
      <td>-3.304</td>
      <td>0.428</td>
      <td>-0.881</td>
      <td>0.147</td>
      <td>0.017</td>
      <td>0.107</td>
      <td>-0.166</td>
      <td>-0.302</td>
      <td>0.056</td>
      <td>0.188</td>
      <td>1.077</td>
      <td>0.467</td>
      <td>0.012</td>
      <td>-0.147</td>
      <td>0.881</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>SEX_priv</td>
      <td>0.106</td>
      <td>-1.514</td>
      <td>0.176</td>
      <td>-2.226</td>
      <td>1.651</td>
      <td>-0.037</td>
      <td>0.070</td>
      <td>-0.128</td>
      <td>-0.769</td>
      <td>-1.467</td>
      <td>0.058</td>
      <td>4.880</td>
      <td>1.341</td>
      <td>-0.029</td>
      <td>-1.651</td>
      <td>2.226</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>SEX_dis</td>
      <td>0.506</td>
      <td>0.067</td>
      <td>0.690</td>
      <td>0.345</td>
      <td>-1.186</td>
      <td>0.069</td>
      <td>0.145</td>
      <td>-0.207</td>
      <td>0.172</td>
      <td>1.638</td>
      <td>0.298</td>
      <td>-2.796</td>
      <td>-0.402</td>
      <td>0.051</td>
      <td>1.186</td>
      <td>-0.345</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>RAC1P_priv</td>
      <td>-0.170</td>
      <td>-2.993</td>
      <td>0.004</td>
      <td>-0.648</td>
      <td>0.887</td>
      <td>0.044</td>
      <td>0.112</td>
      <td>-0.197</td>
      <td>-0.356</td>
      <td>-0.521</td>
      <td>-0.125</td>
      <td>1.595</td>
      <td>0.830</td>
      <td>0.024</td>
      <td>-0.887</td>
      <td>0.648</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>RAC1P_dis</td>
      <td>1.288</td>
      <td>1.898</td>
      <td>1.380</td>
      <td>-1.431</td>
      <td>-1.167</td>
      <td>-0.041</td>
      <td>0.083</td>
      <td>-0.086</td>
      <td>-0.276</td>
      <td>1.310</td>
      <td>0.822</td>
      <td>-0.123</td>
      <td>-0.173</td>
      <td>-0.018</td>
      <td>1.167</td>
      <td>1.431</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>SEX&amp;RAC1P_priv</td>
      <td>0.031</td>
      <td>-1.490</td>
      <td>0.079</td>
      <td>-2.130</td>
      <td>1.929</td>
      <td>-0.005</td>
      <td>-0.062</td>
      <td>0.048</td>
      <td>-0.448</td>
      <td>-1.617</td>
      <td>0.079</td>
      <td>4.834</td>
      <td>1.346</td>
      <td>-0.010</td>
      <td>-1.929</td>
      <td>2.130</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Folktables_GA_2018</td>
      <td>LogisticRegression</td>
      <td>SEX&amp;RAC1P_dis</td>
      <td>2.165</td>
      <td>2.029</td>
      <td>1.977</td>
      <td>-0.514</td>
      <td>-3.525</td>
      <td>0.019</td>
      <td>-0.124</td>
      <td>0.241</td>
      <td>0.918</td>
      <td>3.204</td>
      <td>1.483</td>
      <td>-4.145</td>
      <td>-1.713</td>
      <td>0.034</td>
      <td>3.525</td>
      <td>0.514</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>overall</td>
      <td>0.100</td>
      <td>-0.168</td>
      <td>0.119</td>
      <td>-0.714</td>
      <td>0.351</td>
      <td>-0.010</td>
      <td>-0.173</td>
      <td>0.325</td>
      <td>-0.063</td>
      <td>-0.334</td>
      <td>0.171</td>
      <td>1.446</td>
      <td>0.375</td>
      <td>-0.025</td>
      <td>-0.351</td>
      <td>0.714</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>SEX_priv</td>
      <td>0.432</td>
      <td>-1.453</td>
      <td>0.498</td>
      <td>-0.543</td>
      <td>-0.296</td>
      <td>-0.040</td>
      <td>0.283</td>
      <td>-0.416</td>
      <td>-0.332</td>
      <td>0.458</td>
      <td>0.406</td>
      <td>0.074</td>
      <td>0.227</td>
      <td>-0.035</td>
      <td>0.296</td>
      <td>0.543</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>SEX_dis</td>
      <td>-0.082</td>
      <td>1.899</td>
      <td>-0.106</td>
      <td>-0.937</td>
      <td>0.662</td>
      <td>0.014</td>
      <td>-0.607</td>
      <td>1.023</td>
      <td>0.097</td>
      <td>-0.776</td>
      <td>0.078</td>
      <td>2.746</td>
      <td>0.496</td>
      <td>-0.018</td>
      <td>-0.662</td>
      <td>0.937</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>RAC1P_priv</td>
      <td>0.404</td>
      <td>-0.290</td>
      <td>0.323</td>
      <td>-0.711</td>
      <td>-0.211</td>
      <td>0.027</td>
      <td>-0.254</td>
      <td>0.457</td>
      <td>0.058</td>
      <td>0.013</td>
      <td>0.453</td>
      <td>0.910</td>
      <td>-0.068</td>
      <td>0.011</td>
      <td>0.211</td>
      <td>0.711</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>RAC1P_dis</td>
      <td>-0.515</td>
      <td>1.943</td>
      <td>-0.309</td>
      <td>-0.688</td>
      <td>1.391</td>
      <td>-0.085</td>
      <td>-0.018</td>
      <td>0.067</td>
      <td>-0.322</td>
      <td>-0.989</td>
      <td>-0.402</td>
      <td>2.588</td>
      <td>1.263</td>
      <td>-0.096</td>
      <td>-1.391</td>
      <td>0.688</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>SEX&amp;RAC1P_priv</td>
      <td>0.435</td>
      <td>-1.536</td>
      <td>0.216</td>
      <td>-0.070</td>
      <td>-0.856</td>
      <td>0.021</td>
      <td>0.190</td>
      <td>-0.227</td>
      <td>0.310</td>
      <td>0.358</td>
      <td>0.482</td>
      <td>-0.332</td>
      <td>-0.913</td>
      <td>0.021</td>
      <td>0.856</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Folktables_GA_2018</td>
      <td>RandomForestClassifier</td>
      <td>SEX&amp;RAC1P_dis</td>
      <td>-1.393</td>
      <td>-2.220</td>
      <td>-1.926</td>
      <td>0.281</td>
      <td>1.799</td>
      <td>-0.010</td>
      <td>-0.438</td>
      <td>0.815</td>
      <td>1.044</td>
      <td>-2.910</td>
      <td>-1.022</td>
      <td>4.817</td>
      <td>-0.035</td>
      <td>-0.046</td>
      <td>-1.799</td>
      <td>-0.281</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>overall</td>
      <td>-0.325</td>
      <td>-1.253</td>
      <td>-0.700</td>
      <td>0.230</td>
      <td>0.248</td>
      <td>0.027</td>
      <td>-0.044</td>
      <td>0.023</td>
      <td>0.689</td>
      <td>-1.079</td>
      <td>-0.348</td>
      <td>1.243</td>
      <td>-0.767</td>
      <td>0.026</td>
      <td>-0.248</td>
      <td>-0.230</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>SEX_priv</td>
      <td>-1.135</td>
      <td>0.000</td>
      <td>-1.308</td>
      <td>1.157</td>
      <td>1.062</td>
      <td>0.010</td>
      <td>-0.114</td>
      <td>0.115</td>
      <td>0.377</td>
      <td>-1.436</td>
      <td>-1.214</td>
      <td>0.469</td>
      <td>-0.526</td>
      <td>0.020</td>
      <td>-1.062</td>
      <td>-1.157</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>SEX_dis</td>
      <td>0.559</td>
      <td>-1.343</td>
      <td>0.038</td>
      <td>-0.753</td>
      <td>-0.603</td>
      <td>0.038</td>
      <td>0.023</td>
      <td>-0.065</td>
      <td>0.906</td>
      <td>-0.537</td>
      <td>0.588</td>
      <td>1.793</td>
      <td>-0.963</td>
      <td>0.029</td>
      <td>0.603</td>
      <td>0.753</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>RAC1P_priv</td>
      <td>-0.393</td>
      <td>0.000</td>
      <td>-0.743</td>
      <td>0.188</td>
      <td>0.427</td>
      <td>0.022</td>
      <td>-0.023</td>
      <td>-0.018</td>
      <td>0.674</td>
      <td>-1.220</td>
      <td>-0.426</td>
      <td>1.407</td>
      <td>-0.723</td>
      <td>0.017</td>
      <td>-0.427</td>
      <td>-0.188</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>RAC1P_dis</td>
      <td>-0.196</td>
      <td>-0.062</td>
      <td>-0.614</td>
      <td>0.318</td>
      <td>-0.057</td>
      <td>0.036</td>
      <td>-0.086</td>
      <td>0.104</td>
      <td>0.704</td>
      <td>-0.819</td>
      <td>-0.201</td>
      <td>0.867</td>
      <td>-0.841</td>
      <td>0.043</td>
      <td>0.057</td>
      <td>-0.318</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>SEX&amp;RAC1P_priv</td>
      <td>-1.000</td>
      <td>0.000</td>
      <td>-1.275</td>
      <td>0.098</td>
      <td>1.764</td>
      <td>0.013</td>
      <td>-0.039</td>
      <td>-0.009</td>
      <td>0.569</td>
      <td>-2.349</td>
      <td>-1.084</td>
      <td>2.760</td>
      <td>-0.706</td>
      <td>0.017</td>
      <td>-1.764</td>
      <td>-0.098</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Folktables_GA_2018</td>
      <td>XGBClassifier</td>
      <td>SEX&amp;RAC1P_dis</td>
      <td>1.090</td>
      <td>-1.609</td>
      <td>-0.009</td>
      <td>-2.652</td>
      <td>-0.634</td>
      <td>0.052</td>
      <td>0.055</td>
      <td>-0.100</td>
      <td>1.346</td>
      <td>-1.882</td>
      <td>1.149</td>
      <td>6.785</td>
      <td>-1.514</td>
      <td>0.049</td>
      <td>0.634</td>
      <td>2.652</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
