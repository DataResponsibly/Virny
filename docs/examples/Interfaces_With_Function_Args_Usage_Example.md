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
if cur_folder_name != "fairness-variance":
    os.chdir("..")

print('Current location: ', os.getcwd())
```

    Current location:  /home/denys_herasymuk/UCU/4course_2term/Bachelor_Thesis/Code/fairness-variance


## Import dependencies


```python
import os
import pandas as pd
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from source.custom_initializers import create_config_obj
from source.custom_classes.data_loaders import CompasWithoutSensitiveAttrsDataset
from source.metrics_computation_interfaces import run_metrics_computation, compute_model_metrics
```

## Configs


```python
config = create_config_obj(config_yaml_path=os.path.join('configs', 'experiment1_compas_config.yaml'))
SAVE_RESULTS_DIR_PATH = os.path.join('results', 'hypothesis_space',
                                     f'{config.dataset_name}_Metrics_{datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")}')
```


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
}
```

## Load dataset


```python
dataset = CompasWithoutSensitiveAttrsDataset(dataset_path='data/COMPAS.csv')
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



## Get metrics for a base model with a compute_model_metrics function and input arguments


```python
model_name = 'DecisionTreeClassifier'
metrics_df = compute_model_metrics(models_config[model_name], config.n_estimators,
                                   dataset, config.test_set_fraction,
                                   config.bootstrap_fraction, config.sensitive_attributes_dct,
                                   model_seed=101,
                                   dataset_name=config.dataset_name,
                                   base_model_name=model_name,
                                   save_results=True,
                                   save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                   debug_mode=False)
print('Subgroups statistical bias and variance metrics: ')
metrics_df
```

    Model random_state:  101
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-01-27 00:50:32 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:00<00:00, 176.21it/s]

    
    


    
    2023-01-27 00:50:32 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:50:36 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    Subgroups statistical bias and variance metrics: 





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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>General_Ensemble_Accuracy</td>
      <td>0.679924</td>
      <td>0.693467</td>
      <td>0.676779</td>
      <td>0.669789</td>
      <td>0.686804</td>
      <td>0.659091</td>
      <td>0.679537</td>
      <td>101</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mean</td>
      <td>0.525578</td>
      <td>0.561690</td>
      <td>0.517193</td>
      <td>0.590746</td>
      <td>0.481339</td>
      <td>0.589092</td>
      <td>0.468776</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Std</td>
      <td>0.071635</td>
      <td>0.078384</td>
      <td>0.070068</td>
      <td>0.069772</td>
      <td>0.072900</td>
      <td>0.088442</td>
      <td>0.073433</td>
      <td>101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IQR</td>
      <td>0.089278</td>
      <td>0.096598</td>
      <td>0.087578</td>
      <td>0.090645</td>
      <td>0.088350</td>
      <td>0.113437</td>
      <td>0.089443</td>
      <td>101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entropy</td>
      <td>0.000000</td>
      <td>0.216088</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.207275</td>
      <td>0.219363</td>
      <td>0.205943</td>
      <td>101</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jitter</td>
      <td>0.122677</td>
      <td>0.139908</td>
      <td>0.118676</td>
      <td>0.109246</td>
      <td>0.131795</td>
      <td>0.141933</td>
      <td>0.130401</td>
      <td>101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.662689</td>
      <td>0.684724</td>
      <td>0.657573</td>
      <td>0.659180</td>
      <td>0.665072</td>
      <td>0.649773</td>
      <td>0.654923</td>
      <td>101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Label_Stability</td>
      <td>0.830152</td>
      <td>0.798191</td>
      <td>0.837573</td>
      <td>0.843794</td>
      <td>0.820890</td>
      <td>0.792727</td>
      <td>0.824826</td>
      <td>101</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TPR</td>
      <td>0.622177</td>
      <td>0.557143</td>
      <td>0.633094</td>
      <td>0.459119</td>
      <td>0.701220</td>
      <td>0.440000</td>
      <td>0.713781</td>
      <td>101</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TNR</td>
      <td>0.729350</td>
      <td>0.767442</td>
      <td>0.718182</td>
      <td>0.794776</td>
      <td>0.671096</td>
      <td>0.746032</td>
      <td>0.638298</td>
      <td>101</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PPV</td>
      <td>0.663020</td>
      <td>0.565217</td>
      <td>0.680412</td>
      <td>0.570312</td>
      <td>0.699088</td>
      <td>0.407407</td>
      <td>0.703833</td>
      <td>101</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FNR</td>
      <td>0.377823</td>
      <td>0.442857</td>
      <td>0.366906</td>
      <td>0.540881</td>
      <td>0.298780</td>
      <td>0.560000</td>
      <td>0.286219</td>
      <td>101</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FPR</td>
      <td>0.270650</td>
      <td>0.232558</td>
      <td>0.281818</td>
      <td>0.205224</td>
      <td>0.328904</td>
      <td>0.253968</td>
      <td>0.361702</td>
      <td>101</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Accuracy</td>
      <td>0.679924</td>
      <td>0.693467</td>
      <td>0.676779</td>
      <td>0.669789</td>
      <td>0.686804</td>
      <td>0.659091</td>
      <td>0.679537</td>
      <td>101</td>
    </tr>
    <tr>
      <th>14</th>
      <td>F1</td>
      <td>0.641949</td>
      <td>0.561151</td>
      <td>0.655901</td>
      <td>0.508711</td>
      <td>0.700152</td>
      <td>0.423077</td>
      <td>0.708772</td>
      <td>101</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Selection-Rate</td>
      <td>0.432765</td>
      <td>0.346734</td>
      <td>0.452742</td>
      <td>0.299766</td>
      <td>0.523052</td>
      <td>0.306818</td>
      <td>0.554054</td>
      <td>101</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Positive-Rate</td>
      <td>0.938398</td>
      <td>0.985714</td>
      <td>0.930456</td>
      <td>0.805031</td>
      <td>1.003049</td>
      <td>1.080000</td>
      <td>1.014134</td>
      <td>101</td>
    </tr>
  </tbody>
</table>
</div>



## Get metrics for a list of models with a run_metrics_computation function and input arguments


```python
models_metrics_dct = run_metrics_computation(dataset, config.test_set_fraction, config.bootstrap_fraction,
                                             config.dataset_name, models_config, config.n_estimators,
                                             config.sensitive_attributes_dct,
                                             model_seed=200,
                                             save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                             save_results=True,
                                             debug_mode=False)
```

    Analyze models in one run:   0%|[31m          [0m| 0/2 [00:00<?, ?it/s]

    ##############################  [Model 1 / 2] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  201
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-01-27 00:50:44 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:00<00:00, 205.15it/s]

    
    


    
    2023-01-27 00:50:44 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:50:47 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics
    Analyze models in one run:  50%|[31m█████     [0m| 1/2 [00:11<00:11, 11.13s/it]

    
    
    
    
    ##############################  [Model 2 / 2] Analyze LogisticRegression  ##############################
    Model random_state:  202
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    


    2023-01-27 00:50:55 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:05<00:00, 19.05it/s]

    
    


    
    2023-01-27 00:51:00 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:51:03 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics
    Analyze models in one run: 100%|[31m██████████[0m| 2/2 [00:27<00:00, 13.55s/it]

    
    
    
    


    



```python

```
