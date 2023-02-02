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
from source.metrics_computation_interfaces import run_metrics_computation_with_config, compute_model_metrics_with_config
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



## Get metrics for a base model with a compute_model_metrics_with_config interface and input arguments as a config


```python
model_name = 'DecisionTreeClassifier'
metrics_df = compute_model_metrics_with_config(models_config[model_name], model_name, dataset,
                                               config, SAVE_RESULTS_DIR_PATH,
                                               save_results=True,
                                               debug_mode=True)
print('Subgroups statistical bias and variance metrics: ')
metrics_df
```

    Model random_state:  623
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    Protected groups splits:
    sex_priv (214, 11)
    sex_dis (842, 11)
    race_priv (420, 11)
    race_dis (636, 11)
    sex&race_priv (93, 11)
    sex&race_dis (515, 11)
    
    
    Top rows of processed X train + validation set: 



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
      <th>age_cat_25 - 45_1</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>age_cat_Greater than 45_1</th>
      <th>age_cat_Greater than 45_0</th>
      <th>age_cat_Less than 25_0</th>
      <th>c_charge_degree_F_0</th>
      <th>c_charge_degree_F_1</th>
      <th>age_cat_25 - 45_0</th>
      <th>c_charge_degree_M_1</th>
      <th>c_charge_degree_M_0</th>
      <th>priors_count</th>
      <th>age_cat_Less than 25_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3600</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.274707</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3043</th>
      <td>-0.102581</td>
      <td>0</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-0.660459</td>
      <td>1</td>
    </tr>
    <tr>
      <th>418</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.467583</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3874</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.274707</td>
      <td>0</td>
    </tr>
    <tr>
      <th>442</th>
      <td>-0.102581</td>
      <td>0</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-0.660459</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4812</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.660459</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4487</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-1.57448</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.467583</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4968</th>
      <td>-0.102581</td>
      <td>0</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.660459</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4394</th>
      <td>-0.102581</td>
      <td>0</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.081831</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>-0.102581</td>
      <td>1</td>
      <td>-0.13003</td>
      <td>-0.149275</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.660459</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    


    2023-01-27 00:30:40 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:00<00:00, 172.04it/s]

    
    


    
    2023-01-27 00:30:41 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:30:45 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


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
      <td>0.667614</td>
      <td>0.696262</td>
      <td>0.660333</td>
      <td>0.647619</td>
      <td>0.680818</td>
      <td>0.655914</td>
      <td>0.669903</td>
      <td>623</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mean</td>
      <td>0.524041</td>
      <td>0.558159</td>
      <td>0.515370</td>
      <td>0.577132</td>
      <td>0.488981</td>
      <td>0.594027</td>
      <td>0.479205</td>
      <td>623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Std</td>
      <td>0.074339</td>
      <td>0.075245</td>
      <td>0.074109</td>
      <td>0.069531</td>
      <td>0.077515</td>
      <td>0.076792</td>
      <td>0.078327</td>
      <td>623</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IQR</td>
      <td>0.087800</td>
      <td>0.082206</td>
      <td>0.089221</td>
      <td>0.079059</td>
      <td>0.093572</td>
      <td>0.078600</td>
      <td>0.095591</td>
      <td>623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entropy</td>
      <td>0.192341</td>
      <td>0.180935</td>
      <td>0.195240</td>
      <td>0.161033</td>
      <td>0.213016</td>
      <td>0.171565</td>
      <td>0.218861</td>
      <td>623</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jitter</td>
      <td>0.120997</td>
      <td>0.111341</td>
      <td>0.123452</td>
      <td>0.101976</td>
      <td>0.133559</td>
      <td>0.105502</td>
      <td>0.137724</td>
      <td>623</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.656477</td>
      <td>0.695280</td>
      <td>0.646615</td>
      <td>0.645976</td>
      <td>0.663412</td>
      <td>0.653763</td>
      <td>0.648427</td>
      <td>623</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Label_Stability</td>
      <td>0.837727</td>
      <td>0.851682</td>
      <td>0.834181</td>
      <td>0.861095</td>
      <td>0.822296</td>
      <td>0.860645</td>
      <td>0.817010</td>
      <td>623</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TPR</td>
      <td>0.631356</td>
      <td>0.618421</td>
      <td>0.633838</td>
      <td>0.472050</td>
      <td>0.713826</td>
      <td>0.472222</td>
      <td>0.708487</td>
      <td>623</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TNR</td>
      <td>0.696918</td>
      <td>0.739130</td>
      <td>0.683857</td>
      <td>0.756757</td>
      <td>0.649231</td>
      <td>0.771930</td>
      <td>0.627049</td>
      <td>623</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PPV</td>
      <td>0.627368</td>
      <td>0.566265</td>
      <td>0.640306</td>
      <td>0.546763</td>
      <td>0.660714</td>
      <td>0.566667</td>
      <td>0.678445</td>
      <td>623</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FNR</td>
      <td>0.368644</td>
      <td>0.381579</td>
      <td>0.366162</td>
      <td>0.527950</td>
      <td>0.286174</td>
      <td>0.527778</td>
      <td>0.291513</td>
      <td>623</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FPR</td>
      <td>0.303082</td>
      <td>0.260870</td>
      <td>0.316143</td>
      <td>0.243243</td>
      <td>0.350769</td>
      <td>0.228070</td>
      <td>0.372951</td>
      <td>623</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Accuracy</td>
      <td>0.667614</td>
      <td>0.696262</td>
      <td>0.660333</td>
      <td>0.647619</td>
      <td>0.680818</td>
      <td>0.655914</td>
      <td>0.669903</td>
      <td>623</td>
    </tr>
    <tr>
      <th>14</th>
      <td>F1</td>
      <td>0.629356</td>
      <td>0.591195</td>
      <td>0.637056</td>
      <td>0.506667</td>
      <td>0.686244</td>
      <td>0.515152</td>
      <td>0.693141</td>
      <td>623</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Selection-Rate</td>
      <td>0.449811</td>
      <td>0.387850</td>
      <td>0.465558</td>
      <td>0.330952</td>
      <td>0.528302</td>
      <td>0.322581</td>
      <td>0.549515</td>
      <td>623</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Positive-Rate</td>
      <td>1.006356</td>
      <td>1.092105</td>
      <td>0.989899</td>
      <td>0.863354</td>
      <td>1.080386</td>
      <td>0.833333</td>
      <td>1.044280</td>
      <td>623</td>
    </tr>
  </tbody>
</table>
</div>



## Get metrics for a list of models with a run_metrics_computation_with_config interface and input arguments as a config


```python
models_metrics_dct = run_metrics_computation_with_config(dataset, config, models_config, SAVE_RESULTS_DIR_PATH, debug_mode=True)
```

    Analyze models in one run:   0%|[31m          [0m| 0/2 [00:00<?, ?it/s]

    ##############################  [Model 1 / 2] Analyze DecisionTreeClassifier  ##############################
    Model random_state:  491
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    Protected groups splits:
    sex_priv (194, 11)
    sex_dis (862, 11)
    race_priv (394, 11)
    race_dis (662, 11)
    sex&race_priv (98, 11)
    sex&race_dis (566, 11)
    
    
    Top rows of processed X train + validation set: 



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
      <th>age_cat_25 - 45_1</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>age_cat_Greater than 45_1</th>
      <th>age_cat_Greater than 45_0</th>
      <th>age_cat_Less than 25_0</th>
      <th>c_charge_degree_F_0</th>
      <th>c_charge_degree_F_1</th>
      <th>age_cat_25 - 45_0</th>
      <th>c_charge_degree_M_1</th>
      <th>c_charge_degree_M_0</th>
      <th>priors_count</th>
      <th>age_cat_Less than 25_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2244</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.083401</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2266</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.862647</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.651029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3323</th>
      <td>-0.093219</td>
      <td>0</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.651029</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1127</th>
      <td>-4.406383</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>3.785338</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-4.495524</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4793</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.862647</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3494</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-0.083401</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1289</th>
      <td>-0.250909</td>
      <td>1</td>
      <td>1.104331</td>
      <td>3.785338</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.085418</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>-0.093219</td>
      <td>1</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.295018</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1636</th>
      <td>-0.093219</td>
      <td>0</td>
      <td>-0.130521</td>
      <td>-0.154060</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.314656</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    


    2023-01-27 00:30:55 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:00<00:00, 159.14it/s]

    
    


    
    2023-01-27 00:30:55 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:31:00 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    [DecisionTreeClassifier] Metrics matrix:



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>General_Ensemble_Accuracy</td>
      <td>0.703598</td>
      <td>0.670103</td>
      <td>0.711137</td>
      <td>0.677665</td>
      <td>0.719033</td>
      <td>0.642857</td>
      <td>0.722615</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mean</td>
      <td>0.521465</td>
      <td>0.567385</td>
      <td>0.511131</td>
      <td>0.582361</td>
      <td>0.485222</td>
      <td>0.593591</td>
      <td>0.475824</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Std</td>
      <td>0.073755</td>
      <td>0.080002</td>
      <td>0.072348</td>
      <td>0.068526</td>
      <td>0.076867</td>
      <td>0.081239</td>
      <td>0.076549</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IQR</td>
      <td>0.080800</td>
      <td>0.088440</td>
      <td>0.079081</td>
      <td>0.076076</td>
      <td>0.083612</td>
      <td>0.083395</td>
      <td>0.081920</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entropy</td>
      <td>0.219793</td>
      <td>0.233656</td>
      <td>0.216673</td>
      <td>0.189168</td>
      <td>0.238021</td>
      <td>0.206719</td>
      <td>0.234097</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jitter</td>
      <td>0.139630</td>
      <td>0.148898</td>
      <td>0.137544</td>
      <td>0.120191</td>
      <td>0.151199</td>
      <td>0.129994</td>
      <td>0.148316</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.681420</td>
      <td>0.680670</td>
      <td>0.681589</td>
      <td>0.677589</td>
      <td>0.683701</td>
      <td>0.690612</td>
      <td>0.685936</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Label_Stability</td>
      <td>0.807689</td>
      <td>0.795155</td>
      <td>0.810510</td>
      <td>0.832640</td>
      <td>0.792840</td>
      <td>0.826939</td>
      <td>0.797951</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TPR</td>
      <td>0.679359</td>
      <td>0.523077</td>
      <td>0.702765</td>
      <td>0.523490</td>
      <td>0.745714</td>
      <td>0.433333</td>
      <td>0.761905</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TNR</td>
      <td>0.725314</td>
      <td>0.744186</td>
      <td>0.719626</td>
      <td>0.771429</td>
      <td>0.689103</td>
      <td>0.735294</td>
      <td>0.673307</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PPV</td>
      <td>0.689024</td>
      <td>0.507463</td>
      <td>0.717647</td>
      <td>0.582090</td>
      <td>0.729050</td>
      <td>0.419355</td>
      <td>0.745342</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FNR</td>
      <td>0.320641</td>
      <td>0.476923</td>
      <td>0.297235</td>
      <td>0.476510</td>
      <td>0.254286</td>
      <td>0.566667</td>
      <td>0.238095</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FPR</td>
      <td>0.274686</td>
      <td>0.255814</td>
      <td>0.280374</td>
      <td>0.228571</td>
      <td>0.310897</td>
      <td>0.264706</td>
      <td>0.326693</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Accuracy</td>
      <td>0.703598</td>
      <td>0.670103</td>
      <td>0.711137</td>
      <td>0.677665</td>
      <td>0.719033</td>
      <td>0.642857</td>
      <td>0.722615</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>14</th>
      <td>F1</td>
      <td>0.684157</td>
      <td>0.515152</td>
      <td>0.710128</td>
      <td>0.551237</td>
      <td>0.737288</td>
      <td>0.426230</td>
      <td>0.753532</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Selection-Rate</td>
      <td>0.465909</td>
      <td>0.345361</td>
      <td>0.493039</td>
      <td>0.340102</td>
      <td>0.540785</td>
      <td>0.316327</td>
      <td>0.568905</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Positive-Rate</td>
      <td>0.985972</td>
      <td>1.030769</td>
      <td>0.979263</td>
      <td>0.899329</td>
      <td>1.022857</td>
      <td>1.033333</td>
      <td>1.022222</td>
      <td>491</td>
      <td>DecisionTreeClassifier</td>
    </tr>
  </tbody>
</table>
</div>


    Analyze models in one run:  50%|[31m█████     [0m| 1/2 [00:15<00:15, 15.99s/it]

    
    
    
    
    ##############################  [Model 2 / 2] Analyze LogisticRegression  ##############################
    Model random_state:  492
    Baseline X_train shape:  (4222, 9)
    Baseline X_test shape:  (1056, 9)
    
    Protected groups splits:
    sex_priv (223, 11)
    sex_dis (833, 11)
    race_priv (402, 11)
    race_dis (654, 11)
    sex&race_priv (107, 11)
    sex&race_dis (538, 11)
    
    
    Top rows of processed X train + validation set: 



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
      <th>age_cat_25 - 45_1</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>age_cat_Greater than 45_1</th>
      <th>age_cat_Greater than 45_0</th>
      <th>age_cat_Less than 25_0</th>
      <th>c_charge_degree_F_0</th>
      <th>c_charge_degree_F_1</th>
      <th>age_cat_25 - 45_0</th>
      <th>c_charge_degree_M_1</th>
      <th>c_charge_degree_M_0</th>
      <th>priors_count</th>
      <th>age_cat_Less than 25_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>658</th>
      <td>-0.089367</td>
      <td>1</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.648738</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4149</th>
      <td>-0.089367</td>
      <td>0</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.270758</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>-0.089367</td>
      <td>1</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.459748</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1331</th>
      <td>-0.089367</td>
      <td>1</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.648738</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3069</th>
      <td>-0.089367</td>
      <td>1</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.270758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4670</th>
      <td>-0.089367</td>
      <td>1</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.270758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>616</th>
      <td>-0.089367</td>
      <td>0</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.270758</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4413</th>
      <td>-0.089367</td>
      <td>0</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.270758</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1244</th>
      <td>-0.089367</td>
      <td>0</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.606756</td>
      <td>0</td>
    </tr>
    <tr>
      <th>947</th>
      <td>-0.089367</td>
      <td>0</td>
      <td>-0.125836</td>
      <td>-0.146782</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.863181</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    


    2023-01-27 00:31:11 abstract_overall_variance_analyzer.py INFO    : Start classifiers testing by bootstrap
    
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 100/100 [00:05<00:00, 19.17it/s]

    
    


    
    2023-01-27 00:31:16 abstract_overall_variance_analyzer.py INFO    : Successfully tested classifiers by bootstrap
    2023-01-27 00:31:19 abstract_overall_variance_analyzer.py INFO    : Successfully computed predict proba metrics


    
    [LogisticRegression] Metrics matrix:



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>General_Ensemble_Accuracy</td>
      <td>0.679924</td>
      <td>0.659193</td>
      <td>0.685474</td>
      <td>0.671642</td>
      <td>0.685015</td>
      <td>0.672897</td>
      <td>0.693309</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mean</td>
      <td>0.520122</td>
      <td>0.565524</td>
      <td>0.507968</td>
      <td>0.581083</td>
      <td>0.482651</td>
      <td>0.606199</td>
      <td>0.472871</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Std</td>
      <td>0.021520</td>
      <td>0.019211</td>
      <td>0.022138</td>
      <td>0.019990</td>
      <td>0.022461</td>
      <td>0.019146</td>
      <td>0.023148</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IQR</td>
      <td>0.027995</td>
      <td>0.024942</td>
      <td>0.028812</td>
      <td>0.026028</td>
      <td>0.029203</td>
      <td>0.024786</td>
      <td>0.030091</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entropy</td>
      <td>0.079387</td>
      <td>0.000000</td>
      <td>0.080575</td>
      <td>0.077811</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jitter</td>
      <td>0.049364</td>
      <td>0.045588</td>
      <td>0.050375</td>
      <td>0.049021</td>
      <td>0.049576</td>
      <td>0.040521</td>
      <td>0.049428</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Per_Sample_Accuracy</td>
      <td>0.674536</td>
      <td>0.652108</td>
      <td>0.680540</td>
      <td>0.664229</td>
      <td>0.680872</td>
      <td>0.663364</td>
      <td>0.689312</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Label_Stability</td>
      <td>0.935625</td>
      <td>0.942422</td>
      <td>0.933806</td>
      <td>0.934527</td>
      <td>0.936300</td>
      <td>0.946168</td>
      <td>0.935725</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TPR</td>
      <td>0.632860</td>
      <td>0.455696</td>
      <td>0.666667</td>
      <td>0.440000</td>
      <td>0.717201</td>
      <td>0.323529</td>
      <td>0.741611</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TNR</td>
      <td>0.721137</td>
      <td>0.770833</td>
      <td>0.704057</td>
      <td>0.809524</td>
      <td>0.649518</td>
      <td>0.835616</td>
      <td>0.633333</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PPV</td>
      <td>0.665245</td>
      <td>0.521739</td>
      <td>0.690000</td>
      <td>0.578947</td>
      <td>0.692958</td>
      <td>0.478261</td>
      <td>0.715210</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FNR</td>
      <td>0.367140</td>
      <td>0.544304</td>
      <td>0.333333</td>
      <td>0.560000</td>
      <td>0.282799</td>
      <td>0.676471</td>
      <td>0.258389</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FPR</td>
      <td>0.278863</td>
      <td>0.229167</td>
      <td>0.295943</td>
      <td>0.190476</td>
      <td>0.350482</td>
      <td>0.164384</td>
      <td>0.366667</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Accuracy</td>
      <td>0.679924</td>
      <td>0.659193</td>
      <td>0.685474</td>
      <td>0.671642</td>
      <td>0.685015</td>
      <td>0.672897</td>
      <td>0.693309</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>14</th>
      <td>F1</td>
      <td>0.648649</td>
      <td>0.486486</td>
      <td>0.678133</td>
      <td>0.500000</td>
      <td>0.704871</td>
      <td>0.385965</td>
      <td>0.728171</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Selection-Rate</td>
      <td>0.444129</td>
      <td>0.309417</td>
      <td>0.480192</td>
      <td>0.283582</td>
      <td>0.542813</td>
      <td>0.214953</td>
      <td>0.574349</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Positive-Rate</td>
      <td>0.951318</td>
      <td>0.873418</td>
      <td>0.966184</td>
      <td>0.760000</td>
      <td>1.034985</td>
      <td>0.676471</td>
      <td>1.036913</td>
      <td>492</td>
      <td>LogisticRegression</td>
    </tr>
  </tbody>
</table>
</div>


    Analyze models in one run: 100%|[31m██████████[0m| 2/2 [00:34<00:00, 17.23s/it]

    
    
    
    


    



```python

```
