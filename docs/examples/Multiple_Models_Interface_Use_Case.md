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


# Multiple Models Interface

In this example, we are going to conduct a deep performance profiling for 4 models. This demonstration will show how to create input arguments for Virny, how to compute overall and disparity metrics with a metric computation interface, and how to build static visualizations based on the calculated metrics. For that, we will use `compute_metrics_with_config` interface that can compute metrics for multiple models. Thus, we will need to do the next steps:

* Initialize input variables

* Compute subgroup metrics

* Perform disparity metrics composition using the Metric Composer

* Create static visualizations using the Metric Visualizer

## Import dependencies


```python
import os
import pandas as pd
from pprint import pprint
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
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.custom_classes.metrics_visualizer import MetricsVisualizer
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.utils.model_tuning_utils import tune_ML_models
from virny.datasets.base import BaseDataLoader
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


```python
models_params_for_tuning = {
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=MODELS_TUNING_SEED),
        'params': {
            "max_depth": [20, 30],
            "min_samples_split" : [0.1],
            "max_features": ['sqrt'],
            "criterion": ["gini", "entropy"]
        }
    },
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
    'XGBClassifier': {
        'model': XGBClassifier(random_state=MODELS_TUNING_SEED, verbosity=0),
        'params': {
            'learning_rate': [0.1],
            'n_estimators': [200],
            'max_depth': [5, 7],
            'lambda':  [10, 100]
        }
    }
}
```

### Create a config object

`compute_metrics_with_config` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: str, a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: float, the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: int, the number of estimators for bootstrap to compute subgroup stability metrics.

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
class CompasWithoutSensitiveAttrsDataset(BaseDataLoader):
    """
    Dataset class for COMPAS dataset that does not contain sensitive attributes among feature columns
     to test blind classifiers

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset

    """
    def __init__(self, dataset_path, subsample_size: int = None):
        df = pd.read_csv(dataset_path)
        if subsample_size:
            df = df.sample(subsample_size)

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

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
```


```python
data_loader = CompasWithoutSensitiveAttrsDataset(dataset_path=os.path.join('virny', 'datasets', 'data', 'COMPAS.csv'))
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

### Tune models and create a models config for metrics computation


```python
tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset, config.dataset_name, n_folds=3)
tuned_params_df
```

    2024/04/23, 12:53:34: Tuning DecisionTreeClassifier...
    2024/04/23, 12:53:35: Tuning for DecisionTreeClassifier is finished [F1 score = 0.6554846983071246, Accuracy = 0.6575048862828714]
    
    2024/04/23, 12:53:35: Tuning LogisticRegression...
    2024/04/23, 12:53:35: Tuning for LogisticRegression is finished [F1 score = 0.6483823116804865, Accuracy = 0.6520611566087312]
    
    2024/04/23, 12:53:35: Tuning RandomForestClassifier...
    2024/04/23, 12:53:36: Tuning for RandomForestClassifier is finished [F1 score = 0.6569271025126497, Accuracy = 0.6586904492688075]
    
    2024/04/23, 12:53:36: Tuning XGBClassifier...
    2024/04/23, 12:53:37: Tuning for XGBClassifier is finished [F1 score = 0.6623616224585352, Accuracy = 0.6646105242187331]





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
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>DecisionTreeClassifier</td>
      <td>0.655485</td>
      <td>0.657505</td>
      <td>{'criterion': 'gini', 'max_depth': 20, 'max_fe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>LogisticRegression</td>
      <td>0.648382</td>
      <td>0.652061</td>
      <td>{'C': 1, 'max_iter': 250, 'penalty': 'l2', 'so...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>RandomForestClassifier</td>
      <td>0.656927</td>
      <td>0.658690</td>
      <td>{'max_depth': 10, 'max_features': 0.6, 'min_sa...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>XGBClassifier</td>
      <td>0.662362</td>
      <td>0.664611</td>
      <td>{'lambda': 100, 'learning_rate': 0.1, 'max_dep...</td>
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

    {'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=20, max_features='sqrt', min_samples_split=0.1,
                           random_state=42),
     'LogisticRegression': LogisticRegression(C=1, max_iter=250, random_state=42, solver='newton-cg'),
     'RandomForestClassifier': RandomForestClassifier(max_depth=10, max_features=0.6, random_state=42),
     'XGBClassifier': XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric=None, feature_types=None,
                  gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
                  interaction_constraints=None, lambda=100, learning_rate=0.1,
                  max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
                  max_delta_step=None, max_depth=5, max_leaves=None,
                  min_child_weight=None, missing=nan, monotone_constraints=None,
                  n_estimators=200, n_jobs=None, num_parallel_tree=None,
                  predictor=None, ...)}


## Subgroup Metric Computation

After that we need to input the _BaseFlowDataset_ object, models config, and config yaml to a metric computation interface and execute it. The interface uses subgroup analyzers to compute different sets of metrics for each privileged and disadvantaged group. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metric computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
metrics_dct = compute_metrics_with_config(base_flow_dataset, config, models_config, SAVE_RESULTS_DIR_PATH,
                                          notebook_logs_stdout=True)
```


    Analyze multiple models:   0%|          | 0/4 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



    Classifiers testing by bootstrap:   0%|          | 0/50 [00:00<?, ?it/s]



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
      <th>sex_priv</th>
      <th>sex_dis</th>
      <th>race_priv</th>
      <th>race_dis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Overall_Uncertainty</td>
      <td>0.899836</td>
      <td>0.909407</td>
      <td>0.897446</td>
      <td>0.896719</td>
      <td>0.901847</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Statistical_Bias</td>
      <td>0.422194</td>
      <td>0.416842</td>
      <td>0.423530</td>
      <td>0.418523</td>
      <td>0.424561</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Std</td>
      <td>0.076228</td>
      <td>0.077296</td>
      <td>0.075962</td>
      <td>0.075141</td>
      <td>0.076929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mean_Prediction</td>
      <td>0.520117</td>
      <td>0.572049</td>
      <td>0.507149</td>
      <td>0.581026</td>
      <td>0.480839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aleatoric_Uncertainty</td>
      <td>0.869944</td>
      <td>0.875791</td>
      <td>0.868484</td>
      <td>0.866015</td>
      <td>0.872477</td>
    </tr>
    <tr>
      <th>5</th>
      <td>IQR</td>
      <td>0.093218</td>
      <td>0.092883</td>
      <td>0.093302</td>
      <td>0.095182</td>
      <td>0.091952</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Epistemic_Uncertainty</td>
      <td>0.029893</td>
      <td>0.033616</td>
      <td>0.028963</td>
      <td>0.030704</td>
      <td>0.029369</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jitter</td>
      <td>0.148098</td>
      <td>0.159899</td>
      <td>0.145152</td>
      <td>0.138860</td>
      <td>0.154056</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Label_Stability</td>
      <td>0.786591</td>
      <td>0.766825</td>
      <td>0.791527</td>
      <td>0.801256</td>
      <td>0.777134</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TPR</td>
      <td>0.687898</td>
      <td>0.573333</td>
      <td>0.709596</td>
      <td>0.578231</td>
      <td>0.737654</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TNR</td>
      <td>0.687179</td>
      <td>0.808824</td>
      <td>0.650334</td>
      <td>0.756554</td>
      <td>0.628931</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PPV</td>
      <td>0.639053</td>
      <td>0.623188</td>
      <td>0.641553</td>
      <td>0.566667</td>
      <td>0.669468</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FNR</td>
      <td>0.312102</td>
      <td>0.426667</td>
      <td>0.290404</td>
      <td>0.421769</td>
      <td>0.262346</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FPR</td>
      <td>0.312821</td>
      <td>0.191176</td>
      <td>0.349666</td>
      <td>0.243446</td>
      <td>0.371069</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Accuracy</td>
      <td>0.687500</td>
      <td>0.725118</td>
      <td>0.678107</td>
      <td>0.693237</td>
      <td>0.683801</td>
    </tr>
    <tr>
      <th>15</th>
      <td>F1</td>
      <td>0.662577</td>
      <td>0.597222</td>
      <td>0.673861</td>
      <td>0.572391</td>
      <td>0.701909</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Selection-Rate</td>
      <td>0.480114</td>
      <td>0.327014</td>
      <td>0.518343</td>
      <td>0.362319</td>
      <td>0.556075</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sample_Size</td>
      <td>1056.000000</td>
      <td>211.000000</td>
      <td>845.000000</td>
      <td>414.000000</td>
      <td>642.000000</td>
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
      <th>sex</th>
      <th>race</th>
      <th>sex&amp;race</th>
      <th>Model_Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy_Difference</td>
      <td>-0.047012</td>
      <td>-0.009436</td>
      <td>-0.039300</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aleatoric_Uncertainty_Difference</td>
      <td>-0.007307</td>
      <td>0.006463</td>
      <td>0.000802</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aleatoric_Uncertainty_Ratio</td>
      <td>0.991656</td>
      <td>1.007463</td>
      <td>1.000922</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Epistemic_Uncertainty_Difference</td>
      <td>-0.004654</td>
      <td>-0.001335</td>
      <td>-0.003381</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Epistemic_Uncertainty_Ratio</td>
      <td>0.861563</td>
      <td>0.956510</td>
      <td>0.892966</td>
      <td>DecisionTreeClassifier</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Disparate_Impact</td>
      <td>1.465176</td>
      <td>1.537383</td>
      <td>1.596796</td>
      <td>XGBClassifier</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Std_Difference</td>
      <td>0.000151</td>
      <td>0.002984</td>
      <td>0.002995</td>
      <td>XGBClassifier</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Std_Ratio</td>
      <td>1.003178</td>
      <td>1.065098</td>
      <td>1.064903</td>
      <td>XGBClassifier</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Equalized_Odds_TNR</td>
      <td>-0.076968</td>
      <td>-0.101583</td>
      <td>-0.123015</td>
      <td>XGBClassifier</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Equalized_Odds_TPR</td>
      <td>0.153535</td>
      <td>0.152053</td>
      <td>0.155233</td>
      <td>XGBClassifier</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 5 columns</p>
</div>



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





<div id="altair-viz-c855d7edc99943b590f9162570867850"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-c855d7edc99943b590f9162570867850") {
      outputDiv = document.getElementById("altair-viz-c855d7edc99943b590f9162570867850");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-d6613044ce497fb40b7284d86df2c2d6"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Accuracy", "F1", "TPR", "TNR", "PPV", "Selection-Rate"], "title": "Accuracy Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-d6613044ce497fb40b7284d86df2c2d6": [{"overall": 0.6875, "metric": "Accuracy", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6625766871165644, "metric": "F1", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6390532544378699, "metric": "PPV", "model_name": "DecisionTreeClassifier"}, {"overall": 0.4801136363636363, "metric": "Selection-Rate", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6871794871794872, "metric": "TNR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6878980891719745, "metric": "TPR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.6846590909090909, "metric": "Accuracy", "model_name": "LogisticRegression"}, {"overall": 0.6392199349945829, "metric": "F1", "model_name": "LogisticRegression"}, {"overall": 0.6526548672566371, "metric": "PPV", "model_name": "LogisticRegression"}, {"overall": 0.428030303030303, "metric": "Selection-Rate", "model_name": "LogisticRegression"}, {"overall": 0.7316239316239316, "metric": "TNR", "model_name": "LogisticRegression"}, {"overall": 0.6263269639065817, "metric": "TPR", "model_name": "LogisticRegression"}, {"overall": 0.7102272727272727, "metric": "Accuracy", "model_name": "RandomForestClassifier"}, {"overall": 0.6758474576271186, "metric": "F1", "model_name": "RandomForestClassifier"}, {"overall": 0.6744186046511628, "metric": "PPV", "model_name": "RandomForestClassifier"}, {"overall": 0.4479166666666667, "metric": "Selection-Rate", "model_name": "RandomForestClassifier"}, {"overall": 0.7367521367521368, "metric": "TNR", "model_name": "RandomForestClassifier"}, {"overall": 0.6772823779193206, "metric": "TPR", "model_name": "RandomForestClassifier"}, {"overall": 0.7026515151515151, "metric": "Accuracy", "model_name": "XGBClassifier"}, {"overall": 0.6652452025586354, "metric": "F1", "model_name": "XGBClassifier"}, {"overall": 0.6680942184154176, "metric": "PPV", "model_name": "XGBClassifier"}, {"overall": 0.4422348484848485, "metric": "Selection-Rate", "model_name": "XGBClassifier"}, {"overall": 0.7350427350427351, "metric": "TNR", "model_name": "XGBClassifier"}, {"overall": 0.6624203821656051, "metric": "TPR", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metrics_bar_char(
    metric_names=['Aleatoric_Uncertainty', 'Overall_Uncertainty', 'Label_Stability', 'Std', 'IQR', 'Jitter'],
    plot_title="Stability and Uncertainty Metrics"
)
```





<div id="altair-viz-8c19e3df855e4c44bbb52a3259fcef63"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-8c19e3df855e4c44bbb52a3259fcef63") {
      outputDiv = document.getElementById("altair-viz-8c19e3df855e4c44bbb52a3259fcef63");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 16, "titleFontSize": 20}, "headerRow": {"labelAlign": "left", "labelAngle": 0, "labelFontSize": 16, "labelPadding": 10, "titleFontSize": 20}}, "data": {"name": "data-b4f0885c17c2f46f31d0b1f500e10aee"}, "mark": "bar", "encoding": {"color": {"field": "model_name", "legend": {"labelFontSize": 15, "labelLimit": 300, "title": "Model Name", "titleFontSize": 15, "titleLimit": 300}, "scale": {"scheme": "tableau20"}, "type": "nominal"}, "row": {"field": "metric", "sort": ["Aleatoric_Uncertainty", "Overall_Uncertainty", "Label_Stability", "Std", "IQR", "Jitter"], "title": "Stability and Uncertainty Metrics", "type": "nominal"}, "x": {"axis": {"grid": true}, "field": "overall", "title": "", "type": "quantitative"}, "y": {"axis": null, "field": "model_name", "type": "nominal"}}, "height": 50, "width": 500, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-b4f0885c17c2f46f31d0b1f500e10aee": [{"overall": 0.8699437461655267, "metric": "Aleatoric_Uncertainty", "model_name": "DecisionTreeClassifier"}, {"overall": 0.0932183704869789, "metric": "IQR", "model_name": "DecisionTreeClassifier"}, {"overall": 0.1480983302411873, "metric": "Jitter", "model_name": "DecisionTreeClassifier"}, {"overall": 0.7865909090909091, "metric": "Label_Stability", "model_name": "DecisionTreeClassifier"}, {"overall": 0.8998362650313954, "metric": "Overall_Uncertainty", "model_name": "DecisionTreeClassifier"}, {"overall": 0.0762281206671186, "metric": "Std", "model_name": "DecisionTreeClassifier"}, {"overall": 0.9086231626304788, "metric": "Aleatoric_Uncertainty", "model_name": "LogisticRegression"}, {"overall": 0.0275305517600382, "metric": "IQR", "model_name": "LogisticRegression"}, {"overall": 0.0415530303030303, "metric": "Jitter", "model_name": "LogisticRegression"}, {"overall": 0.9435984848484849, "metric": "Label_Stability", "model_name": "LogisticRegression"}, {"overall": 0.9113463914637512, "metric": "Overall_Uncertainty", "model_name": "LogisticRegression"}, {"overall": 0.0220015222316574, "metric": "Std", "model_name": "LogisticRegression"}, {"overall": 0.8369293613974009, "metric": "Aleatoric_Uncertainty", "model_name": "RandomForestClassifier"}, {"overall": 0.0912416222914894, "metric": "IQR", "model_name": "RandomForestClassifier"}, {"overall": 0.1124111008039578, "metric": "Jitter", "model_name": "RandomForestClassifier"}, {"overall": 0.8398106060606061, "metric": "Label_Stability", "model_name": "RandomForestClassifier"}, {"overall": 0.8621636502678327, "metric": "Overall_Uncertainty", "model_name": "RandomForestClassifier"}, {"overall": 0.0698687436268046, "metric": "Std", "model_name": "RandomForestClassifier"}, {"overall": 0.8719396591186523, "metric": "Aleatoric_Uncertainty", "model_name": "XGBClassifier"}, {"overall": 0.063116195258882, "metric": "IQR", "model_name": "XGBClassifier"}, {"overall": 0.0683766233766233, "metric": "Jitter", "model_name": "XGBClassifier"}, {"overall": 0.9115909090909092, "metric": "Label_Stability", "model_name": "XGBClassifier"}, {"overall": 0.8810163683831591, "metric": "Overall_Uncertainty", "model_name": "XGBClassifier"}, {"overall": 0.047656238079071, "metric": "Std", "model_name": "XGBClassifier"}]}}, {"mode": "vega-lite"});
</script>




```python
visualizer.create_overall_metric_heatmap(
    model_names=list(models_params_for_tuning.keys()),
    metrics_lst=visualizer.all_accuracy_metrics + visualizer.all_uncertainty_metrics,
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_Use_Case_files/Multiple_Models_Interface_Use_Case_43_0.png)
    



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
        'IQR_Difference',
        'Std_Ratio',
    ],
    groups_lst=config.sensitive_attributes_dct.keys(),
    tolerance=0.005,
)
```


    
![png](Multiple_Models_Interface_Use_Case_files/Multiple_Models_Interface_Use_Case_44_0.png)
    



```python

```
