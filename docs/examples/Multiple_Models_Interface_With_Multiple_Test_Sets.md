# Multiple Models Interface For Multiple Test Sets

In this example, we are going to conduct a deep performance profiling for 2 models. For that, we will use `compute_metrics_with_multiple_test_sets` interface that will run metric computation for multiple models and test each model using multiple test sets.

## Import dependencies


```python
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from virny.user_interfaces.multiple_models_with_multiple_test_sets_api import compute_metrics_with_multiple_test_sets
from virny.utils.custom_initializers import create_config_obj, create_models_metrics_dct_from_database_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.datasets import CompasWithoutSensitiveAttrsDataset
```


## Initialize Input Variables

Based on the library flow, we need to create 3 input objects for a user interface:

* A **config yaml** that is a file with configuration parameters for different user interfaces for metric computation.

* A **dataset class** that is a wrapper above the user’s raw dataset that includes its descriptive attributes like a target column, numerical columns, categorical columns, etc. This class must be inherited from the BaseDataset class, which was created for user convenience.

* Finally, a **models config** that is a Python dictionary, where keys are model names and values are initialized models for analysis. This dictionary helps conduct audits for different analysis modes and analyze different types of models.


```python
TEST_SET_FRACTION = 0.2
DATASET_SPLIT_SEED = 42
```

### Create a config object

`compute_metrics_with_multiple_test_sets` interface requires that your **yaml file** includes the following parameters:

* **dataset_name**: str, a name of your dataset; it will be used to name files with metrics.

* **bootstrap_fraction**: float, the fraction from a train set in the range [0.0 - 1.0] to fit models in bootstrap (usually more than 0.5).

* **n_estimators**: int, the number of estimators for bootstrap to compute subgroup stability metrics.

* **sensitive_attributes_dct**: dict, a dictionary where keys are sensitive attribute names (including intersectional attributes), and values are disadvantaged values for these attributes. Intersectional attributes must include '&' between sensitive attributes. You do not need to specify disadvantaged values for intersectional groups since they will be derived from disadvantaged values in sensitive_attributes_dct for each separate sensitive attribute in this intersectional pair.

Note that disadvantaged value in a sensitive attribute dictionary must be **the same as in the original dataset**. For example, when distinct values of the _sex_ column in the original dataset are 'F' and 'M', and after pre-processing they became 0 and 1 respectively, you still need to set a disadvantaged value as 'F' or 'M' in the sensitive attribute dictionary.



```python
ROOT_DIR = os.getcwd()
config_yaml_path = os.path.join(ROOT_DIR, 'experiment_config.yaml')
config_yaml_content = \
"""dataset_name: COMPAS_Without_Sensitive_Attributes
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
```

### Create a Dataset class

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

### Create a models config

**models_config** is a Python dictionary, where keys are model names and values are initialized models for analysis


```python
models_config = {
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='gini',
                                                     max_depth=20,
                                                     max_features=0.6,
                                                     min_samples_split=0.1),
    'RandomForestClassifier': RandomForestClassifier(max_depth=4,
                                                     max_features=0.6,
                                                     min_samples_leaf=1,
                                                     n_estimators=50),
}
```

## Subgroup Metric Computation

After that we need to input the _BaseFlowDataset_ object, models config, and config yaml to a metric computation interface and execute it. The interface uses subgroup analyzers to compute different sets of metrics for each privileged and disadvantaged group. As for now, our library supports **Subgroup Variance Analyzer** and **Subgroup Error Analyzer**, but it is easily extensible to any other analyzers. When the variance and error analyzers complete metric computation, their metrics are combined, returned in a matrix format, and stored in a file if defined.


```python
import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv(os.path.join(ROOT_DIR, 'secrets.env'))  # Take environment variables from .env

# Provide the mongodb atlas url to connect python to mongodb using pymongo
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
# Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
client = MongoClient(CONNECTION_STRING)
collection = client[os.getenv("DB_NAME")]['preprocessing_results']


def db_writer_func(run_models_metrics_df, collection=collection):
    run_models_metrics_df.columns = run_models_metrics_df.columns.str.lower()  # Rename Pandas columns to lower case
    collection.insert_many(run_models_metrics_df.to_dict('records'))
```


```python
import uuid

custom_table_fields_dct = {
    'session_uuid': str(uuid.uuid4()),
    'preprocessing_techniques': 'one hot encoder and scaler',
}
print('Current session uuid: ', custom_table_fields_dct['session_uuid'])
```

    Current session uuid:  8d31eaab-5d6d-4830-9b23-c29355efa90b



```python
extra_test_sets_lst = [(base_flow_dataset.X_test, base_flow_dataset.y_test, base_flow_dataset.init_sensitive_attrs_df)]
compute_metrics_with_multiple_test_sets(dataset=base_flow_dataset,
                                        extra_test_sets_lst=extra_test_sets_lst,
                                        config=config,
                                        models_config=models_config,
                                        custom_tbl_fields_dct=custom_table_fields_dct,
                                        db_writer_func=db_writer_func)
```

    Analyze multiple models:   0%|[31m          [0m| 0/2 [00:00<?, ?it/s]
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 50/50 [00:00<00:00, 112.87it/s]
    Analyze multiple models:  50%|[31m█████     [0m| 1/2 [00:06<00:06,  6.70s/it]
    Classifiers testing by bootstrap: 100%|[34m██████████[0m| 50/50 [00:03<00:00, 16.63it/s]
    Analyze multiple models: 100%|[31m██████████[0m| 2/2 [00:16<00:00,  8.05s/it]



```python
def read_model_metric_dfs_from_db(collection, session_uuid):
    cursor = collection.find({'session_uuid': session_uuid})
    records = []
    for record in cursor:
        del record['_id']
        records.append(record)

    model_metric_dfs = pd.DataFrame(records)

    # Capitalize column names to be consistent across the whole library
    new_column_names = []
    for col in model_metric_dfs.columns:
        new_col_name = '_'.join([c.capitalize() for c in col.split('_')])
        new_column_names.append(new_col_name)

    model_metric_dfs.columns = new_column_names
    return model_metric_dfs
```


```python
model_metric_dfs = read_model_metric_dfs_from_db(collection, custom_table_fields_dct['session_uuid'])
models_metrics_dct = create_models_metrics_dct_from_database_df(model_metric_dfs)
```


```python
models_metrics_dct['RandomForestClassifier'].head(20)
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
      <th>Model_Name</th>
      <th>Model_Params</th>
      <th>Dataset_Name</th>
      <th>Num_Estimators</th>
      <th>Test_Set_Index</th>
      <th>Tag</th>
      <th>Record_Create_Date_Time</th>
      <th>Session_Uuid</th>
      <th>Preprocessing_Techniques</th>
      <th>...</th>
      <th>sex&amp;race_dis_incorrect</th>
      <th>sex&amp;race_priv</th>
      <th>sex&amp;race_priv_correct</th>
      <th>sex&amp;race_priv_incorrect</th>
      <th>sex_dis</th>
      <th>sex_dis_correct</th>
      <th>sex_dis_incorrect</th>
      <th>sex_priv</th>
      <th>sex_priv_correct</th>
      <th>sex_priv_incorrect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Accuracy</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.701521</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.704142</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.677725</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Accuracy</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.701521</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.704142</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.677725</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Aleatoric_Uncertainty</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.927344</td>
      <td>0.907714</td>
      <td>0.897444</td>
      <td>0.931851</td>
      <td>0.905728</td>
      <td>0.897731</td>
      <td>0.924762</td>
      <td>0.914713</td>
      <td>0.899245</td>
      <td>0.947244</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Aleatoric_Uncertainty</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.927344</td>
      <td>0.907714</td>
      <td>0.897444</td>
      <td>0.931851</td>
      <td>0.905728</td>
      <td>0.897731</td>
      <td>0.924762</td>
      <td>0.914713</td>
      <td>0.899245</td>
      <td>0.947244</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Epistemic_Uncertainty</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.005648</td>
      <td>0.005803</td>
      <td>0.005455</td>
      <td>0.006622</td>
      <td>0.006179</td>
      <td>0.006217</td>
      <td>0.006088</td>
      <td>0.005261</td>
      <td>0.004777</td>
      <td>0.006279</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Epistemic_Uncertainty</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.005648</td>
      <td>0.005803</td>
      <td>0.005455</td>
      <td>0.006622</td>
      <td>0.006179</td>
      <td>0.006217</td>
      <td>0.006088</td>
      <td>0.005261</td>
      <td>0.004777</td>
      <td>0.006279</td>
    </tr>
    <tr>
      <th>14</th>
      <td>F1</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.560224</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.685930</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.492537</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>F1</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.560224</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.685930</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.492537</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FNR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.468085</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.310606</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.560000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>FNR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.468085</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.310606</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.560000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>FPR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.204142</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.282851</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.191176</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>FPR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.204142</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.282851</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.191176</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>IQR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.054553</td>
      <td>0.052841</td>
      <td>0.050180</td>
      <td>0.059094</td>
      <td>0.055071</td>
      <td>0.054580</td>
      <td>0.056240</td>
      <td>0.050716</td>
      <td>0.046855</td>
      <td>0.058835</td>
    </tr>
    <tr>
      <th>27</th>
      <td>IQR</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.054553</td>
      <td>0.052841</td>
      <td>0.050180</td>
      <td>0.059094</td>
      <td>0.055071</td>
      <td>0.054580</td>
      <td>0.056240</td>
      <td>0.050716</td>
      <td>0.046855</td>
      <td>0.058835</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Jitter</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.096073</td>
      <td>0.079947</td>
      <td>0.063406</td>
      <td>0.118825</td>
      <td>0.076779</td>
      <td>0.068873</td>
      <td>0.095595</td>
      <td>0.095305</td>
      <td>0.069125</td>
      <td>0.150360</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Jitter</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.096073</td>
      <td>0.079947</td>
      <td>0.063406</td>
      <td>0.118825</td>
      <td>0.076779</td>
      <td>0.068873</td>
      <td>0.095595</td>
      <td>0.095305</td>
      <td>0.069125</td>
      <td>0.150360</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Label_Stability</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.868323</td>
      <td>0.893536</td>
      <td>0.917724</td>
      <td>0.836688</td>
      <td>0.896710</td>
      <td>0.908706</td>
      <td>0.868160</td>
      <td>0.872986</td>
      <td>0.909650</td>
      <td>0.795882</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Label_Stability</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.868323</td>
      <td>0.893536</td>
      <td>0.917724</td>
      <td>0.836688</td>
      <td>0.896710</td>
      <td>0.908706</td>
      <td>0.868160</td>
      <td>0.872986</td>
      <td>0.909650</td>
      <td>0.795882</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Mean_Prediction</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>0</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.724</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.494237</td>
      <td>0.573680</td>
      <td>0.590608</td>
      <td>0.533895</td>
      <td>0.510769</td>
      <td>0.512022</td>
      <td>0.507785</td>
      <td>0.575480</td>
      <td>0.594258</td>
      <td>0.535992</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Mean_Prediction</td>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>COMPAS_Without_Sensitive_Attributes</td>
      <td>50</td>
      <td>1</td>
      <td>OK</td>
      <td>2024-01-29 12:53:00.728</td>
      <td>8d31eaab-5d6d-4830-9b23-c29355efa90b</td>
      <td>one hot encoder and scaler</td>
      <td>...</td>
      <td>0.494237</td>
      <td>0.573680</td>
      <td>0.590608</td>
      <td>0.533895</td>
      <td>0.510769</td>
      <td>0.512022</td>
      <td>0.507785</td>
      <td>0.575480</td>
      <td>0.594258</td>
      <td>0.535992</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 29 columns</p>
</div>




```python
client.close()
```
