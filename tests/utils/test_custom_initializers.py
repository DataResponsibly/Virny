from tests import compas_without_sensitive_attrs_dataset_class, config_params, compas_dataset_class
from virny.utils.custom_initializers import create_base_pipeline


# ========================== Test create_base_pipeline ==========================
def test_create_base_pipeline_true1(compas_without_sensitive_attrs_dataset_class, config_params):
    base_pipeline = create_base_pipeline(compas_without_sensitive_attrs_dataset_class,
                                         config_params.sensitive_attributes_dct,
                                         model_seed=config_params.runs_seed_lst[0],
                                         test_set_fraction=config_params.test_set_fraction)

    assert base_pipeline.X_train_val.shape == (4222, 14)
    assert base_pipeline.X_test.shape == (1056, 14)
    assert base_pipeline.y_train_val.shape == (4222,)
    assert base_pipeline.y_test.shape == (1056,)

    assert len(base_pipeline.test_protected_groups) == len(config_params.sensitive_attributes_dct) * 2


def test_create_base_pipeline_true2(compas_dataset_class, config_params):
    new_sensitive_attributes_dct = {'sex': 0, 'race': 'Caucasian'}
    base_pipeline = create_base_pipeline(compas_dataset_class,
                                         new_sensitive_attributes_dct,
                                         model_seed=config_params.runs_seed_lst[0],
                                         test_set_fraction=config_params.test_set_fraction)

    assert base_pipeline.X_train_val.shape == (4222, 19)
    assert base_pipeline.X_test.shape == (1056, 19)
    assert base_pipeline.y_train_val.shape == (4222,)
    assert base_pipeline.y_test.shape == (1056,)

    assert len(base_pipeline.test_protected_groups) == len(new_sensitive_attributes_dct) * 2
