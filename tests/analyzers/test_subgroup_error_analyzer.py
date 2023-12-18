import pandas as pd
from sklearn.metrics import confusion_matrix

from virny.configs.constants import *
from virny.analyzers.subgroup_error_analyzer import SubgroupErrorAnalyzer


def test_overall_accuracy_metrics_computation():
    y_test = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_preds = pd.Series([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])

    error_analyzer = SubgroupErrorAnalyzer(X_test=pd.DataFrame(),
                                           y_test=pd.DataFrame(),
                                           sensitive_attributes_dct=dict(),
                                           test_protected_groups=dict(),
                                           computation_mode=None)
    prediction_metrics = error_analyzer._compute_metrics(y_test, y_preds)

    # Check accuracy metrics
    TN, FP, FN, TP = confusion_matrix(y_test, y_preds).ravel()

    alpha = 0.000_001
    expected_TPR = TP/(TP+FN)
    expected_TNR = TN/(TN+FP)
    expected_PPV = TP/(TP+FP)
    expected_FNR = FN/(FN+TP)
    expected_FPR = FP/(FP+TN)
    expected_ACCURACY = (TP+TN)/(TP+TN+FP+FN)
    expected_F1 = (2*TP)/(2*TP+FP+FN)
    expected_SELECTION_RATE = (TP+FP)/(TP+FP+TN+FN)
    expected_POSITIVE_RATE = (TP+FP)/(TP+FN)

    assert abs(prediction_metrics[TPR] - expected_TPR) < alpha
    assert abs(prediction_metrics[TNR] - expected_TNR) < alpha
    assert abs(prediction_metrics[PPV] - expected_PPV) < alpha
    assert abs(prediction_metrics[FNR] - expected_FNR) < alpha
    assert abs(prediction_metrics[FPR] - expected_FPR) < alpha
    assert abs(prediction_metrics[ACCURACY] - expected_ACCURACY) < alpha
    assert abs(prediction_metrics[F1] - expected_F1) < alpha
    assert abs(prediction_metrics[SELECTION_RATE] - expected_SELECTION_RATE) < alpha
    assert abs(prediction_metrics[POSITIVE_RATE] - expected_POSITIVE_RATE) < alpha
