# count_prediction_metrics

Compute means, stds, iqr, entropy, jitter, label stability, and transform predictions to pd.Dataframe.

Return a 1D numpy array of predictions, 2D array of each model prediction for y_test, a data structure of metrics.

## Parameters

- **y_true**

    True labels

- **uq_results**

    2D array of prediction proba for the zero value label by each model

- **with_predict_proba** (*bool*) – defaults to `True`

    [Optional] A flag if model can return probabilities for its predictions.  If no, only metrics based on labels (not labels and probabilities) will be computed.




