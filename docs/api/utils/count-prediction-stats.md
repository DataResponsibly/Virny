# count_prediction_stats

Compute means, stds, iqr, entropy, jitter, label stability, and transform predictions to pd.Dataframe.

Return a 1D numpy array of predictions, 2D array of each model prediction for y_test, a data structure of metrics.

## Parameters

- **y_test** (*pandas.core.frame.DataFrame*)

    True labels

- **uq_results**

    2D array of prediction labels by each model




