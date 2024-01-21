# jitter

Jitter is a stability metric that shows how the base model predictions fluctuate. Values closer to 0 -- perfect stability, values closer to 1 -- extremely bad stability.



## Parameters

- **y_true** (*pandas.core.frame.DataFrame*)

    A pandas dataframe of true labels. Is not used in this function, required for consistency.

- **uq_labels** (*pandas.core.frame.DataFrame*)

    `uq_labels` variable from count_prediction_metrics()




