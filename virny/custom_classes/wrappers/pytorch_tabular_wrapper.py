import copy
import pandas as pd


class PytorchTabularWrapper:
    """
    A wrapper for Pytorch Tabular models. The wrapper aligns fit, predict, and predict_proba methods
     to be compatible with sklearn models.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def __copy__(self):
        new_estimator = copy.copy(self.estimator)
        return PytorchTabularWrapper(estimator=new_estimator)

    def __deepcopy__(self, memo):
        new_estimator = copy.deepcopy(self.estimator)
        return PytorchTabularWrapper(estimator=new_estimator)

    def get_params(self):
        return eval(str(self.estimator.config))

    def set_params(self, random_state: int):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, seed: int):
        self.estimator.fit(train=pd.concat([X, y], axis=1), seed=seed)
        return self

    def predict_proba(self, X, seed: int):
        return self.estimator.predict(X, tta_seed=seed).values

    def predict(self, X, seed: int):
        return self.estimator.predict(X, tta_seed=seed)
