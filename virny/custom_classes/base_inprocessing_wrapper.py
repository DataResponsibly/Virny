import abc


class BaseInprocessingWrapper(metaclass=abc.ABCMeta):
    """
    A wrapper for fair inprocessors from aif360. The wrapper aligns fit, predict, and predict_proba methods
    to be compatible with sklearn models.
    """
    @abc.abstractmethod
    def fit(self, X, y):
        """Fits an inprocessor based on X and y pandas dataframes. Returns self."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, X):
        """
        Predicts using the fitted inprocessor based on X features pandas dataframe. Returns probabilities
        for **ZERO** class. These probabilities will be used by Virny in the downstream metric computation.

        If the inprocessor does not support prediction probabilities, the implementation of this method can be skipped,
         and with_predict_proba = False should be set in the metric computation interface.

        Indexes from the returned array of predictions should start from zero.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        """
        Predicts using the fitted inprocessor based on X features pandas dataframe. Returns labels for each
        sample. Indexes from the returned array of predictions should start from zero.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __deepcopy__(self, memo):
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self):
        """Returns parameters of the wrapper. Alignment with sklearn API."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_params(self, random_state: int):
        """Set a random state of the inprocessor."""
        raise NotImplementedError
