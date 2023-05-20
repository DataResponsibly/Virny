from virny.incremental_ml.river_utils import iter_pd_dataset


class IncrementalPandasDataset:
    """
    Generic data loader that converts a pandas df to a River dataset for incremental models
    """
    def __init__(self, pd_dataset, target, converters: dict):
        self.pd_dataset = pd_dataset
        self.target = target
        self.converters = converters

    def __iter__(self):
        return iter_pd_dataset(
            self.pd_dataset,
            target=self.target,
            converters=self.converters
        )
