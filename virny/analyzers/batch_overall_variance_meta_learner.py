import gc

import numpy as np
import pandas as pd

from virny.utils.stability_utils import count_prediction_metrics
from virny.analyzers.batch_overall_variance_analyzer import BatchOverallVarianceAnalyzer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from catboost import CatBoostClassifier


class BatchOverallVarianceMetaLearner(BatchOverallVarianceAnalyzer):
    """
    Analyzer to compute subgroup variance metrics based on the meta-learner ensemble approach [^1].

    Parameters
    ----------
    base_model
        Base model for stability measuring
    base_model_name
        Model name like 'DecisionTreeClassifier' or 'LogisticRegression'
    X_train
        Processed features train set
    y_train
        Targets train set
    X_test
        Processed features test set
    y_test
        Targets test set
    target_column
        Name of the target column
    dataset_name
        Name of dataset, used for correct results naming
    n_estimators
        Number of estimators in ensemble to measure base_model stability
    with_predict_proba
        [Optional] A flag if model can return probabilities for its predictions.
         If no, only metrics based on labels (not labels and probabilities) will be computed.
    notebook_logs_stdout
        [Optional] True, if this interface was execute in a Jupyter notebook,
         False, otherwise.
    verbose
        [Optional] Level of logs printing. The greater level provides more logs.
         As for now, 0, 1, 2 levels are supported.

    References
    ----------
    [^1]: Lahoti, P., Gummadi, K. and Weikum, G., 2023.
    Responsible model deployment via model-agnostic uncertainty learning.
    Machine Learning, 112(3), pp.939-970.

    """
    def __init__(self, meta_learner_config: dict, base_model, base_model_name: str,
                 X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                 target_column: str, dataset_name: str, n_estimators: int, with_predict_proba: bool = True,
                 notebook_logs_stdout: bool = False, verbose: int = 0):
        # TODO: update documentation for meta_learner_config, save_results
        super().__init__(base_model=base_model,
                         base_model_name=base_model_name,
                         bootstrap_fraction=None,
                         X_train=X_train,
                         y_train=y_train,
                         X_test=X_test,
                         y_test=y_test,
                         target_column=target_column,
                         dataset_name=dataset_name,
                         n_estimators=n_estimators,
                         with_predict_proba=with_predict_proba,
                         notebook_logs_stdout=notebook_logs_stdout,
                         verbose=verbose)

        # TODO: validate meta_learner_config
        self.meta_learner_config = meta_learner_config
        self.y_pred_test = None  # Placeholders for based model predictions on a test set
        self.error_test = None  # Placeholders for based model errors on a test set

    def compute_metrics(self, save_results: bool = True, with_fit: bool = True):
        """
        Measure metrics for the base model. Save results to a .csv file.

        Parameters
        ----------
        save_results
            If to save result metrics in a file
        with_fit
            If to fit estimators in bootstrap

        """
        # Quantify uncertainty for the base model.
        # Arguments of the UQ_by_boostrap method are not used
        # but aligned with a UQ_by_boostrap in AbstractOverallVarianceAnalyzer.
        self.models_predictions = self.UQ_by_boostrap(boostrap_size=None, with_replacement=None, with_fit=with_fit)

        # Count metrics based on prediction proba results
        # TODO: can we approximate MEAN_PREDICTION and STATISTICAL_BIAS using the meat-learner?
        _, self.prediction_metrics = count_prediction_metrics(self.error_test.values, self.models_predictions, self.with_predict_proba)
        self._logger.info(f'Successfully computed predict proba metrics')

        if save_results:
            self.save_metrics_to_file()
        else:
            return self.y_pred_test, self.y_test

    def UQ_by_boostrap(self, boostrap_size: int, with_replacement: bool, with_fit: bool = True) -> dict:
        """
        Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples
        and applying postprocessing intervention.

        Return a dictionary where keys are models indexes, and values are lists of
         correspondent model predictions for X_test set.

        Parameters
        ----------
        boostrap_size
            Number of records in bootstrap splits
        with_replacement
            Enable replacement or not
        with_fit
            Whether to fit estimators in bootstrap

        """
        if self._verbose >= 1:
            print('\n', flush=True)
        self._logger.info('Start uncertainty quantification using a meta-learner ensemble')

        # Step 1: Fit bbox on the original train set
        bbox_clf = self._fit_model(self.base_model, self.X_train, self.y_train)
        self._logger.info('Black box model is fitted')

        # Step 2: Get prediction labels and probabilities for train and test sets using the fitted bbox
        y_pred_train = self._batch_predict(bbox_clf, self.X_train)
        y_pred_test = self._batch_predict(bbox_clf, self.X_test)
        y_pred_prob_train = self._batch_predict_proba_all(bbox_clf, self.X_train)
        y_pred_prob_test = self._batch_predict_proba_all(bbox_clf, self.X_test)

        # Step 3: Create error train and test sets
        error_train = (y_pred_train != self.y_train).astype(int)
        error_test = (y_pred_test != self.y_test).astype(int)

        # Store y_pred_test to compute model correctness. And store error_test to compute model arbitrariness.
        self.y_pred_test = pd.Series(y_pred_test, index=self.y_test.index)
        self.error_test = pd.Series(error_test, index=self.y_test.index)

        print('self.y_pred_test.index[:10] -- ', self.y_pred_test.index[:10])
        print('self.error_test.index[:10] -- ', self.error_test.index[:10])

        # Step 4: Tune and fit a meta-learner based on the concatenated X_train and y_pred_prob_train sets
        X_train_meta_learner = np.concatenate([self.X_train.values, y_pred_prob_train], axis=1)
        X_test_meta_learner = np.concatenate([self.X_test.values, y_pred_prob_test], axis=1)

        meta_learner_clf = BatchOverallVarianceMetaLearner._build_ensemble(meta_learner_config=self.meta_learner_config,
                                                                           n_estimators=self.n_estimators)
        meta_learner_clf.fit(X_train_meta_learner, error_train)  # Tune and fit a meta-learner
        self._logger.info('Meta-learner ensemble is tuned and fitted')
        print('\n\nBest params for estimators in the meta-learner ensemble')
        meta_learner_best_params = dict()
        for name, est in meta_learner_clf.named_estimators_.items():
            meta_learner_best_params[name] = est.best_params_
            print(f'{name}: {meta_learner_best_params[name]}', flush=True)

        # Step 5: Get prediction labels and probabilities for a test set using the tuned meta-learner
        models_predictions = {idx: [] for idx in range(self.n_estimators)}

        # Fetch pred_prob P(z|h,x) of each SGBT in the E-SGBT (auditor ensemble model)
        self.models_lst = meta_learner_clf.estimators_
        for idx, auditor_estimator in enumerate(self.models_lst):
            models_predictions[idx] = auditor_estimator.predict_proba(X_test_meta_learner)[:, 0]

        gc.collect()  # Enforce garbage collection after the most intensive part of the pipeline

        if self._verbose >= 1:
            print('\n', flush=True)
        self._logger.info('Successfully applied a meta-learner ensemble on the test set')

        return models_predictions

    @staticmethod
    def _build_ensemble(meta_learner_config, n_estimators, cv=3, n_jobs=-1, scoring=None):
        meta_learner_name = meta_learner_config['model']
        meta_learner_params = meta_learner_config['params']

        estimators = []
        for estimator_num in range(1, n_estimators + 1):
            # Each estimator in the ensemble is initialized with a different seed
            # to have minimum correlation between the estimators in the ensemble
            if meta_learner_name.lower() == 'gbt':
                estimator = GradientBoostingClassifier(random_state=estimator_num)
                estimator.classes_ = np.array([0, 1])
            elif meta_learner_name.lower() == 'cbt':
                estimator = CatBoostClassifier(loss_function='Logloss',
                                               verbose=True,
                                               boosting_type='Plain',
                                               bootstrap_type='Bernoulli',
                                               posterior_sampling=True,
                                               used_ram_limit='20gb',
                                               random_seed=estimator_num)
            else:
                raise ValueError(f'Model {meta_learner_name} is not supported as a meta-learner')

            _clf = GridSearchCV(estimator=estimator,
                                param_grid=meta_learner_params,
                                cv=cv,
                                n_jobs=n_jobs,
                                scoring=scoring)
            estimators.append((f'{meta_learner_name}-{estimator_num}', _clf))

        # Constructing ensemble
        ensemble_clf = VotingClassifier(estimators=estimators,
                                        voting='soft',
                                        weights=None,
                                        verbose=True)

        return ensemble_clf
