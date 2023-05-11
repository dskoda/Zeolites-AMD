import tqdm
from typing import Union, List
import numpy as np
import pandas as pd
import multiprocess as mp
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error
from .classify import train_classifier, get_datasets_with_validation, get_metrics


class HyperparameterOptimizer:
    def __init__(
        self,
        classifier_class,
        hyperparameter_ranges,
        val_size: float = 0.2,
        test_size: float = 0.2,
        balanced: bool = True,
        random_seed: int = 42,
        label: str = None
    ):
        self.Classifier = classifier_class
        self.hyperparameter_ranges = hyperparameter_ranges
        self.val_size = val_size
        self.test_size = test_size
        self.balanced = balanced
        self.seed = random_seed
        self.label = label

    def optimize_hyperparameters(self, X, y, n_workers=8):
        X_train, X_val, X_test, y_train, y_val, y_test = get_datasets_with_validation(
            X, y, self.val_size, self.test_size, self.balanced, self.seed
        )

        param_grid = ParameterGrid(self.hyperparameter_ranges)
        p = mp.Pool(n_workers)

        train_fn = lambda params: self.train(params, X_train, X_val, X_test, y_train, y_val, y_test)

        results = []
        for res in p.imap_unordered(train_fn, param_grid, chunksize=1):
            if res:
                results.append(res)

        return results

    def train(self, params, X_train, X_val, X_test, y_train, y_val, y_test):
        try:
            clf = self.Classifier(**params)
            clf.fit(X_train, y_train)

        except ValueError:
            return {}

        y_val_pred = clf.predict(X_val)
        y_val_score = clf.predict_proba(X_val)[:, 1]

        val_metrics = get_metrics(y_val, y_val_pred, y_val_score)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        y_test_pred = clf.predict(X_test)
        y_test_score = clf.predict_proba(X_test)[:, 1]
        test_metrics = get_metrics(y_test, y_test_pred, y_test_score)
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

        return {
            "label": self.label,
            "classifier": type(clf).__name__,
            "n_pos": y_train.sum(),
            "n_neg": (1 - y_train).sum(),
            "seed": self.seed,
            "params": params,
            **val_metrics,
            **test_metrics,
        }
