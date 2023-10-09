import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.3
RANDOM_SEED = 42


def get_metrics(y_true, y_pred, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pr, rc, _ = precision_recall_curve(y_true, y_score)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "pr": pr,
        "rc": rc,
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(rc, pr),
    }


def get_datasets(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    balanced: bool = True,
    random_seed=RANDOM_SEED,
):
    i = y.astype(bool)
    n_pos = i.sum()
    n_neg = (~i).sum()

    # Creating the datasets
    X_pos, y_pos = X[i], y[i]
    X_neg, y_neg = X[~i], y[~i]

    # Creating balanced datasets: subsample the negative data
    # to make the positive and negative data compatible
    if balanced:
        i = np.arange(n_neg)
        i_neg = np.random.choice(i, n_pos)
        X_neg, y_neg = X_neg[i_neg], y_neg[i_neg]

    # Use the same number of positive-negative points in the train/test sets
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
        X_pos, y_pos, test_size=test_size, random_state=random_seed
    )
    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(
        X_neg, y_neg, test_size=test_size, random_state=random_seed
    )

    # concatenate the datasets
    concat_fn = pd.concat if isinstance(X, pd.DataFrame) else np.concatenate

    X_train = concat_fn([X_train_pos, X_train_neg])
    X_test = concat_fn([X_test_pos, X_test_neg])
    y_train = concat_fn([y_train_pos, y_train_neg]).astype(int)
    y_test = concat_fn([y_test_pos, y_test_neg]).astype(int)

    return X_train, X_test, y_train, y_test


def get_datasets_with_validation(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = TEST_SIZE,
    test_size: float = TEST_SIZE,
    balanced: bool = True,
    random_seed=RANDOM_SEED,
):
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = get_datasets(
        X,
        y,
        test_size=temp_size,
        balanced=balanced,
        random_seed=random_seed,
    )
    X_val, X_test, y_val, y_test = get_datasets(
        X_temp,
        y_temp,
        test_size=(test_size / temp_size),
        balanced=balanced,
        random_seed=random_seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    balanced: bool = True,
    normalized: bool = True,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
):
    if normalized:
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)

    X_train, X_test, y_train, y_test = get_datasets(
        X,
        y,
        test_size=test_size,
        balanced=balanced,
        random_seed=random_seed,
    )

    # Fitting the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    return {
        "classifier": type(clf).__name__,
        "y_test": list(y_test),
        "y_pred": list(y_pred),
        "n_pos": y_train.sum(),
        "n_neg": (1 - y_train).sum(),
        "seed": random_seed,
        **get_metrics(y_test, y_pred, y_score),
    }


def get_best_classifier(**params):
    import xgboost as xgb

    BEST_HYPERPARAMS = {
        "colsample_bytree": 0.5,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "n_estimators": 200,
        "subsample": 0.5,
    }

    return xgb.XGBClassifier(
        objective="binary:logistic",
        **BEST_HYPERPARAMS,
        **params,
    )
