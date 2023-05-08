import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve,
)


TEST_SIZE = 0.3
RANDOM_SEED = 42


def get_metrics(y_true, y_pred, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pr, rc, _ = pr_curve = precision_recall_curve(y_true, y_score)

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
    n_pos = y.sum()
    n_neg = (~y).sum()

    # Creating the datasets
    X_pos, y_pos = X.loc[y], y.loc[y]
    X_neg, y_neg = X.loc[~y], y.loc[~y]

    # Creating balanced datasets: subsample the negative data
    # to make the positive and negative data compatible
    if balanced:
        i = np.arange(n_neg)
        i_neg = np.random.choice(i, n_pos)
        X_neg, y_neg = X_neg.iloc[i_neg], y_neg.iloc[i_neg]

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


def train_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    balanced: bool = True,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
):
    X_train, X_test, y_train, y_test = get_datasets(
        X, y, test_size=test_size, balanced=balanced, random_seed=random_seed
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
