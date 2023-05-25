import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
from zeo_amd.hparams import HyperparameterOptimizer

warnings.filterwarnings("ignore")


def get_ranges():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import xgboost as xgb

    classifiers_hyperparameters = [
        (
            LogisticRegression,
            {
                "penalty": ["l2", "none"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear", "sag", "saga"],
            },
        ),
        (
            LogisticRegression,
            {
                "penalty": ["l1"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["saga"],
                "l1_ratio": [0.25, 0.5, 0.75, 1],
            },
        ),
        (
            RandomForestClassifier,
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            },
        ),
        (
            xgb.XGBClassifier,
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6],
                "min_child_weight": [1, 2, 3],
                "subsample": [0.5, 0.75, 1],
                "colsample_bytree": [0.5, 0.75, 1],
            },
        ),
    ]
    return classifiers_hyperparameters


def get_data(distance_matrix, synthesis_table):
    dm = pd.read_csv(distance_matrix, index_col=0)
    synth = pd.read_csv(synthesis_table, index_col=0)
    synth = synth.loc[dm.index]

    return dm, synth


def get_args():
    parser = argparse.ArgumentParser(description="Compare two folders using the AMD.")
    parser.add_argument("distance_matrix", type=str, help="path to distance matrix")
    parser.add_argument("synthesis_table", type=str, help="path to synthesis table")
    parser.add_argument(
        "--min_synthesis",
        type=float,
        default=0.25,
        help="minimum fraction of synthesis recipes to consider a positive label",
    )
    parser.add_argument(
        "--min_positive",
        type=int,
        default=10,
        help="minimum number of positive labels in a dataset",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="size of the validation set",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="size of the test set",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="number of runs",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="number of runs",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "-b",
        "--balanced",
        action="store_true",
        default=False,
        help="If true, creates a balanced dataset",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="If true, normalizes the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.json",
        help="Name of the output file that will be generated when creating the results (default: output.json)",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if os.path.exists(args.output):
        sys.exit()

    dm, synth = get_data(args.distance_matrix, args.synthesis_table)
    classifiers_hyperparameters = get_ranges()

    results = []
    for _label in synth.columns:
        # Get the information for the dataset
        X = dm.values
        y = (synth[_label] > args.min_synthesis).values

        n_pos = y.sum()

        if n_pos < args.min_positive:
            continue

        for i, (cls, ranges) in enumerate(classifiers_hyperparameters):
            print(_label, cls.__name__)

            for run in range(args.n_runs):
                seed = random.randint(0, 100000) if args.seed is None else args.seed

                opt = HyperparameterOptimizer(
                    cls,
                    ranges,
                    val_size=args.val_size,
                    test_size=args.test_size,
                    balanced=args.balanced,
                    normalized=args.normalized,
                    random_seed=seed,
                    label=_label,
                )

                res = opt.optimize_hyperparameters(X, y, n_workers=args.n_workers)

                for _dict in res:
                    _dict["params_index"] = i
                    _dict["run"] = run

                results += res

    df = pd.DataFrame(results)
    df.to_json(args.output)


if __name__ == "__main__":
    main()
