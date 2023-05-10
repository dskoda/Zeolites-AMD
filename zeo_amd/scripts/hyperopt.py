import os
import sys
import argparse
import numpy as np
import pandas as pd

from zeo_amd.hparams import HyperparameterOptimizer


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
        "-s", "--seed",
        type=int,
        default=1886,
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

    results = []
    for _label in tqdm.tqdm(synth.columns):
        # Get the information for the dataset
        X = dm.values
        y = (synth[_label] > args.min_synthesis).values
        
        n_pos = y.sum()

        if n_pos < args.min_positive:
            continue
        
        for cls, ranges in classifiers_hyperparameters:
            opt = HyperparameterOptimizer(
                cls,
                ranges,
                val_size=args.val_size,
                test_size=args.test_size,
                balanced=args.balanced,
                random_seed=args.seed,
            )

            results += opt.optimize_hyperparameters(X, y, n_runs=args.n_runs, n_workers=args.n_workers)
            
        break

    df = pd.DataFrame(results)
    df.to_json(args.output)


if __name__ == "__main__":
    main()
