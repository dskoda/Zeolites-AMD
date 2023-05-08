import os
import sys
import argparse
from zeo_amd.compare import compare_folders


def get_args():
    parser = argparse.ArgumentParser(description="Compare two folders using the AMD.")
    parser.add_argument("path_1", type=str, help="path to first folder")
    parser.add_argument("path_2", type=str, help="path to second folder")
    parser.add_argument(
        "-k",
        type=int,
        default=100,
        help="number of neighbors to use in the AMD (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.csv",
        help="Name of the output file that will be generated when creating the distance matrix (default: output.csv)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    k = args.k

    if os.path.exists(args.output):
        sys.exit()

    results = compare_folders(path1, path2, k)
    results.to_csv(args.output)
