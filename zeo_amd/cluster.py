import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score


def get_positive_clusters(
    true_labels: np.ndarray, clusters: np.ndarray, min_cluster_size: int = 2
):
    df = pd.DataFrame([true_labels, clusters], index=["label", "cluster"]).T

    grp = df.groupby("cluster")["label"]
    positive = grp.count()[(grp.sum() > 0)]
    idx = positive.loc[positive >= min_cluster_size].index

    df = df.loc[df.cluster.isin(idx)]

    return df["label"].values, df["cluster"].values


def recall_homogeneity(true_labels: np.ndarray, clusters: np.ndarray):
    """Computes the homogeneity of a given clustering by:

        1. Selecting only the clusters with at least one positive label
        2. Computing the homogeneity per cluster
        3. Normalizing the homogeneity for all clusters

    Arguments:
        true_labels (list): true binary labels
        clusters (list): assigned clusters
    """

    df = get_positive_clusters(true_labels, clusters)

    if len(df) == 0:
        return np.nan

    # compute the homogeneity on a per-cluster basis
    grp = df.groupby("cluster")
    H = grp.mean()
    min_score = 1 / grp.count()

    norm_H = (H - min_score) / (1 - min_score)

    return norm_H.mean().item()
