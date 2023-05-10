import itertools
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform, pdist


def get_graph(dm: pd.DataFrame):
    graph = nx.Graph()

    graph.add_weighted_edges_from(
        (i1, i2, d) for (i1, i2), d in zip(itertools.combinations(range(len(dm)), 2), squareform(dm.values))
    )
    _attrs = {i: z for i, z in enumerate(dm.index)}
    nx.set_node_attributes(graph, _attrs, "zeo")
    return graph


def get_mst(dm: pd.DataFrame):
    graph = get_graph(dm)
    return nx.minimum_spanning_tree(graph)
