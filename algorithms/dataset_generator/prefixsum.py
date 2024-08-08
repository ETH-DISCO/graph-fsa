from typing import Callable, Optional

import torch
from torch_geometric.utils import from_networkx

from torch_geometric.data import Data, InMemoryDataset
import dataset_generator.utils as utils
import numpy as np
import networkx as nx


class PrefixSum:
    # Creates a PrefixSum on a path - output must be sum mod 2
    # input is one-hot [value, isRoot]
    def __init__(self, num_nodes=8, num_graphs=100, rangeValue=0):

        self.num_features = 4
        self.num_classes = 2

        self.data = self.__makedata(num_graphs, num_nodes, rangeValue)

    def __makedata(self, num_graphs=200, num_nodes=8, rangeValue=0):

        graphs = []

        for nodeCount in range(
            max(num_nodes - rangeValue, 1), num_nodes + rangeValue + 1
        ):

            binary_strs = []
            while len(binary_strs) < min(num_graphs, 2 ** nodeCount):
                graph_size = nodeCount
                ss = "".join([str(np.random.randint(0, 2)) for _ in range(graph_size)])
                if ss not in binary_strs:
                    binary_strs.append(ss)
            graphs += [self.__gen_graph(s) for s in binary_strs]

        return graphs

    def __gen_graph(self, s):
        n = len(s)
        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)
        leafs = np.random.permutation([x for x in G.nodes() if G.degree(x) == 1])

        root = rand_perm[0]
        labels = [[0.0, 0.0, 1.0, 0.0] for i in range(n)]
        ylabels = [[0.0] for i in range(n)]
        labels[root] = [0.0, 0.0, 0.0, 1.0]

        counter = 0
        for i, node in enumerate(rand_perm):
            x = int(s[i])
            labels[node][x] = 1.0
            counter = (counter + x) % 2
            ylabels[node] = counter

        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)

        dG.edge_attr = torch.ones(G.number_of_edges() * 2, 1)

        return dG
