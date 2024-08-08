from typing import Callable, Optional

import torch
from torch_geometric.utils import from_networkx

from torch_geometric.data import Data, InMemoryDataset
import dataset_generator.utils as utils
import numpy as np
import networkx as nx
import random


class RootValue:
    # Root becomes input value (0,1) - each node on path should learn this value
    # input is one-hot [value, isRoot] while value is 0 if not root
    def __init__(self, num_nodes=8, num_graphs=1, rangeValue=0):

        self.num_features = 3
        self.num_classes = 2

        self.data = self.__makedata(num_nodes, num_graphs, rangeValue)

    def __makedata(self, num_nodes=8, num_graphs=1, rangeValue=0):
        graphs = []

        for i in range(max(1, num_nodes - rangeValue), num_nodes + rangeValue + 1):
            g1 = self.__gen_graph(i, 0)
            g2 = self.__gen_graph(i, 1)
            graphs += [[g1, g2] for _ in range(num_graphs)]

        return [item for sublist in graphs for item in sublist]

    def __gen_graph(self, n, value):

        rand_perm = np.arange(n)
        G = nx.path_graph(rand_perm)

        root = rand_perm[0]

        labels = [[0.0, 0.0, 0.0] for i in range(n)]
        labels[root] = [0.0, 0.0, 1.0]

        # value root
        labels[root][value] = 1

        ylabels = [value for _ in range(n)]

        dG = from_networkx(G)
        dG.y = torch.tensor(ylabels)
        dG.x = torch.tensor(labels)

        dG.edge_attr = torch.ones(G.number_of_edges() * 2, 1)

        return dG
