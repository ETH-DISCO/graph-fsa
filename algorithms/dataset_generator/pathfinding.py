from typing import Callable, Optional

import torch
from torch_geometric.utils import from_networkx

from torch_geometric.data import Data, InMemoryDataset
import dataset_generator.utils as utils
import numpy as np
import networkx as nx


class PathFinding:
    # Creates a Tree and marks the shortest path between two nodes
    # input is one-hot [isEndpoint]
    def __init__(self, num_nodes=15, num_graphs=200, rangeValue=0):
        super().__init__()
        self.num_classes = 2
        self.num_features = 2

        self.data = self.__makedata(num_graphs, num_nodes, rangeValue)

    def __gen_graph(self, num_nodes, num):
        nx_graph = nx.random_tree(n=num_nodes, seed=num)
        tree = from_networkx(nx_graph)
        tree.x = torch.zeros(num_nodes, 2)
        tree.y = torch.zeros(num_nodes)
        tree.x[0][1] = 1
        tree.x[1][1] = 1
        shortest_path = nx.shortest_path(nx_graph, source=0, target=1)
        for node in shortest_path:
            tree.y[node] = 1
        for node in range(num_nodes):
            if tree.x[node][1] == 0:
                tree.x[node][0] = 1
        tree.edge_attr = torch.ones(nx_graph.number_of_edges() * 2, 1)
        return tree

    def __makedata(self, num_graphs=200, num_nodes=8, rangeValue=0, allow_sizes=False):
        graphs = []
        for nodes in range(max(num_nodes - rangeValue, 2), num_nodes + rangeValue + 1):
            graphs += [self.__gen_graph(nodes, i) for i in range(num_graphs)]
        return graphs
