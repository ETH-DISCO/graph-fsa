from typing import Callable, Optional

import torch
from torch_geometric.utils import from_networkx
import networkx as nx

from torch_geometric.data import Data, InMemoryDataset
import dataset_generator.utils as utils
import numpy as np


class Distance:
    def __init__(self, num_nodes=8, num_graphs=10, min_diameter=1, max_diameter=None):

        self.num_features = 2
        self.num_classes = 2

        self.data = self.__makedata(num_graphs, num_nodes, min_diameter, max_diameter)

    def __makedata(
        self, num_graphs=200, num_nodes=8, min_diameter=1, max_diameter=None
    ):
        return [
            self.__gen_graph(num_nodes, min_diameter, max_diameter)
            for _ in range(num_graphs)
        ]

    def __gen_graph(self, num_nodes, min_diameter, max_diameter):

        if not max_diameter:
            g = utils.randomgraph(num_nodes)
        else:
            g = utils.randomgraph_dimater(num_nodes, min_diameter, max_diameter)

        origin = np.random.randint(0, num_nodes)
        queue = [(origin, 0)]
        seen = {origin}
        dist_0 = set()

        while queue:
            node, distance = queue.pop(0)
            if distance % 2 == 0:
                dist_0.add(node)

            for nb in g.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, distance + 1))
        data = from_networkx(g)
        data.x = torch.tensor(
            [[1.0, 0.0] if x != origin else [0.0, 1.0] for x in range(num_nodes)]
        )

        data.edge_attr = torch.ones(g.number_of_edges() * 2, 1)

        data.y = torch.tensor([0.0 if n in dist_0 else 1.0 for n in range(num_nodes)])
        y = data.y

        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True

        # set all masks as we always use the entire graph for eval/test/train
        data.train_mask = train_mask
        data.val_mask = train_mask
        data.test_mask = train_mask

        return data
