import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

class WireworldGraph:
    
    EMPTY = 0
    ELECTRON_HEAD = 1
    ELECTRON_TAIL = 2
    CONDUCTOR = 3

    def __init__(self, grid_size=8, num_graphs=1, steps=1, seed=None):

        if seed is not None: 
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.data = self.__makedata(grid_size, num_graphs, steps)



    def __makedata(self, grid_size, num_graphs, steps):
        graphs = []

        for _ in range(num_graphs):
            g = self.__gen_graph(grid_size, steps)
            graphs.append(g)

        return graphs

    def __gen_graph(self, grid_size, steps):
        # Create a grid graph
        G = nx.grid_2d_graph(grid_size, grid_size)

        # Generate Moore neighborhood for each node
        for node in G.nodes():
            x, y = node
            neighbors = [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) 
                         if (i, j) != (x, y) and 0 <= i < grid_size and 0 <= j < grid_size]
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Randomly initialize cell values
        labels_random = np.random.choice([self.EMPTY, self.ELECTRON_HEAD, self.ELECTRON_TAIL, self.CONDUCTOR],
                                         size=(grid_size * grid_size, 1))

        # Apply Wireworld rules
        labels = labels_random
        for _ in range(steps):
            labels = self.__apply_wireworld(G, labels, grid_size)

        # Convert to tensor
        labels_random_tensor = torch.tensor(labels_random, dtype=torch.float)
        labels_list = [label[0] for label in labels]
        labels_tensor = torch.tensor(labels_list, dtype=torch.float)

        # Convert networkx graph to PyG Data
        dG = from_networkx(G)
        dG.x = labels_random_tensor
        dG.y = labels_tensor

        return dG

    def __apply_wireworld(self, G, labels, grid_size):
        new_labels = np.zeros_like(labels)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            electron_head_neighbors = sum([1 for neighbor in neighbors if labels[self.__node_to_idx(neighbor, grid_size)] == self.ELECTRON_HEAD])

            current_label = labels[self.__node_to_idx(node, grid_size)]
            
            # Apply Wireworld rules
            if current_label == self.EMPTY:
                new_labels[self.__node_to_idx(node, grid_size)] = self.EMPTY
            elif current_label == self.ELECTRON_HEAD:
                new_labels[self.__node_to_idx(node, grid_size)] = self.ELECTRON_TAIL
            elif current_label == self.ELECTRON_TAIL:
                new_labels[self.__node_to_idx(node, grid_size)] = self.CONDUCTOR
            else:  # Conductor
                if 1 <= electron_head_neighbors <= 2:
                    new_labels[self.__node_to_idx(node, grid_size)] = self.ELECTRON_HEAD
                else:
                    new_labels[self.__node_to_idx(node, grid_size)] = self.CONDUCTOR

        return new_labels

    def __node_to_idx(self, node, grid_size):
        return node[0] * grid_size + node[1]
