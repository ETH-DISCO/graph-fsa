import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

class OneDimensionalAutomataGridGraph:

    def __init__(self, grid_size=8, num_graphs=1, steps=1, rule_number=30, seed=None):

        if seed is not None: 
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rule_number = rule_number
        self.rule_bin = format(rule_number, '08b')[::-1]  # binary representation of the rule, reversed for easy indexing
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

        # Remove horizonal edges to get 1D neighborhood
        # removing horizontal edges makes it easier for our graph numbering
        for (x1, y1), (x2, y2) in list(G.edges()):
            if x1 != x2:
                G.remove_edge((x1, y1), (x2, y2))
        

        # Randomly initialize cell values for each row
        labels_random = np.random.choice([0, 1], size=(grid_size, grid_size))

        # Apply 1D automata rules for each row
        labels = labels_random.copy()
        for _ in range(steps):
            for row in range(grid_size):
                labels[row] = self.__apply_1d_automata(labels[row])

        # Convert to tensor
        labels_random_tensor = torch.tensor(labels_random, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        # Convert networkx graph to PyG Data
        dG = from_networkx(G)
        dG.x = labels_random_tensor.view(-1, 1)
        dG.y = labels_tensor.view(-1, 1)

        return dG

    def __apply_1d_automata(self, row_labels):
        new_labels = np.zeros_like(row_labels)
        for idx in range(len(row_labels)):
            left_val = row_labels[idx - 1] if idx - 1 >= 0 else 0
            center_val = row_labels[idx]
            right_val = row_labels[idx + 1] if idx + 1 < len(row_labels) else 0

            # Convert triplet to index for the rule
            triplet = f"{int(left_val)}{int(center_val)}{int(right_val)}"
            rule_idx = int(triplet, 2)
            new_labels[idx] = int(self.rule_bin[rule_idx])

            #print("triplet", triplet)

        return new_labels


if __name__ == "__main__":
    # Example usage
    dataset = OneDimensionalAutomataGridGraph(grid_size=5, num_graphs=3, steps=2, rule_number=10)


