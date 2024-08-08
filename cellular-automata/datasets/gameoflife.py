import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

class GameOfLifeGraph:
    
    def __init__(self, grid_size=8, num_graphs=1, steps=1, toroidal=False, seed=None):
        self.toroidal = toroidal
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
            if self.toroidal:
                neighbors = [(i % grid_size, j % grid_size) for i in range(x-1, x+2) for j in range(y-1, y+2) 
                                if (i, j) != (x, y)]
            else:
                neighbors = [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) 
                                if (i, j) != (x, y) and 0 <= i < grid_size and 0 <= j < grid_size]
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
                

        # Randomly initialize cell values
        labels_random = np.random.choice([0, 1], size=(grid_size * grid_size, 1))

        # Apply Game of Life rules
        labels = labels_random
        for _ in range(steps):
            labels = self.__apply_game_of_life(G, labels, grid_size)

        # Convert to tensor
        labels_random_tensor = torch.tensor(labels_random, dtype=torch.float)
        labels_list = [label[0] for label in labels]
        labels_tensor = torch.tensor(labels_list, dtype=torch.float)

        # Convert networkx graph to PyG Data
        dG = from_networkx(G)
        dG.x = labels_random_tensor
        dG.y = labels_tensor

        return dG

    def __apply_game_of_life(self, G, labels, grid_size):
        new_labels = np.zeros_like(labels)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            live_neighbors = sum([labels[self.__node_to_idx(neighbor, grid_size)] for neighbor in neighbors])

            # Apply Game of Life rules
            if labels[self.__node_to_idx(node, grid_size)] == 1:
            
                if live_neighbors < 2 or live_neighbors > 3:
                    new_labels[self.__node_to_idx(node, grid_size)] = 0
                else:
                    new_labels[self.__node_to_idx(node, grid_size)] = 1
            else:
                if live_neighbors == 3: # or live_neighbors == 6:
                    new_labels[self.__node_to_idx(node, grid_size)] = 1

        return new_labels

    def __node_to_idx(self, node, grid_size):
        return node[0] * grid_size + node[1]



class HexagonalGameOfLifeGraph:
    
    def __init__(self, grid_size=8, num_graphs=1, steps=1, toroidal=False, seed=None):        
    
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

        # Generate hexagonal neighborhood for each node
        for node in G.nodes():
            x, y = node
            if x % 2 == 0:  # Even rows
                neighbors = [(x-1, y-1), (x-1, y), (x, y-1),
                            (x, y+1), (x+1, y-1), (x+1, y)]
            else:  # Odd rows
                neighbors = [(x-1, y), (x-1, y+1), (x, y-1),
                            (x, y+1), (x+1, y), (x+1, y+1)]
            for neighbor in neighbors:
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                    G.add_edge(node, neighbor)
        # Randomly initialize cell values
        labels_random = np.random.choice([0, 1], size=(grid_size * grid_size, 1))

        # Apply Game of Life rules
        labels = labels_random
        for _ in range(steps):
            labels = self.__apply_game_of_life(G, labels, grid_size)

        # Convert to tensor
        labels_random_tensor = torch.tensor(labels_random, dtype=torch.float)
        labels_list = [label[0] for label in labels]
        labels_tensor = torch.tensor(labels_list, dtype=torch.float)

        # Convert networkx graph to PyG Data
        dG = from_networkx(G)
        dG.x = labels_random_tensor
        dG.y = labels_tensor

        return dG

    def __apply_game_of_life(self, G, labels, grid_size):
        new_labels = np.zeros_like(labels)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            live_neighbors = sum([labels[self.__node_to_idx(neighbor, grid_size)] for neighbor in neighbors])

            # Apply Game of Life rules for hexagonal grid
            if labels[self.__node_to_idx(node, grid_size)] == 1:
                if live_neighbors < 2 or live_neighbors > 4:
                    new_labels[self.__node_to_idx(node, grid_size)] = 0
                else:
                    new_labels[self.__node_to_idx(node, grid_size)] = 1
            else:
                if live_neighbors == 3:
                    new_labels[self.__node_to_idx(node, grid_size)] = 1
            
        return new_labels

    def __node_to_idx(self, node, grid_size):
        return node[0] * grid_size + node[1]
