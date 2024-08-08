import networkx as nx
from torch_geometric.utils import from_networkx
import math

# collection of functions to generate graphs
def generate_tree(num_nodes, seed):
    nx_graph = nx.random_tree(n=num_nodes, seed=seed)
    return nx_graph


def generate_fully_connected_graph(num_nodes, seed):
    nx_graph = nx.complete_graph(num_nodes, seed=seed)
    return nx_graph


def generate_path(num_nodes, seed):
    nx_graph = nx.path_graph(num_nodes)
    return nx_graph


def generate_regular3_graph(num_nodes, seed):
    nx_graph = nx.random_regular_graph(3, num_nodes, seed=seed)
    return nx_graph


def generate_cycle_graph(num_nodes, seed):
    nx_graph = nx.cycle_graph(num_nodes)
    return nx_graph


# we only expect num_nodes that have an integer sqrt
def generate_grid_graph(num_nodes, seed):
    dim1 = int(math.sqrt(num_nodes))
    nx_graph = nx.grid_graph(dim=(dim1, dim1))
    return nx_graph


def generate_erdos_renyi_graph(num_nodes, seed):
    nx_graph = nx.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
    return nx_graph
