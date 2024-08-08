import networkx as nx
import numpy as np
import random


def randomgraph_dimater(n, min_diameter, max_diameter):

    diameter = random.randint(min_diameter, max_diameter)
    g = nx.complete_graph(n)

    count = 0
    # we remove random edges until goal diameter "diameter" or more is reached
    while True:
        current_diameter = nx.diameter(g)
        count += 1

        if count > 1000:
            count = 0
            g = nx.complete_graph(n)

        if current_diameter >= diameter:
            return g

        if n >= 100 and max_diameter > 2:
            repeat = n
        elif n >= 100:
            repeat = 500
        else:
            repeat = 1

        for _ in range(repeat):
            i, j = np.random.permutation(n)[:2]
            if g.has_edge(i, j):
                g.remove_edge(i, j)
                if not nx.is_connected(g):
                    g.add_edge(i, j)


def randomgraph(n, **args):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    tree = set()
    nodes = list(range(n))
    current = np.random.choice(nodes)
    tree.add(current)
    while len(tree) < n:
        nxt = np.random.choice(nodes)
        if not nxt in tree:
            tree.add(nxt)
            g.add_edge(current, nxt)
            g.add_edge(nxt, current)
        current = nxt

    for _ in range(n // 5):
        i, j = np.random.permutation(n)[:2]
        while g.has_edge(i, j):
            i, j = np.random.permutation(n)[:2]
        g.add_edge(i, j)
        g.add_edge(j, i)
    return g


def get_localized_distances(g, n):
    seen = set()
    distances = {}
    queue = [(n, 0)]
    while queue:
        node, distance = queue.pop(0)
        if node in distances and distances[node] < distance:
            continue
        distances[node] = distance
        for nb in g.neighbors(node):
            if nb not in seen:
                seen.add(node)
                queue.append((nb, distance + 1))
    return [distances[i] for i in range(g.number_of_nodes())]
