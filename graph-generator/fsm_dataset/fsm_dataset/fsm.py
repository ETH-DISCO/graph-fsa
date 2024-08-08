from graphviz import Digraph
from itertools import product
import numpy as np
import pickle
import random
import os

from .gnn_simulator import NeuralFSM

class FSM:
    def __init__(self, num_states, starting_states, final_states, transition_matrix):
        self.num_states = num_states
        self.starting_states = starting_states
        self.final_states = final_states

        self.matrix = transition_matrix

        self.gnn = NeuralFSM()
        self.gnn.eval()
        self.logs = [] 

    def simulate(self, start_states, edge_index, iterations=10):
        self.gnn.set_iterations(iterations)
        output_states = self.gnn(start_states, edge_index, self.matrix)
        self.logs.append(self.gnn.log)

        return output_states

    # edge_filter means we only visualize iterations used during simulation
    def visualize(self, path, edge_filter=False):
        G = Digraph(
            graph_attr={"rankdir": "LR"}, node_attr={"shape": "circle", "width": "0.8"}
        )

        state_names = ["r" + str(i) for i in range(self.num_states)]

        for i, f in enumerate(self.final_states):
            state_names[f] = "f" + str(i)

        for i, s in enumerate(self.starting_states):
            state_names[s] = "s" + str(i)

        for state_name in state_names:
            G.node(state_name, penwidth="3px")

        formatting_string = "0" + str(len(state_names)) + "b"

        for x, y in product(range(len(state_names)), range(len(state_names))):
            edges = []

            for c in range(2 ** len(state_names)):
            
                if self.matrix[c][x][y] == 1:
                    if edge_filter and (c,x,y) in self.gnn.log:
                        edges.append(c)
                    
                    elif not edge_filter:
                        edges.append(c)

            edge_strings = []
            for e1 in edges:

                def bin_count(n):
                    c = 0
                    while n:
                        c += n & 1
                        n = n >> 1
                    return c

                edge_string = "[" + str(format(e1, formatting_string)) + "]"
                edge_strings.append(edge_string)

            for edge in set(edge_strings):
                G.edge(state_names[x], state_names[y], "".join(edge))

        G.render(path, format="png")


def load_fsm(path):
    f = open(path, "rb")
    fsm = pickle.load(f)
    f.close()
    return fsm


def save_fsm(fsm, path):
    # make sure the directory exists
    directorypath = "/".join(path.split("/")[:-1])
    if not os.path.exists(directorypath):
        os.makedirs(directorypath)

    print("model path", path)
    f = open(path, "wb")
    pickle.dump(fsm, f)
    f.close()



def generate_sparse_fsm(
    number_states, starting_states, final_states, seed, enforce_final=True, stay_in_state_prob = 0.2
):

    random.seed(seed)

    # we create a random deterministic state machine
    transition_matrix = np.zeros((2 ** number_states, number_states, number_states))

    # add random iterations to starting states
    transitions = [i for i in range(2 ** number_states)]

    for s in range(number_states):

        if s in final_states and enforce_final:
            continue

        weights = [0 for _ in range(number_states)]
        next_states = []

        while len(next_states) < 2: 
            next_state = random.randint(0, number_states-1)
            if next_state == s or next_state in next_states:
                continue
            next_states.append(next_state)
        
        prop = (1 - stay_in_state_prob) / 2
        weights[next_states[0]] = prop
        weights[next_states[1]] = prop
        weights[s] = stay_in_state_prob

        for t in range(2 ** number_states):

            s2 = random.choices([i for i in range(number_states)], weights=weights, k=1)[0]
            transition_matrix[t][s][s2] = 1

    if enforce_final:
        for f in final_states:
            for t in range(2 ** number_states):
                transition_matrix[t][f][f] = 1

    fsmObject = FSM(number_states, starting_states, final_states, transition_matrix)
    return fsmObject    


def generate_fsm(
    number_states, starting_states, final_states, seed, enforce_final=True
):
    random.seed(seed)

    # we create a random deterministic state machine
    transition_matrix = np.zeros((2 ** number_states, number_states, number_states))

    # add random iterations to starting states
    transitions = [i for i in range(2 ** number_states)]

    for t in range(2 ** number_states):
        for s in range(number_states):

            if s in final_states and enforce_final:
                continue

            s2 = random.randint(0, number_states - 1)
            transition_matrix[t][s][s2] = 1

    if enforce_final:
        for f in final_states:
            for t in range(2 ** number_states):
                transition_matrix[t][f][f] = 1

    fsmObject = FSM(number_states, starting_states, final_states, transition_matrix)
    return fsmObject    