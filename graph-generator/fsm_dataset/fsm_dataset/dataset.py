import torch
from torch_geometric.data import InMemoryDataset, download_url
from enum import Enum

from .fsm import generate_fsm, FSM, save_fsm, load_fsm, generate_sparse_fsm

import os
from torch_geometric.utils import from_networkx
import random
import json
import hashlib
from . import helper

# we introduce the FSMDataset which holds graph_type and a specifc seed
# graph_type -> defines which graph type to use
# num_states -> number of states used in generated FSM
# final_states -> list of final states in fsm (index entries in list with int)
# starting_states -> list of starting states in fsm (index entries in list with int)


# TODO: is this assumption reasonable or does this need to be fixed?
# we are always generating 1000 graphs with size -> nodes = 6
# as we genereate randomly it might be that we have duplicate graphs
# it might be that there are less then 1000 possible combinations in this case we definetly will have duplicate graphs

class GRAPH_TYPE(Enum):
    TREE = 1
    FULLY_CONNECTED = 2
    PATH = 3
    REGULAR_3 = 4
    CYCLE = 5
    GRID = 6
    ERDOS_RENY = 7


class FSMDataset(InMemoryDataset):
    def __init__(
        self, root, graph_type, num_states, starting_states, final_states, seed=1, sparse=False
    ):

        self.graph_type = graph_type
        self.seed = seed
        self.start_states = starting_states
        self.final_states = final_states
        self.num_states = num_states
        self.sparse = sparse
        self.fsm = None
        random.seed(seed)

        super().__init__(root, None, None, None)

        self.data = torch.load(self.processed_paths[0])
        self.val_data = torch.load(self.processed_paths[1])
        self.test_data = torch.load(self.processed_paths[2])
        self.ood_test_data = torch.load(self.processed_paths[3])
        self.fsm = load_fsm(self.processed_paths[4])


    def __get_dataset_hash(self):
        
        if self.sparse:
            hashTuple = (
                str(self.graph_type),
                self.num_states,
                str(self.start_states),
                str(self.final_states),
                self.seed,
                self.sparse,
            )
        else: 
            hashTuple = (
                str(self.graph_type),
                self.num_states,
                str(self.start_states),
                str(self.final_states),
                self.seed                
            )


        dataset_hash = hashlib.md5(str(hashTuple).encode("utf-8")).hexdigest()
        return str(dataset_hash)

    @property
    def raw_file_names(self):

        dataset_hash = self.__get_dataset_hash()

        base_path = dataset_hash
        print("base path", base_path)
        
        file_name_dataset = "data_" + dataset_hash + ".pt"
        file_name_val_dataset = "data_val_" + dataset_hash + ".pt"
        file_name_test_dataset = "data_test_" + dataset_hash + ".pt"
        file_name_ood_test_dataset = "data_ood_test_" + dataset_hash + ".pt"

        file_name_fsm = "fsm_" + dataset_hash + ".pickle"
        file_name_info = "info_" + dataset_hash + ".json"

        if self.sparse:
            file_name_fsm_visual = "fsm_sparse_" + dataset_hash
        else: 
            file_name_fsm_visual = "fsm_" + dataset_hash

        file_name_fsm_visual_filter = "fsm_filter_" + dataset_hash
        file_name_fsm_visual_filter_big = "fsm_filter_extrapolate_" + dataset_hash

        return [
            os.path.join(base_path, elm) for elm in [
                file_name_dataset,
                file_name_val_dataset,
                file_name_test_dataset,
                file_name_ood_test_dataset,
                file_name_fsm,
                file_name_info,
                file_name_fsm_visual,
                file_name_fsm_visual_filter,
                file_name_fsm_visual_filter_big
            ]
        ]

    @property
    def processed_file_names(self):
        dataset_hash = self.__get_dataset_hash()

        dataset_name = "data_" + dataset_hash + ".pt"
        dataset_val_name = "data_val_" + dataset_hash + ".pt"
        dataset_test_name = "data_test_" + dataset_hash + ".pt"
        dataset_ood_test_name = "data_ood_test_" + dataset_hash + ".pt"
        dataset_fsm_name = "fsm_" + dataset_hash + ".pt"

        return [dataset_name, dataset_val_name, dataset_test_name, dataset_ood_test_name, dataset_fsm_name]

    def generate_graph(self, num_nodes, seed):

        if self.graph_type == GRAPH_TYPE.TREE:
            tree = helper.generate_tree(num_nodes, seed)
            return tree

        elif self.graph_type == GRAPH_TYPE.FULLY_CONNECTED:
            graph = helper.generate_fully_connected_graph(num_nodes, seed)
            return graph

        elif self.graph_type == GRAPH_TYPE.PATH:
            path = helper.generate_path(num_nodes, seed)
            return path

        elif self.graph_type == GRAPH_TYPE.REGULAR_3:
            regular = helper.generate_regular3_graph(num_nodes, seed)
            return regular

        elif self.graph_type == GRAPH_TYPE.CYCLE:
            cycle = helper.generate_cycle_graph(num_nodes, seed)
            return cycle

        elif self.graph_type == GRAPH_TYPE.GRID:
            grid = helper.generate_grid_graph(num_nodes, seed)
            return grid

        elif self.graph_type == GRAPH_TYPE.ERDOS_RENY:
            graph = helper.generate_erdos_renyi_graph(num_nodes, seed)
            return graph


    # for grid graph we generate num_nodes x num_nodes big grid
    def generate_graphs(self, fsm, num_nodes, num_graphs, iterations=10, start_seed=0, node_threshold=0.72, ood=False):
        print("node_threshold", node_threshold)

        graphs = []
        count = 0

        while len(graphs) < num_graphs:
            nx_graph = self.generate_graph(num_nodes, start_seed + count)
            graph = from_networkx(nx_graph)
            graph.x = torch.zeros(num_nodes, fsm.num_states)
            graph.edge_attr = torch.ones(nx_graph.number_of_edges() * 2, 1)
            graph.mask = torch.ones(num_nodes)

            #graph.node_attrs = ["state_" + str(i) for i in range(fsm.num_states)]

            # create weight tuple for out-of-distribution data
            weights = None

            # implementing test with node in one state
            extra_node_id = random.randint(0, len(graph.x)-1)

            #if ood: 
            #    # distribute 20% of probability on all except one state (80%) to generate OOD - dataset
            #    weights = [20 / (len(fsm.starting_states) - 2) for _ in range(len(fsm.starting_states)-1)]
            #    focus_state = fsm.starting_states.index(random.choices(fsm.starting_states)[0])
            #    weights[focus_state] = 80
            #    # make sure last starting state is not used
            #    weights[len(fsm.starting_states) - 1] = 0

            for node_id, node in enumerate(graph.x):

                if node_id == extra_node_id:
                    node[fsm.starting_states[-1]] = 1
                
                else:
                    starting_state = random.choices(fsm.starting_states, weights=weights)[0]
                    node[starting_state] = 1

            output = fsm.simulate(graph.x, graph.edge_index, iterations)
            graph.y = output
            counter = 0

            # todo check that all states are in final state - if not we need to find fallback logic
            for i, entry in enumerate(output.tolist()):
                if entry.index(1) in self.final_states:
                    counter += 1
                else:
                    graph.mask[i] = 0

            if counter/num_nodes >= node_threshold:
                graphs.append(graph)
            # remove last entry in fsm log
            else:
                fsm.logs = fsm.logs[:-1]

                
            count += 1
            
        return graphs

    # in our case download means that we are generating the necessary graphs with the seed generated
    def download(self):

        dataset_path = self.raw_paths[0]
        val_dataset_path = self.raw_paths[1]
        test_dataset_path = self.raw_paths[2]
        test_ood_dataset_path = self.raw_paths[3]
        fsm_path = self.raw_paths[4]
        info_path = self.raw_paths[5]
        fsm_visual_path = self.raw_paths[6]

        # Step 1: Generate random FSM with our seed

        if self.sparse:

            fsm = generate_sparse_fsm(
                self.num_states, self.start_states, self.final_states, self.seed
            )

        else:
            fsm = generate_fsm(
                self.num_states, self.start_states, self.final_states, self.seed
            )

        self.fsm = fsm

        save_fsm(fsm, fsm_path)
        fsm.visualize(self.raw_paths[8], edge_filter=True)

        # Step 2: Generate datasets
        num_nodes = 6
        node_threshold = 1
        # Val and Train Graphs
        train_graphs = self.generate_graphs(fsm, num_nodes, 1000, node_threshold=node_threshold, ood=True)
        fsm.visualize(self.raw_paths[7], edge_filter=True)
        print("generated train")

        val_graphs = self.generate_graphs(fsm, 6, 100, start_seed=10000, node_threshold=node_threshold, ood=True)
        
        # OOD Test graphs
        ood_test_graphs = self.generate_graphs(fsm, 6, 100, start_seed=42000, ood=True, node_threshold=0.7)
        print("generated ood")

        # Extrapolation Test Graphs        
        test_graphs = []
        # list holding tuples defining (graph size, iteration, num_graphs)

        if self.graph_type == GRAPH_TYPE.GRID:
            sizes = [(625, 30, 20), (1600, 45, 20), (2500, 65, 20)] # , (1000, 120, 5), (10000, 120, 2)]
        else:
            sizes = [(10, 15, 100), (20, 25, 100), (50, 60, 100), (100, 110, 100)] # , (1000, 120, 5), (10000, 120, 2)]
        for s in sizes:
            test_graphs.append(self.generate_graphs(fsm, s[0], s[2], iterations=s[1], node_threshold=0.7, ood=True))

        print("generated extrapolation")

        # save fsm again with logs
        save_fsm(fsm, fsm_path)

        # save dataset lists
        torch.save(train_graphs, dataset_path)
        torch.save(val_graphs, val_dataset_path)
        torch.save(test_graphs, test_dataset_path)
        torch.save(ood_test_graphs, test_ood_dataset_path)

        # Step 3: create info file
        dataset_info = {}
        dataset_info["start_state"] = str(fsm.starting_states)
        dataset_info["final_states"] = str(fsm.final_states)
        dataset_info["num_states"] = str(fsm.num_states)
        dataset_info["seed"] = self.seed
        dataset_info["graph_type"] = str(self.graph_type)

        with open(info_path, "w") as f:
            f.write(json.dumps(dataset_info))

        fsm.visualize(fsm_visual_path)

    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.raw_paths[0])
        val_data_list = torch.load(self.raw_paths[1])
        test_data_list = torch.load(self.raw_paths[2])
        test_ood_data_list = torch.load(self.raw_paths[3])
        fsm = load_fsm(self.raw_paths[4])

        # TODO - do we want to add additional processing here? 
        # if not we can also copy data with os.copy

        torch.save(data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(test_ood_data_list ,self.processed_paths[3])
        save_fsm(fsm, self.processed_paths[4])


# different functions to test dataset generation
if __name__ == "__main__":
    dataset = FSMDataset("test/", GRAPH_TYPE.TREE, 4, [2, 3], [0, 1], seed=3)
