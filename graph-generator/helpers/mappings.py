import torch
import numpy as np


# TODO: this is a temporary hack - we should probably
# have a second training pipeline to structure the code better
def map_x_to_start_state_id(inputs):

    ids = []
    helper = 0

    for x in inputs:
        if x[0] == 1:
            ids.append(0)
        elif x[1] == 1 and helper == 0:
            ids.append(1)
            helper += 1
        else:
            ids.append(2)

    return ids


def map_x_to_memory(inputs):

    input_memory = []

    for x in inputs:

        input = torch.zeros(2)
        input[0] = 1 if x[0] == 1 else 0
        input[1] = 1 if x[2] == 1 else 0

        input_memory.append(torch.unsqueeze(input, 0))

    return torch.cat(input_memory, dim=0)


def map_x_to_state(inputs, states_n, dataset_name, use_noise=True):
    input_states = []

    help_count = 0

    for x in inputs:

        tensor = torch.zeros(states_n)
        if use_noise:
            tensor = torch.zeros(states_n)

        if dataset_name == "PrefixSum":

            # Prefix sum mapping
            tensor[3] = 1 if x[0] + x[2] == 2 else 0
            tensor[4] = 1 if x[1] + x[2] == 2 else 0
            tensor[5] = 1 if x[0] + x[3] == 2 else 0
            tensor[6] = 1 if x[1] + x[3] == 2 else 0
            tensor[7] = 1 if x[0] + x[3] == 3 else 0
            tensor[8] = 1 if x[1] + x[3] == 3 else 0


        elif dataset_name == "RootValue":

            # RootValue mapping
            tensor[2] = 1 if x[0] + x[2] == 2 else 0
            tensor[3] = 1 if x[1] + x[2] == 2 else 0
            tensor[4] = 1 if x[2] == 0 else 0

        elif dataset_name == "Distance":

            # Distance mapping
            tensor[2] = x[0]
            tensor[3] = x[1]

        elif dataset_name == "PathFinding":

            # PathFinding mapping
            if x[0] == 1:
                tensor[2] = 1
            elif x[1] == 1 and help_count == 0:
                tensor[3] = 1
                help_count = 1
            else:
                tensor[4] = 1
        elif dataset_name == "Coloring":
            #if x[0] == 1:
            tensor[2] = 1
            #else: 
            #    tensor[3] = 1

        else:
            # just map each node to same state (works in the Tree_Cycle case)
            tensor[2] = 1



        input_states.append(torch.unsqueeze(tensor, dim=0))

    return torch.cat(input_states, dim=0)
