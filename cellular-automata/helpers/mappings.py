import torch
import numpy as np


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


def map_x_to_state(inputs, states_n):
    input_states = []

    for x in inputs:
        tensor = torch.zeros(states_n)
        tensor[int(x[0].item())] = 1
        input_states.append(torch.unsqueeze(tensor, dim=0))

    return torch.cat(input_states, dim=0)
