import torch
from torch import nn
import numpy as np

from torch_geometric.nn import MessagePassing


# helper function to map multi-hot tensor to one-hot tensor
def multiple_hot_tensor_to_one_hot(tensor, base):
    number = 0

    for i in range(tensor.shape[-1]):
        value = tensor[i]
        number += value * base ** i

    output = torch.zeros(base ** tensor.shape[-1])
    output[number.to(torch.long)] = 1
    return output


def one_hot_tensor_to_number(tensor):
    listTensor = tensor.tolist()
    return listTensor.index(1)


class TransitionLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

        self.threshold = torch.zeros(1)
        self.threshold[0] = 0.5

    def forward(self, x, edge_index, T):
        self.transition_char = []
        return self.propagate(edge_index, x=x, T=T)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

        aggr = (message_sums > self.threshold[0]).float()
        aggr = torch.clamp(aggr, min=0, max=1)

        return aggr

    def update(self, aggr_out, x, T):
        new_states = []

        for i in range(x.shape[0]):

            state = x[i]
            char = multiple_hot_tensor_to_one_hot(aggr_out[i], 2)
            self.transition_char.append(char)

            new_state = torch.einsum("x,s,xst->t", char, state, T)
            new_states.append(torch.unsqueeze(new_state, dim=0))

        return torch.cat(new_states, dim=0)


# Define the FSM
class NeuralFSM(nn.Module):
    # CHAR_N is the number of transitions from state to state
    # STATE_N is the number of states in our FSM
    def __init__(self):
        super().__init__()

        self.iterations = 20

        self.f = TransitionLayer()
        self.log = []

    def set_iterations(self, number):
        self.iterations = number

    def forward(self, s0, edge_index, matrix):

        self.log = []
        states = []
        states.append(torch.unsqueeze(s0, dim=0))

        T = torch.from_numpy(matrix).to(torch.float32)
        s = s0
        log_list = []

        for _ in range(self.iterations):
            new_s = self.f(s, edge_index, T)
            states.append(torch.unsqueeze(new_s, dim=0))

            transitionTuples = [
                (
                    one_hot_tensor_to_number(self.f.transition_char[i]),
                    one_hot_tensor_to_number(s[i]),
                    one_hot_tensor_to_number(new_s[i]),
                )
                for i in range(len(s))
            ]
            log_list.append(transitionTuples)

            s = new_s

        self.log = log_list

        # at the moment we only consider the last state (output)
        # states = torch.cat(states[-1], dim=0)
        states = states[-1][0]

        return states
