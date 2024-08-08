import torch
from torch import nn
import numpy as np

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


"""
    GraphFSA - but neighbourhood aware
    This modified aggregation has been used for the 1D-Cellular Automata tests and is neighbourhood aware
"""

# helper function to map multi-hot tensor to one-hot tensor
def multiple_hot_tensor_to_one_hot(tensor, base):
    number = 0

    for i in range(tensor.shape[-1]):
        value = tensor[i]
        number += value * base ** i

    output = torch.zeros(base ** tensor.shape[-1])
    output[number.to(torch.long)] = 1
    return output


class TransitionLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

        self.threshold = torch.zeros(1)
        self.threshold[0] = 0.5

        self.logging = False
        self.transition_log = None

    def forward(self, x, edge_index, T):
        return self.propagate(edge_index, x=x, T=T)

    def aggregate(self, inputs, index, ptr, dim_size):

        out = torch.zeros((dim_size, 2))
        counts = torch.zeros(dim_size, dtype=torch.long, device=inputs.device)
    
        for i, idx in enumerate(index):
            
            if idx % self.GRID_SIZE == 0:
                out[idx, 1] = torch.argmax(inputs[i])

            elif idx % self.GRID_SIZE == self.GRID_SIZE-1:
                out[idx, 0] = torch.argmax(inputs[i])

            else:                
                out[idx, counts[idx]] = torch.argmax(inputs[i])
                counts[idx] += 1
            
        message_sums = out

        aggr_1 = (message_sums > self.threshold[0]).float()
        
        aggr = aggr_1
        aggr = torch.clamp(aggr, min=0, max=1)

        if self.logging:
            self.transition_log = aggr

        return aggr

    def update(self, aggr_out, x, T):
        new_states = []

        for i in range(x.shape[0]):

            state = x[i]
            char = multiple_hot_tensor_to_one_hot(aggr_out[i], 3)

            new_state = torch.einsum("x,s,xst->t", char, state, T)
            new_states.append(torch.unsqueeze(new_state, dim=0))

        return torch.cat(new_states, dim=0)


# Define the FSM
class NeuralCA(nn.Module):
    # CHAR_N is the number of transitions from state to state
    # STATE_N is the number of states in our FSM
    def __init__(self, CHAR_N, STATE_N, GRID_SIZE):
        super().__init__()
       
        self.T = torch.nn.Parameter(
            torch.rand((CHAR_N, STATE_N, STATE_N)),
            requires_grad=True,
        )

        self.iterations = 1

        self.f = TransitionLayer()

        self.f.GRID_SIZE = GRID_SIZE

        self.hardmax = False

        self.logging = False

    def set_logging(self, enable):
        self.logging = enable
        self.f.logging = enable

    def set_grid_size(self, GRID_SIZE):
        self.f.GRID_SIZE = GRID_SIZE

    def set_iterations(self, number):
        self.iterations = number

    def set_hardmax(self, use_hardmax):
        self.hardmax = use_hardmax

    def forward(self, s0, edge_index):

        transitions = []

        if self.hardmax:
            T = torch.nn.functional.one_hot(
                torch.argmax(self.T, dim=-1), self.T.shape[-1]
            ).to(torch.float32)
        else:
            T = torch.nn.functional.softmax(self.T, dim=-1)
   
        states = []
        states.append(torch.unsqueeze(s0, dim=0))

        s = s0

        for _ in range(self.iterations):
            new_s = self.f(s, edge_index, T)
            if self.logging:
                transitions.append(self.f.transition_log)
            states.append(torch.unsqueeze(new_s, dim=0))
            s = new_s

        if self.logging:
            return states[-1][0], torch.cat(states, dim=0), transitions

        # at the moment we only consider the last state (output)
        # states = torch.cat(states[-1], dim=0)
        states = states[-1][0]

        return states
