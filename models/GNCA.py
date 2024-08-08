import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
 
# Assuming a generalized MessagePassing layer for the sake of simplicity
class GeneralGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden, aggregate='add', **kwargs):
        super(GeneralGNN, self).__init__(aggr=aggregate)
 
        self.mlp_pre = self.get_mlp(
                    in_channels,
                    hidden,
                    hidden,
                    torch.nn.ReLU,
                    last_activation=True,
                )
        self.mlp_post = self.get_mlp(
                    2 * hidden,
                    hidden,
                    out_channels,
                    torch.nn.ReLU,
                    last_activation=False,
                )
 
        self.lin = nn.Linear(hidden, hidden)
 
 
    def forward(self, x, edge_index):
        out = self.mlp_post(torch.cat([self.mlp_pre(x), self.propagate(edge_index, x=x)], dim=1))
        return out
 
    def message(self, x_j):
        return F.relu(self.lin(self.mlp_pre(x_j)))
 
    def update(self, aggr_out):
        return aggr_out
 
 
    def get_mlp(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation_function,
        last_activation=True,
    ):
        modules = [
            torch.nn.Linear(input_dim, int(hidden_dim)),
            activation_function(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            activation_function(),
            torch.nn.Linear(int(hidden_dim), output_dim),
        ]
        if last_activation:
            modules.append(activation_function())
        return torch.nn.Sequential(*modules)
 
 
 
class GNNCASimple(nn.Module):
    def __init__(
        self,
        activation=F.relu,
        message_passing="add",
        batch_norm=False,
        hidden=256,
        hidden_activation=F.relu,
        dim_size=1,
        **kwargs
    ):
        super(GNNCASimple, self).__init__(**kwargs)
        self.activation = activation
        self.batch_norm = batch_norm
        self.hidden = hidden
        self.hidden_activation = hidden_activation
        self.aggregate = message_passing
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden)
        self.mp = GeneralGNN(dim_size, dim_size, hidden, aggregate=self.aggregate)
 
        self.iterations = 1
        self.hardmax = False
 
    def set_iterations(self, iterations):
        self.iterations = iterations
 
    def get_mlp(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        normalization,
        activation_function,
        last_activation=True,
    ):
        modules = [
            torch.nn.Linear(input_dim, int(hidden_dim)),
            normalization(int(hidden_dim)),
            activation_function(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            normalization(int(hidden_dim)),
            activation_function(),
            torch.nn.Linear(int(hidden_dim), output_dim),
        ]
        if last_activation:
            modules.append(normalization(output_dim))
            modules.append(activation_function())
        return torch.nn.Sequential(*modules)
 
    def set_hardmax(self, hardmax):
        self.hardmax = hardmax
 
 
    def forward(self, x, edge_index):
        for _ in range(self.iterations):
            x = self.mp(x, edge_index)
            if self.batch_norm:
                x = self.bn(x)
            x = self.activation(x)
 
        if self.hardmax:
            return F.one_hot(torch.argmax(x, dim=1), x.size(1)).float()
        return x 
# Example usage:
# model = GNNCASimple()
# data contains node features x and edge indices edge_index
# out = model.steps(data, steps=10)