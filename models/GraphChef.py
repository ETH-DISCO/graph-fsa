"""
Modified DT+GNN model from DT+GNN: A Fully Explainable Graph Neural Network using Decision Trees

Code: https://openreview.net/forum?id=9IlzJa5cAv

(we added recursive components)
"""

import torch
from torch import nn
import numpy as np
from torch.nn import Module, ModuleList, Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# Set manual seed for reproducibility
torch.manual_seed(0)


def gumbel_softmax(logits, tau=1.0, beta=1.0, hard=False, dim=-1):
    noise = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
    gumbels = noise.exponential_().log()
    gumbels = logits + gumbels * beta
    gumbels = gumbels / tau
    m = torch.nn.Softmax(dim)
    y_soft = m(gumbels)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        zeroes = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard = zeroes.scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret



class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        y_soft = input.softmax(dim=-1)
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(input).scatter_(-1, index, 1.0)
        ctx.save_for_backward(y_soft, y_hard)
        return y_hard, y_soft

    @staticmethod
    def backward(ctx, grad_output, grad_out_y_soft):
        y_soft, y_hard = ctx.saved_tensors
        grad = grad_output * y_hard + grad_out_y_soft * y_soft
        return grad


def argmax(x):
    return ArgMax.apply(x)[0]


class MLP(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        self.bns = ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = F.relu(self.bns[i](lin(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


class LinearSoftmax(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gumbel=True,
        temperature=1.0,
        use_batch_norm=True,
    ):
        super(LinearSoftmax, self).__init__()
        self.__name__ = "LinearSoftmax"
        self.lin1 = Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.argmax = False
        self.gumbel = gumbel
        self.softmax_temp = temperature
        self.beta = 0.0
        self.alpha = 1.0
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.lin1(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.argmax:
            x_d = argmax(x)
        elif self.gumbel:
            x_d = gumbel_softmax(x, hard=True, tau=self.softmax_temp, beta=self.beta)
        else:
            x_d = softmax(x / self.softmax_temp, dim=-1)

        if np.random.random() > self.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x

class MLPSoftmax(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gumbel=True,
        temperature=1.0,
        hidden_units=16,
        dropout=0.0,
    ):
        super(MLPSoftmax, self).__init__()
        self.__name__ = "LinearSoftmax"
        self.mlp = MLP(in_channels, hidden_units, out_channels, 2, dropout)
        self.gumbel = gumbel
        self.argmax = False
        self.beta = 0.0
        self.alpha = 1.0
        self.softmax_temp = temperature

    def forward(self, x):
        x = self.mlp(x)
        if self.argmax:
            x_d = argmax(x)
        elif self.gumbel:
            x_d = gumbel_softmax(x, hard=True, tau=self.softmax_temp, beta=self.beta)
        else:
            x_d = softmax(x / self.softmax_temp, dim=-1)

        if np.random.random() > self.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x

class InputLayer(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        softmax_temp,
        gumbel=True,
        network="linear",
        hidden_units=16,
    ):
        super(InputLayer, self).__init__()
        self.__name__ = "FirstLayer"
        if network == "mlp":
            self.lin1 = MLP(in_channels, hidden_units, out_channels, 2, 0.0)
        else:
            self.lin1 = Linear(in_channels, out_channels)
        self.gumbel = gumbel
        self.argmax = False
        self.beta = 0.0
        self.alpha = 1.0
        self.softmax_temp = softmax_temp

    def forward(self, x):
        if x is not torch.FloatTensor:
            x = x.float()
        x = self.lin1(x)
        if self.argmax:
            x_d = argmax(x)
        elif self.gumbel:
            x_d = gumbel_softmax(x, hard=True, tau=self.softmax_temp, beta=self.beta)
        else:
            x_d = softmax(x / self.softmax_temp, dim=-1)

        if np.random.random() > self.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x

class PoolingLayer(Module):
    def __init__(
        self, in_channels, out_channels, network="linear", hidden_units=16, dropout=0.0
    ):

        print("in_channels", in_channels)
        super(PoolingLayer, self).__init__()
        self.__name__ = "PoolingLayer"
        self.lin2 = Linear(in_channels, out_channels)
        if network == "mlp":
            self.lin2 = MLP(in_channels, hidden_units, out_channels, 2, dropout)

    def forward(self, x):
        x = self.lin2(x)
        return x
        # return log_softmax(x, dim=-1)

class MLPLayer(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        network="mlp",
        hidden_units=16,
        dropout=0.0,
        num_layers=2,
    ):
        super(MLPLayer, self).__init__()
        self.__name__ = "MLPLayer"
        self.lin2 = Linear(in_channels, out_channels)
        if network == "mlp":
            self.lin2 = MLP(
                in_channels, hidden_units, out_channels, num_layers, dropout
            )

    def forward(self, x):
        x = self.lin2(x)
        return x
        # return log_softmax(x, dim=-1)


class StoneAgeGNNLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        bounding_parameter,
        hidden_units=16,
        dropout=0.0,
        gumbel=False,
        temperature=1.0,
        index=0,
        network="linear",
        use_batch_norm=True,
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)
        if network == "mlp":
            self.linear_softmax = MLPSoftmax(
                in_channels,
                out_channels,
                gumbel,
                temperature,
                hidden_units=hidden_units,
                dropout=dropout,
            )
        else:
            self.linear_softmax = LinearSoftmax(
                in_channels,
                out_channels,
                gumbel,
                temperature,
                use_batch_norm=use_batch_norm,
            )
        self.bounding_parameter = bounding_parameter

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        return torch.clamp(message_sums, min=0, max=self.bounding_parameter)

    def update(self, inputs, x):
        combined = torch.cat((inputs, x), 1)
        return self.linear_softmax(combined)


class StoneAgeGNNEdgeLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        bounding_parameter,
        hidden_units=16,
        dropout=0.0,
        gumbel=False,
        temperature=1.0,
        index=0,
        network="mlp",
        use_batch_norm=True,
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)

        if network == "mlp":
            self.linear_softmax = MLPSoftmax(
                in_channels // 2,
                out_channels,
                gumbel,
                temperature,
                hidden_units=hidden_units,
                dropout=dropout,
            )

            self.linear_softmax_edge = MLPSoftmax(
                in_channels,
                out_channels,
                gumbel,
                temperature,
                hidden_units=hidden_units,
                dropout=dropout,
            )
        else:
            self.linear_softmax = LinearSoftmax(
                in_channels // 2,
                out_channels,
                gumbel,
                temperature,
                use_batch_norm=use_batch_norm,
            )

            self.linear_softmax_edge = LinearSoftmax(
                in_channels,
                out_channels,
                gumbel,
                temperature,
                use_batch_norm=use_batch_norm,
            )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bounding_parameter = bounding_parameter

    def forward(self, x, edge_index):
        out2 = self.propagate(edge_index, x=x)
        out = self.linear_softmax( (1 + self.eps) * x + out2)
        return out 

    def message(self, x_j, x_i):
        concatenated = torch.cat((x_j, x_i), dim=1)
        out = self.linear_softmax_edge(concatenated)
        return out

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        aggr = torch.clamp(message_sums, min=0, max=self.bounding_parameter)
        return aggr

    def update(self, inputs, x):
        return inputs


class RecurrentGraphChef(Module):
    def __init__(self, in_channels, out_channels, bounding_parameter, state_size, num_layers=1, 
                 gumbel=False, softmax_temp=1.0, network="linear", use_pooling=True, 
                 use_batch_norm=True, hidden_units=16, dropout=0.0, iterations=12, 
                 skip_input_inclusive=False, use_edge_mlp=False):
        super().__init__()

        self.input = InputLayer(
            in_channels,
            state_size,
            gumbel=gumbel,
            softmax_temp=softmax_temp,
            network=network,
            hidden_units=hidden_units,
        )
        self.initial_gumbel = gumbel

        self.output = PoolingLayer(
            state_size,
            out_channels,
            network="mlp",
            hidden_units=hidden_units,
            dropout=dropout,
        )

        self.skip_input = MLPLayer(
            in_channels + state_size,
            state_size,
            num_layers=2,
            network=network,
            hidden_units=hidden_units,
            dropout=dropout,
        )

        self.skip_input_inclusive = skip_input_inclusive
        self.stone_age = ModuleList()
        self.num_layers = num_layers
        self.use_pooling = use_pooling
        self.out_channels = out_channels
        self.state_size = state_size
        self.dropout = dropout

        for i in range(num_layers):

            input_size = state_size * 2

            if i == 0 and self.skip_input_inclusive:
                input_size += in_channels * 2

            if use_edge_mlp:
                self.stone_age.append(
                    StoneAgeGNNEdgeLayer(
                        input_size,
                        state_size,
                        bounding_parameter=bounding_parameter,
                        gumbel=gumbel,
                        temperature=softmax_temp,
                        index=i,
                        network=network,
                        use_batch_norm=use_batch_norm,
                        hidden_units=hidden_units,
                        dropout=dropout,
                    )
                )

            else:

                self.stone_age.append(
                    StoneAgeGNNLayer(
                        input_size,
                        state_size,
                        bounding_parameter=bounding_parameter,
                        gumbel=gumbel,
                        temperature=softmax_temp,
                        index=i,
                        network=network,
                        use_batch_norm=use_batch_norm,
                        hidden_units=hidden_units,
                        dropout=dropout,
                    )
                )

        self.iterations = iterations

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_gumbel(self, gumbel):
        self.input.gumbel = gumbel
        for i in range(self.num_layers):
            layer = self.stone_age[i]
            layer.linear_softmax.gumbel = False if i == self.num_layers - 1 else gumbel

    def set_argmax(self, enabled):
        self.input.argmax = enabled
        for i in range(self.num_layers):
            layer = self.stone_age[i]
            layer.linear_softmax.argmax = enabled

    def set_hardmax(self, hardmax):
        self.set_argmax(hardmax)

    def set_softmax_temp(self, temperature):
        self.input.softmax_temp = temperature
        for i in range(self.num_layers):
            layer = self.stone_age[i]
            layer.linear_softmax.softmax_temp = temperature

    def set_beta(self, beta):
        self.input.beta = beta
        for i in range(self.num_layers):
            layer = self.stone_age[i]
            layer.linear_softmax.beta = beta

    def set_alpha(self, alpha):
        self.input.alpha = alpha
        for i in range(self.num_layers):
            layer = self.stone_age[i]
            layer.linear_softmax.alpha = alpha

    def forward(self, x, edge_index, batch=None, return_layers=False):

        x_orig = x
        x = self.input(x)
        xs = [x]
        layers = []

        if return_layers:
            layers.append(x)

        for i in range(self.iterations):
            
            x_plus_orig = torch.cat([x, x_orig], dim=1)
            first = True

            for layer in self.stone_age:

                if first and self.skip_input_inclusive:
                    first = False
                    x = x_plus_orig

                x = layer(x, edge_index)

                xs.append(x)

                if return_layers:
                    layers.append(x)
        x = self.output(x)

        if return_layers:
            return x, layers

        return x
