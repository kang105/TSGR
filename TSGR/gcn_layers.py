import math

import torch
from torch.nn import init, Dropout, LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation, dropout, withbn=True, withloop=True, bias=True,
                 res=True):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res
        self.dropout = Dropout(dropout)

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            # self.bn = torch.nn.BatchNorm1d(out_features)
            # self.bn.cuda()
            self.bn = LayerNorm(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.self_weight is not None:
        #     stdv = 1. / math.sqrt(self.self_weight.size(1))
        #     self.self_weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        # glorot(self.weight)
        # if self.self_weight is not None:
        #     glorot(self.self_weight)
        # zeros(self.bias)


    def forward(self, input, adj):
        d_matrix = torch.diag(sum(adj, 1)**(-1/2))
        adj_ = torch.mm(torch.mm(d_matrix, adj), d_matrix)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        # if self.bn is not None:
        #     output = self.bn(output)
            # output = self.bn(torch.reshape(output,(int(output.shape[0] / 148), 148, output.shape[-1])).permute(1, 0, 2))
            # output = output.permute(1, 0, 2)
            # output = torch.reshape(output, (output.shape[0] * output.shape[1], output.shape[-1]))

        # Res
        if self.res:
            return self.bn(self.dropout(self.sigma(output)) + input)
        else:
            return self.bn(self.dropout(self.sigma(output)))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphBaseBlock(Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN
    """

    def __init__(self, feature_size, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=0.1,
                 aggrmethod="nores", dense=False):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.feature_size = feature_size
        self.__makehidden()


    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            layer = GraphConvolutionBS(in_features=self.feature_size[i], out_features=self.feature_size[i + 1], activation=self.activation, withbn=self.withbn, withloop=self.withloop, bias=True,
                 dropout=self.dropout)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            # x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

