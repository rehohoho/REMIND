import math
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).contiguous().view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.contiguous().view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).contiguous().view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(AGCN, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.levels = nn.ModuleDict({
            '1': TCN_GCN_unit(3, 64, A, residual=False),
            '2': TCN_GCN_unit(64, 64, A),
            '3': TCN_GCN_unit(64, 64, A),
            '4': TCN_GCN_unit(64, 64, A),
            '5': TCN_GCN_unit(64, 128, A, stride=2),
            '6': TCN_GCN_unit(128, 128, A),
            '7': TCN_GCN_unit(128, 128, A),
            '8': TCN_GCN_unit(128, 256, A, stride=2),
            '9': TCN_GCN_unit(256, 256, A),
            '10': TCN_GCN_unit(256, 256, A),
        })

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (1, 3, 300, 25, 2) -> (1, 2, 25, 3, 300) -> (1, 150, 300)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) # -> (1, 2, 25, 3, 300) -> (2, 3, 300, 25)

        for _, level in self.levels.items():
            x = level(x)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)


class AGCN_ClassifyAfterLevel(AGCN):
    
    def __init__(self, level, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(AGCN_ClassifyAfterLevel, self).__init__(num_class, num_point, num_person, graph, graph_args, in_channels)

        for i in range(level + 1, len(self.levels) + 1):
            del self.levels[str(i)]


class AGCN_StartAtLevel(AGCN):

    def __init__(self, level, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(AGCN_StartAtLevel, self).__init__(num_class, num_point, num_person, graph, graph_args, in_channels)
        
        del self.data_bn
        for i in range(1, level):
            del self.levels[str(i)]
        self.num_person = num_person

    def forward(self, x):
        N, C, T, V = x.size()
        
        for _, level in self.levels.items():
            x = level(x)
        
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N//self.num_person, self.num_person, c_new, -1) #B, C, T, V -> #B//2, 2, C, T*V
        x = x.mean(3).mean(1) # B//2, C

        return self.fc(x)


if __name__ == '__main__':

    model = AGCN(graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    g = AGCN_ClassifyAfterLevel(9, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    f = AGCN_StartAtLevel(10, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    
    def get_name_to_module(model):
        name_to_module = {}
        for m in model.named_modules():
            name_to_module[m[0]] = m[1]
        return name_to_module

    print('\n\norig')
    print(list(get_name_to_module(model).keys()))
    print('\n\ng')
    print(list(get_name_to_module(g).keys())) # extract features from levels.8 or levels.8.relu
    print('\n\nf')
    print(list(get_name_to_module(f).keys()))
    