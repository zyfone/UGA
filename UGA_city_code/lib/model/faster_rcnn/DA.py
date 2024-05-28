from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function, Variable

import torch.nn

class GRLayer(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)

    
class _InstanceDA_channel(nn.Module):
    def __init__(self, in_channle=4096):
        super(_InstanceDA_channel, self).__init__()
        self.fc_1_inst = nn.Linear(in_channle, 256)
        self.fc_2_inst = nn.Linear(256, 128)
        self.fc_3_inst = nn.Linear(128, 64)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.fc_1_inst(x))
        x = self.relu((self.fc_2_inst(x)))
        x = self.relu(self.bn2(self.fc_3_inst(x)))
        return x