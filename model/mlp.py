#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, constant_weight=None):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()
        #self.layer_hidden = nn.Linear(dim_hidden, dim_out)

            # # initialize the weights to a specified, constant value
            # if (constant_weight is not None):
            #     for m in self.modules():
            #         if isinstance(m, nn.Linear):
            #             nn.init.constant_(m.weight, constant_weight)
            #             nn.init.constant_(m.bias, 0)
            # else:
            #     nn.init.xavier_normal_(self.layer_input.weight)
            # # By default, PyTorch uses Lecun initialization

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        #x = self.dropout(x)
        #x = self.relu(x)
        #x = self.layer_hidden(x)
        return x