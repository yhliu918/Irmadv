
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (v-defu@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init



class MLP(nn.Module):
    def __init__(self,hidden_dim=256):
        super(MLP, self).__init__()
        lin1 = nn.Linear(2 * 14 * 14, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 2)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
        out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out

def mnist_net():
    return MLP()