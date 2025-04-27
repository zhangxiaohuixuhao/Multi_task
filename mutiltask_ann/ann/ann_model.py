import torch
import torch.nn as nn
import numpy as np
import utility.utils as utils


class model(nn.Module):
    def __init__(self,
                 hid_dim,
                 hid_num,
                 num_class,
                 batchsize,
                 num_layer,
                 device=torch.device('cpu')):
        super(model, self).__init__()
        self.hid_dim = hid_dim
        self.batchsize = batchsize
        self.num_layer = num_layer
        self.num_class = num_class
        self.hid_num = hid_num
        # self.fc1 = nn.Linear(int(self.hid_dim), int(self.hid_num))
        # self.fc_out = nn.Linear(int(self.hid_num), num_class)
        self.fc_out = nn.Linear(int(self.hid_dim), num_class)
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        # using bn
        # x = (x - x.mean()) / x.std()
        # x = self.fc1(x)
        x = (x - x.mean()) / x.std()
        x = self.fc_out(x)
        return x
