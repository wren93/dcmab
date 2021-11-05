import torch.nn as nn


class MCDropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(MCDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        self.dropout.train()
        return self.dropout(x)
        

class DCMABNet(nn.Module):
    def __init__(self, n_input=128, n_hidden=256, dropout_rate=0.2):
        super(DCMABNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(n_input, n_hidden))
        self.net.add_module('mcdp1', MCDropout(dropout_rate))
        self.net.add_module('relu1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(n_hidden, n_hidden))
        self.net.add_module('mcdp2', MCDropout(dropout_rate))
        self.net.add_module('relu2', nn.ReLU())
        self.net.add_module('fc3', nn.Linear(n_hidden, 1))

    def forward(self, x):
        return self.net(x)
