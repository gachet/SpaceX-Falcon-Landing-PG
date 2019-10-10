import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import device

class Critic(nn.Module):
    def __init__(self, state_dim, activ):
        super(Critic, self).__init__()
        
        self.input = nn.Linear(state_dim, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.input(state)
        x = self.activ(x)
        
        x = self.hidden1(x)
        x = self.activ(x)
        
        value = self.output(x)
    
        return value