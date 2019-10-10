import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .device import device

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, activ):
        super(Actor, self).__init__()
        
        self.input = nn.Linear(state_dim, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_dim)
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state, action=None):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.input(state)
        x = self.activ(x)
        
        x = self.hidden1(x)
        x = self.activ(x)
        
        logits = self.output(x)
        
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy