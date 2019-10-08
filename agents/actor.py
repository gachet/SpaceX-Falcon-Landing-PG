import torch
import torch.nn as nn
from torch.distributions import Categorical

from .device import device

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, activ):
        super(Actor, self).__init__()
        
        self.input = nn.Linear(state_dim, 32)
        self.hidden1 = nn.Linear(32, 32)
        self.output = nn.Linear(32, action_dim)
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state):
        state = torch.FloatTensor(state).to(device)
        
        x = self.input(state)
        x = self.activ(x)
        
        x = self.hidden1(x)
        x = self.activ(x)
        
        logits = self.output(x)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy