import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .device import device

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # TODO
        
        self.to(device)

    def forward(self, state, action=None):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        # TODO
        
#        return mean, action, log_prob, entropy, value