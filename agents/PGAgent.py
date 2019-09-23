import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from .device import  device

class Policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, activ):
        
        super(Policy, self).__init__()
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
        self.activ = activ

    def forward(self, x):
        for layer in self.layers:
            x = self.activ(layer(x))
        return F.softmax(x, dim=1)

class PGAgent:
    def __init__(self, 
                 state_size=10, 
                 hidden_size=(32, 64, 128), 
                 action_size=7, 
                 activ=F.relu, 
                 optim=optim.Adam, 
                 lr=1e-3):
        
        super(PGAgent, self).__init__()
        self.policy = Policy(state_size, hidden_size, action_size, activ).to(device)
        self.optim = optim(self.policy.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def train(self, 
              env, 
              n_episodes=1000, 
              max_t=1000, 
              gamma=.99, 
              print_every=100):

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = env.reset()
            
            for t in range(max_t):
                action, log_prob = self.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                
                if done:
                    break
            
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([d*r for d, r in zip(discounts, rewards)])
            
            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()
            
            self.optim.zero_grad()
            policy_loss.backward()
            self.optim.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        env.close()
            
        return scores