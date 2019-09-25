import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from .device import device

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
                 state_size, action_size,
                 hidden_size=(32, 64, 128), 
                 activ=F.relu, 
                 optim=optim.Adam, 
                 lr=1e-3):
        
        super(PGAgent, self).__init__()
        self.policy = Policy(state_size, hidden_size, action_size, activ).to(device)
        self.optim = optim(self.policy.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).cpu()
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def train(self, 
              env, 
              n_episodes=1000, 
              max_t=1000, 
              gamma=.99, 
              print_every=100):

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes+1):
            log_probs = []
            entropies = []
            rewards = []
            state = env.reset()
            
            for t in range(max_t):
                action, log_prob, entropy = self.act(state)
                log_probs.append(log_prob)
                entropies.append(entropy)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                
                if done:
                    break
            
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            # Normalizing the rewards:
#            rewards = np.array(rewards)
#            
#            mean = rewards.mean()
#            std = rewards.std() + 1e-5
#            
#            rewards = ((rewards - mean) / std).tolist()
            
            discounts = [gamma**r for r in range(len(rewards)+1)]
            G = [d*r for d, r in zip(discounts, rewards)]
            
            b = np.array(G).mean() # Baseline
#            b = 0
            
            losses = []
            for log_prob, entropy, g in zip(log_probs, entropies, G):
                losses.append(-log_prob * (g - b) - (0.01 * entropy))
                
#            loss = torch.cat(losses).sum() - (0.01 * entropy.mean())
            loss = torch.cat(losses).sum()
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        env.close()
            
        return scores