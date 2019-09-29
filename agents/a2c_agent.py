import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from .device import device

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activ=F.tanh):
        
        super(Actor, self).__init__()
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
        self.activ = activ

    def forward(self, x):
        for layer in self.layers:
            x = self.activ(layer(x))
        return F.softmax(x, dim=1)

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, activ=F.tanh):
        
        super(Critic, self).__init__()
        
        dims = (state_size,) + hidden_size + (1,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
        self.activ = activ

    def forward(self, x):
        for layer in self.layers:
            x = self.activ(layer(x))
        return x


class A2CAgent:
    def __init__(self, state_size, action_size):
        
        super(A2CAgent, self).__init__()
        
        self.policy = Actor(state_size, 
                            action_size, 
                            hidden_size=(64,))
        
        self.value = Critic(state_size, 
                            hidden_size=(64,))
        
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_optim = optim.Adam(self.value.parameters(), lr=1e-3)
    
    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        
        probs = self.policy(state)
        value = self.value(state)
        
        dist  = Categorical(probs)
        action = dist.sample()
        
        return action, value, dist
    
    def train(self, 
              envs, 
              n_episodes=2000,
              max_steps=1000, 
              steps=5,
              gamma=.99, 
              coef_ent=.01, 
              coef_val=.25, 
              clip_val=.5,
              print_every=100,
              solved=1.0):

        num_envs = len(envs.ps)
        scores_deque = deque(maxlen=100)
        
        for i_episode in range(1, n_episodes+1):
            
            scores = 0
            state = envs.reset()
            
            for t in range(max_steps):
                
                log_probs = []
                values = []
                entropies = []
                rewards = []
                masks = []
                
                # Collect n steps of data
                for s in range(steps):
                    action, value, dist = self.act(state)
                    
                    log_probs.append(dist.log_prob(action).unsqueeze(-1))
                    values.append(value)
                    entropies.append(dist.entropy().unsqueeze(-1))
                    
                    next_state, reward, done, _ = envs.step(action.cpu().numpy())
                    
                    scores += reward.mean()
                    
                    rewards.append(torch.FloatTensor(reward).unsqueeze(-1).to(device))
                    masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(device))
                    
                    state = next_state
                    
                    if done.any():
                        break
            
                next_state = torch.FloatTensor(next_state).to(device)
                next_value = self.value(next_state)
                
                # return = R + γV(s)
                returns = next_value.detach()
                for i in reversed(range(len(rewards))):
                    returns = rewards[i] + gamma * returns * masks[i]
                
                log_probs = torch.cat(log_probs).view(num_envs, -1)
                values = torch.cat(values).view(num_envs, -1)
                entropies = torch.cat(entropies).view(num_envs, -1)
                
                # advantage = R + γV(s') - V(s)
                advantage = returns - values.detach()
                
                # A2C losses
                policy_loss = -(log_probs * advantage).mean()
                value_loss = (returns - values).pow(2).mean()
                policy_ent = entropies.mean()
                
#                loss = policy_loss + (coef_val * value_loss.detach()) - (coef_ent * policy_ent)
                loss = policy_loss - (coef_ent * policy_ent)
                
                self.policy_optim.zero_grad()
                loss.backward()
#                nn.utils.clip_grad_norm_(self.policy.parameters(), clip_val)
                self.policy_optim.step()
                
                self.value_optim.zero_grad()
                value_loss.backward()
#                nn.utils.clip_grad_norm_(self.value.parameters(), clip_val)
                self.value_optim.step()
            
                if done.any():
                    break
            
            scores_deque.append(scores)
            score_mean = np.mean(scores_deque)
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.3f}\tPolicy loss: {:.3f}\tValue loss: {:.3f}\tEntropy: {:.3f}\tTotal loss: {:.3f}'\
                      .format(i_episode, 
                              score_mean, 
                              policy_loss, 
                              value_loss, 
                              policy_ent,
                              loss))
            
            if score_mean >= solved:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, score_mean))
                break
        
        envs.close()
            
        return scores