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

class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Policy, self).__init__()
#        self.affine = nn.Linear(input_size, hidden_size)
#        self.action = nn.Linear(hidden_size, output_size)
#        self.value = nn.Linear(hidden_size, 1)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
#            layer_init(nn.Linear(hidden_size, hidden_size)),
#            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1))
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
#            layer_init(nn.Linear(hidden_size, hidden_size)),
#            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, output_size)),
            nn.Softmax(dim=1),
        )
        
    def forward(self, state):
#        state = F.relu(self.affine(state))
#        probs = F.softmax(self.action(state), dim=1)
#        value = self.value(state)
        
        value = self.critic(state)
        probs = self.actor(state)
        
        return probs, value


class ActorCriticAgent:
    def __init__(self, 
                 state_size, action_size,
                 hidden_size=64,
                 optim=optim.Adam, 
                 lr=1e-3):
        
        super(ActorCriticAgent, self).__init__()
        
        self.policy = Policy(state_size, 
                             action_size, 
                             hidden_size=hidden_size).to(device)
        
        self.optim = optim(self.policy.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        probs, value = self.policy(state)
        dist  = Categorical(probs)
        action = dist.sample()
        return action, value, dist
    
    def train(self, 
              envs, 
              n_episodes=2000,
              max_t=20, 
              gamma=.99, 
              coef_ent=.01, 
              coef_val=.25, 
              print_every=100):

        num_envs = len(envs.ps)
        scores_deque = deque(maxlen=100)
        
        for i_episode in range(1, n_episodes+1):
            
            log_probs = []
            values = []
            entropies = []
            rewards = []
            masks = []
            scores = 0
            
            state = envs.reset()
            
            for t in range(max_t):
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
            
            scores_deque.append(scores)
            
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.policy(next_state)
            
            # return = R + yV(s)
            returns = next_value.detach()
            for i in reversed(range(len(rewards))):
                returns = rewards[i] + gamma * returns * masks[i]
            
            log_probs = torch.cat(log_probs).view(num_envs, -1)
            values = torch.cat(values).view(num_envs, -1)
            entropies = torch.cat(entropies).view(num_envs, -1)
            
            # advantage = R + yV(s') - V(s)
            advantage = returns - values.detach()
            
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = (returns - values).pow(2).mean()
            entropy_loss = entropies.mean()
            
            loss = actor_loss + (coef_val * critic_loss) - (coef_ent * entropy_loss)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.3f}\tActor loss: {:.3f}\tCritic loss: {:.3f}\tEntropy loss: {:.3f}\tTotal loss: {:.3f}'\
                      .format(i_episode, 
                              np.mean(scores_deque), 
                              actor_loss, 
                              critic_loss, 
                              entropy_loss,
                              loss))
        
        envs.close()
            
        return scores