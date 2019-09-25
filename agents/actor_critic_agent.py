import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from .device import device


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Policy, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value


class ActorCriticAgent:
    def __init__(self, 
                 state_size, action_size,
                 hidden_size=256,
                 optim=optim.Adam, 
                 lr=3e-4):
        
        super(ActorCriticAgent, self).__init__()
        self.policy = Policy(state_size, 
                             action_size, 
                             hidden_size=hidden_size).to(device)
        self.optim = optim(self.policy.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        dist, value = self.policy(state)
        action = dist.sample()
        return action, value, dist
    
    def train(self, 
              envs, 
              n_episodes=2000,
              max_t=1000, 
              gamma=.99, 
              print_every=50):

        scores_deque = deque(maxlen=100)
        
        for i_episode in range(1, n_episodes+1):
            
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            scores = 0
            
            state = envs.reset()
            
            for t in range(max_t):
                action, value, dist = self.act(state)
                
                log_probs.append(dist.log_prob(action))
                values.append(value)
                entropy += dist.entropy().mean()
                
                next_state, reward, done, _ = envs.step(action.cpu().numpy())
                
                scores += reward.mean()
                
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                state = next_state
            
            scores_deque.append(scores)
            
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.policy(next_state)
            
            R = next_value
            returns = []
            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma * R * masks[i]
                returns.insert(0, R)
            
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns)
            values = torch.cat(values)
            
            advantage = returns - values
            
            actor_loss  = -(log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            
            loss = actor_loss + critic_loss - 0.001 * entropy
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        envs.close()
            
        return scores