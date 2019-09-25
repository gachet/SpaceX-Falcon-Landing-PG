import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

from .device import device

class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
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
        return probs, value

class ActorCriticAgent:
    def __init__(self, 
                 state_size, action_size,
                 optim=optim.Adam, 
                 lr=3e-4):
        
        super(ActorCriticAgent, self).__init__()
        self.policy = Policy(state_size, action_size, hidden_size=20).to(device)
        self.optim = optim(self.policy.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, value = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value, dist.entropy()
    
    def train(self, 
              env, 
              n_episodes=2000,
              max_t=1000, 
              gamma=.99, 
              print_every=100):

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes+1):
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropies = []
            state = env.reset()
            
            for t in range(max_t):
                action, log_prob, value, entropy = self.act(state)
                
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy.mean())
                
                next_state, reward, done, _ = env.step(action)
                
                rewards.append(reward)
                masks.append(1 - done)
                
                state = next_state
                
                if done:
                    break
            
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            discounts = [gamma**i for i in range(len(rewards)+1)]
            returns = [d*r*m for d, r, m in zip(discounts, rewards, masks)]
            
            log_probs = torch.cat(log_probs)
            returns = torch.FloatTensor(returns)
            values = torch.cat(values)
            entropy = sum(entropies)
            
            advantage = returns - values
            
            actor_loss  = -(log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        env.close()
            
        return scores