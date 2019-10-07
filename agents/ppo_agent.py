import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

from .device import device

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, std=0.0):
        super(Policy, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

class PPOAgent:
    def __init__(self, config):
        
        super(PPOAgent, self).__init__()
        self.config = config

        self.policy = Policy(config.state_dim, 
                             config.action_dim, 
                             config.hidden_units, 
                             config.activ)
        
        self.optim = config.optim(self.policy.parameters(), lr=config.lr)
        
        self.reset()
    
    def reset(self):
        self.state = self.config.envs.reset()
        self.total_steps = 0
    
    def act(self, state):
        self.policy.eval()
        with torch.no_grad():
            action, _, _, _ = self.policy([state])
        self.policy.train()
        
        return action.item()
    
    def step(self):
        config = self.config
        envs = config.envs
        num_envs = config.num_envs
        state = self.state
        
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        masks = []
                
        for _ in range(config.steps):
            action, log_prob, entropy, value = self.policy(state)
            
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            rewards.append(torch.FloatTensor(reward).unsqueeze(-1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(device))
            
            state = next_state
            
            self.total_steps += num_envs
            
            if done.all():
                break
        
        self.state = state
        _, _, _, next_value = self.policy(state)

        values.append(next_value)
        
        returns = []
        advantages = []
        
        ret = next_value.detach()
        adv = torch.zeros((num_envs, 1))
        for i in reversed(range(len(rewards))):
            # G = R + γV(s')
            ret = rewards[i] + config.gamma * ret * masks[i]
            
            if config.use_gae:
                value_hat = values[i+1].detach()
                
                # δ = R + γV(s') - V(s)
                td_error = rewards[i] + config.gamma * value_hat * masks[i] - values[i].detach()
                
                # GAE = δ' + λδ
                adv = td_error + config.lamda * config.gamma * adv * masks[i]
            else:
                # A(s, a) = R + γV(s') - V(s)
                adv = ret - values[i].detach()
            
            returns.insert(0, ret)
            advantages.insert(0, adv)
            
        log_probs = torch.cat(log_probs)
        values = torch.cat(values[:-1])
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        entropies = torch.cat(entropies)
        
        policy_loss = (-log_probs * advantages - config.ent_weight * entropies).mean()
        value_loss = F.mse_loss(values, returns)
        
        loss = (policy_loss + config.val_loss_weight * value_loss)
        
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
        self.optim.step()
        
        return done.all(), value_loss, policy_loss, loss
    
    def train(self):
        config = self.config
        
        for i_episode in range(1, config.num_episodes+1):
            self.reset()   
            while self.total_steps <= config.max_steps:
                done, value_loss, policy_loss, loss = self.step()
                
                if done:
                    break
            
            if i_episode % config.log_every == 0:
                score = self.eval_episode()
                
                print('Episode {}\tValue loss: {:.2f}\tPolicy loss: {:.2f}\tLoss: {:.2f}\tScore: {:.2f}'\
                      .format(i_episode, 
                              value_loss,
                              policy_loss, 
                              loss, 
                              score))
                
                if score >= config.solved_with:
                    print('Environment solved with {:.2f}!'.format(score))
                    break
        
        config.envs.close()
    
    def eval_episode(self):
        env = self.config.eval_env
        render = self.config.render_eval
        state = env.reset()
        
        total_score = 0
        
        if render:
            env.render()
        
        while True:
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            
            if render:
                env.render()
            
            total_score += reward
            
            if done:
                break
        
        if render:
            env.close()
        
        return total_score
    
    def run_episode(self, debug=True):
        env = self.config.envs
        state = env.reset()
        
        total_score = 0
        
        env.render()
        
        while True:
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            
            env.render()
            
            total_score += reward
            
            if debug:
                print('Reward: {:.2f}'.format(reward))

            if done:
                break
        
        if debug:
            print('Total reward: {:.2f}'.format(total_score))
                
        env.close()