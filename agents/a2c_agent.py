import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from .device import device

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_units,
                 activ):
        super(Actor, self).__init__()
        
        dims = (state_dim,) + hidden_units + (action_dim,)
        
        self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) \
                 for dim_in, dim_out \
                 in zip(dims[:-1], dims[1:])])
    
        self.activ = activ
        
        self.to(device)

    def forward(self, states):
        x = torch.FloatTensor(states).to(device)
        
        for layer in self.layers[:-1]:
            x = self.activ(layer(x))
        
        logits = self.layers[-1](x)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy

class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 hidden_units,
                 activ):
        super(Critic, self).__init__()
        
        dims = (state_dim,) + hidden_units + (1,)
        
        layers = [layer_init(nn.Linear(dim_in, dim_out)) \
                  for dim_in, dim_out \
                  in zip(dims[:-1], dims[1:])]
        
#         batch_norm = [nn.BatchNorm1d(dim_out) for dim_out in dims[1:]]
        
        self.layers = nn.ModuleList(layers)
#         self.batch_norm = nn.ModuleList(batch_norm)
    
        self.activ = activ
        
        self.to(device)

    def forward(self, states):
        x = torch.FloatTensor(states).to(device)
        
#         for layer, batch_norm in zip(self.layers, self.batch_norm):
        for layer in self.layers[:-1]:
            x = layer(x)
#             x = batch_norm(x)
            x = self.activ(x)
#             x = F.dropout(x, 0.5)
        
        value = self.layers[-1](x)
    
        return value

class A2CAgent:
    def __init__(self, config):
        
        super(A2CAgent, self).__init__()
        self.config = config

        self.policy = Actor(config.state_dim, 
                            config.action_dim, 
                            config.hidden_actor, 
                            config.activ_actor)
        
        self.value = Critic(config.state_dim, 
                            config.hidden_critic, 
                            config.activ_critic)
        
        self.optim_policy = config.optim_actor(self.policy.parameters(), 
                                               lr=config.lr_actor)
        self.optim_value = config.optim_critic(self.value.parameters(), 
                                               lr=config.lr_critic)
        
        self.reset()
    
    def reset(self):
        self.state = self.config.envs.reset()
        self.total_steps = 0
    
    def act(self, state):
        self.policy.eval()
        with torch.no_grad():
            action, _, _ = self.policy([state])
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
            action, log_prob, entropy = self.policy(state)
            value = self.value(state)
            
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
        next_value = self.value(state)

        values.append(next_value)
        
        returns = []
        advantages = []
        
        R = next_value.detach()
        adv = torch.zeros((num_envs, 1))
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = values[i].detach()
            
            if config.use_gae:
                value_next = values[i+1].detach()
                
                # δ = r + γV(s') - V(s)
                delta = reward + config.gamma * value_next * masks[i] - value
                
                # GAE = δ' + λδ
                adv = delta + config.lamda * config.gamma * adv * masks[i]
            else:
                # R = r + γV(s')
                R = reward + config.gamma * R * masks[i]
                
                # A(s, a) = r + γV(s') - V(s)
                adv = R - value
            
            advantages.insert(0, adv)
            returns.insert(0, adv + value)
            
        log_probs = torch.cat(log_probs)
        values = torch.cat(values[:-1])
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        entropies = torch.cat(entropies)
        
        policy_loss = (-log_probs * advantages - config.ent_weight * entropies).mean()
        value_loss = F.mse_loss(values, returns)
        
        self.optim_policy.zero_grad()
        policy_loss.backward()
#        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
        self.optim_policy.step()
        
        self.optim_value.zero_grad()
        value_loss.backward()
#        nn.utils.clip_grad_norm_(self.value.parameters(), config.grad_clip)
        self.optim_value.step()
        
        return done.all(), value_loss, policy_loss#, loss
    
    def train(self):
        config = self.config
        
        for i_episode in range(1, config.num_episodes+1):
            self.reset()   
            while self.total_steps <= config.max_steps:
                done, value_loss, policy_loss = self.step()
                
                if done:
                    break
            
            if i_episode % config.log_every == 0:
                score = self.eval_episode()
                print('Episode {}\tValue loss: {:.2f}\tPolicy loss: {:.2f}\tScore: {:.2f}'\
                      .format(i_episode, 
                              value_loss,
                              policy_loss, 
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