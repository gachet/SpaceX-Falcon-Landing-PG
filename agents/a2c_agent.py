import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .device import device

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Policy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_units,
                 activ):
        super(Policy, self).__init__()
        
        dims = (state_dim,) + hidden_units
        
        self.feature_layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) \
             for dim_in, dim_out \
             in zip(dims[:-1], dims[1:])])
    
        self.activ = activ
    
        features_output = dims[-1]
        
        self.action_layer = layer_init(nn.Linear(features_output, action_dim))
        self.value_layer = layer_init(nn.Linear(features_output, 1))
        
        self.to(device)

    def forward(self, states):
        x = torch.FloatTensor(states)
        
        for layer in self.feature_layers:
            x = self.activ(layer(x))
    
        logits = self.action_layer(x)
        values = self.value_layer(x)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy, values

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
        x = torch.FloatTensor(states)
        
        for layer in self.layers:
            x = self.activ(layer(x))
        
        dist = Categorical(logits=x)
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
        
        self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) \
                 for dim_in, dim_out \
                 in zip(dims[:-1], dims[1:])])
    
        self.activ = activ
        
        self.to(device)

    def forward(self, states):
        x = torch.FloatTensor(states)
        
        for layer in self.layers:
            x = self.activ(layer(x))
    
        return x

class A2CAgent:
    def __init__(self, config):
        
        super(A2CAgent, self).__init__()
        self.config = config
        
        self.policy = Policy(config.state_dim, 
                             config.action_dim, 
                             config.hidden_units, 
                             config.activ)
        
#        self.policy = Actor(config.state_dim, 
#                            config.action_dim, 
#                            config.hidden_units, 
#                            config.activ)
#        
#        self.value = Critic(config.state_dim, 
#                            config.hidden_units, 
#                            config.activ)
        
        self.optim = config.optim(self.policy.parameters(), lr=config.lr)
        
#        self.optim_policy = config.optim(self.policy.parameters(), lr=config.lr)
#        self.optim_value = config.optim(self.value.parameters(), lr=config.lr)
        
        self.reset()
    
    def reset(self):
        self.state = self.config.envs.reset()
        self.total_steps = 0
    
    def act(self, state):
        self.policy.eval()
        
        with torch.no_grad():
            action, _, _, _ = self.policy(state)
#            action, _, _ = self.policy(state)
        
        self.policy.train()
        
        return action
    
    def step(self):
        config = self.config
        envs = config.envs
        num_envs = len(envs.ps)
        state = self.state
        
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        masks = []
                
        for _ in range(config.rollout):
            action, log_prob, entropy, value = self.policy(state)
            
#            action, log_prob, entropy = self.policy(state)
#            value = self.value(state)
            
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            
            rewards.append(torch.FloatTensor(reward).unsqueeze(-1))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1))
            
            state = next_state
            
            self.total_steps += config.num_envs
            
            if done.any():
                break
        
        self.state = state
        action, log_prob, entropy, value = self.policy(state)
#        value = self.value(state)
        
#        advantages = []
        returns = []
        
#        adv = torch.FloatTensor(np.zeros((num_envs, 1)))
        ret = value.detach()
        for i in reversed(range(len(rewards))):
            # returns = R + γV(s)
            ret = rewards[i] + config.gamma * ret * masks[i]
            returns.insert(0, ret.detach())
            
            # advantages = R + γV(s') - V(s)
#            adv = ret - values[i].detach()
#            advantages.insert(0, adv.detach())
        
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        returns = torch.cat(returns)
#        advantages = torch.cat(advantages)
        entropies = torch.cat(entropies)
        
        # advantages = R + γV(s') - V(s)
        advantages = (returns - values)
        
        policy_loss = (-log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()
        
#        policy_loss = -(log_probs * advantages).sum()
#        value_loss = (returns - values).pow(2).sum()
#        entropy_loss = entropies.sum()
        
        loss = (policy_loss - config.ent_weight * entropy_loss + config.val_loss_weight * value_loss)
#        loss = (policy_loss - config.ent_weight * entropy_loss)
        
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
        self.optim.step()
        
#        self.optim_policy.zero_grad()
#        loss.backward()
#        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
#        self.optim_policy.step()
#        
#        self.optim_value.zero_grad()
#        value_loss.backward()
#        nn.utils.clip_grad_norm_(self.value.parameters(), config.grad_clip)
#        self.optim_value.step()
        
        return done.any(), policy_loss, value_loss, entropy_loss, loss
    
    def train(self):
        config = self.config
        
        for i_episode in range(1, config.num_episodes+1):
            self.reset()
            
            while self.total_steps <= config.max_steps:
                done, policy_loss, value_loss, entropy_loss, loss = self.step()
                
                if done:
                    break
            
            if i_episode % config.log_every == 0:
                score = self.eval_episode()
                
                print('Episode {}\tEval score: {:.2f}\tPolicy loss: {:.2f}\tValue loss: {:.2f}\tEntropy loss: {:.2f}\tTotal loss: {:.2f}'\
                      .format(i_episode, 
                              score, 
                              policy_loss, 
                              value_loss, 
                              entropy_loss, 
                              loss))
                
                if score >= config.solved_with:
                    print('Environment solved with {:.2f}!'.format(score))
                    break
        
        config.envs.close()
    
    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        
        score = 0
        
        while True:
            action = self.act([state])
            action = action.item()
            state, reward, done, _ = env.step(action)
            
            score += reward
            
            if done:
                break
        
        return score
    
    def run_episode(self):
        env = self.config.envs
        state = env.reset()
        
        env.render()
            
        while True:
            action = self.act([state])
            action = action.item()
            state, reward, done, _ = env.step(action)
            
            env.render()
            
            if done:
                break
        
        env.close()