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
        
        self.action_layer = layer_init(nn.Linear(features_output, action_dim), 1e-3)
        self.value_layer = layer_init(nn.Linear(features_output, 1), 1e-3)
        
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


class A2CAgent:
    def __init__(self, config):
        
        super(A2CAgent, self).__init__()
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
        action, _, _, _ = self.policy(state)
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
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(-1))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1))
            
            state = next_state
            
            if done.any():
                break
        
        self.state = state
        action, log_prob, entropy, value = self.policy(state)
        
#        actions.append(action)
#        log_probs.append(log_prob)
#        entropies.append(entropy)
#        values.append(value)
        
        advantages = []
        returns = []
        
        adv = torch.FloatTensor(np.zeros((num_envs, 1)))
        ret = value.detach()
        for i in reversed(range(len(rewards))):
            # returns = R + γV(s)
            ret = rewards[i] + config.gamma * ret * masks[i]
            returns.insert(0, ret.detach())
            
            # advantages = R + γV(s') - V(s)
            adv = ret - values[i].detach()
            advantages.insert(0, adv.detach())
            
            self.total_steps += config.num_envs
        
        log_prob = torch.cat(log_probs)
        value = torch.cat(values)
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        entropies = torch.cat(entropies)
        
        policy_loss = -(log_prob * advantages).mean()
        value_loss = (returns - value).pow(2).mean()
        entropy_loss = entropies.mean()
        
        loss = (policy_loss - config.ent_weight * entropy_loss + config.val_loss_weight * value_loss)
        
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
        self.optim.step()
        
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
                print('Episode {}\tEval score: {:.2f}\tPolicy loss: {:.3f}\tValue loss: {:.3f}\tEntropy loss: {:.3f}\tTotal loss: {:.3f}'\
                      .format(i_episode, 
                              score, 
                              policy_loss, 
                              value_loss, 
                              entropy_loss, 
                              loss))
        
        config.envs.close()
    
    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        score = 0
        
        self.policy.eval()
        
        while True:
            action = self.act([state])
            action = action.item()
            state, reward, done, _ = env.step(action)
            score += reward
            
            if done:
                break
        
        self.policy.train()
        return score