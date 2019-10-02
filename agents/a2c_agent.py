import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        feature_layers = []
#        batchnorm_layers = []
        
        for dim_in, dim_out in  zip(dims[:-1], dims[1:]):
            feature_layers.append(layer_init(nn.Linear(dim_in, dim_out)))
#            batchnorm_layers.append(nn.BatchNorm1d(dim_out))
        
        self.feature_layers = nn.ModuleList(feature_layers)
#        self.batchnorm_layers = nn.ModuleList(batchnorm_layers)
    
        self.activ = activ
    
        features_out = dims[-1]
        
        self.action_layer = layer_init(nn.Linear(features_out, action_dim))
        self.value_layer = layer_init(nn.Linear(features_out, 1))
        
        self.to(device)

    def forward(self, states):
        x = torch.FloatTensor(states)
        
        for linear in self.feature_layers:
#        for linear, batch_norm in zip(self.feature_layers, self.batchnorm_layers):
            x = linear(x)
#            x = batch_norm(x)
            x = self.activ(x)
            x = F.dropout(x, 0.5)
    
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
            x = F.dropout(x, 0.3)
    
        return x

class A2CAgent:
    def __init__(self, config):
        
        super(A2CAgent, self).__init__()
        self.config = config
        
#        self.policy = Policy(config.state_dim, 
#                             config.action_dim, 
#                             config.hidden_units, 
#                             config.activ)
        
        self.policy = Actor(config.state_dim, 
                            config.action_dim, 
                            config.hidden_actor, 
                            config.activ_actor)
        
        self.value = Critic(config.state_dim, 
                            config.hidden_critic, 
                            config.activ_critic)
        
#        self.optim = config.optim(self.policy.parameters(), lr=config.lr)
        
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
#            action, _, _, _ = self.policy(state)
            action, _, _ = self.policy(state)
        self.policy.train()
        
        return action
    
    def step(self):
        config = self.config
        envs = config.envs
        num_envs = envs.nenvs
        state = self.state
        
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        masks = []
                
        for _ in range(config.rollouts):
#            action, log_prob, entropy, value = self.policy(state)
            
            action, log_prob, entropy = self.policy(state)
            value = self.value(state)
            
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
#        _, _, _, next_value = self.policy(state)
        next_value = self.value(state)
        
        returns = []
        advantages = []
        
        ret = next_value.detach()
        adv = torch.zeros((num_envs, 1))
        for i in reversed(range(len(rewards))):
            # R + γV(s')
            ret = rewards[i] + (config.gamma ** i) * ret * masks[i]
            returns.insert(0, ret)
            
            # R + γV(s') - V(s)
            adv = ret - values[i].detach()
            advantages.insert(0, adv)
            
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        entropies = torch.cat(entropies)
        
        policy_loss = (-log_probs * advantages).mean()
        value_loss = nn.MSELoss()(returns, values)
        entropy_loss = entropies.mean()
        
#        policy_loss = (-log_probs * advantages).sum()
#        value_loss = (returns - values).pow(2).sum()
#        entropy_loss = entropies.sum()
        
#        loss = (policy_loss - \
#                config.ent_weight * entropy_loss + \
#                config.val_loss_weight * value_loss)
        loss = (policy_loss - config.ent_weight * entropy_loss)
        
#        self.optim.zero_grad()
#        loss.backward()
#        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
#        self.optim.step()
        
        self.optim_policy.zero_grad()
        self.optim_value.zero_grad()
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
        
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), config.grad_clip)
        
        self.optim_policy.step()
        self.optim_value.step()
        
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
                
                print('Episode {}\tEval score: {:.2f}\tPolicy loss: {:.2f}\tValue loss: {:.2f}\tEntropy loss: {:.2f}\tLoss: {:.2f}'\
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
        
        total_score = 0
        
        while True:
            action = self.act([state])
            action = action.item()
            state, reward, done, _ = env.step(action)
            
            total_score += reward
            
            if done:
                break
        
        return total_score
    
    def run_episode(self, debug=True):
        env = self.config.envs
        state = env.reset()
        
        total_score = 0
        
        env.render()
        
        while True:
            action = self.act([state])
            action = action.item()
            state, reward, done, _ = env.step(action)
            
            total_score += reward
            
            if debug:
                print('Reward: {:.2f}'.format(reward))
            
            env.render()
            
            if done:
                break
        
        if debug:
            print('Total reward: {:.2f}'.format(total_score))
                
        env.close()