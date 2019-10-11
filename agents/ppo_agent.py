import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a2c_agent import A2CAgent
from .device import device

class PPOAgent(A2CAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def step(self):
        
        log_probs, entropies, values, rewards, masks, states, actions = \
            self.collect_data()
        
        returns = self.compute_return(values, rewards, masks)

        policy_loss, value_loss = self.learn(log_probs, 
                                             entropies, 
                                             values, 
                                             returns, 
                                             states, 
                                             actions)
        
        return value_loss, policy_loss
    
    def random_indices(self, len_states, shuffle=True):
        ppo_batch_size = self.config.ppo_batch_size
        indices = np.arange(len_states)
        
        if shuffle:
            indices = np.random.permutation(indices)
        
        len_total = len_states // ppo_batch_size * ppo_batch_size
        batches = indices[:len_total].reshape(-1, ppo_batch_size)
        
        for batch in batches:
            yield batch
        
        rest = len_states % ppo_batch_size
        if rest > 0:
            yield indices[-rest:]
        
        
    
    def learn(self, log_probs, entropies, values, returns, states, actions):
        ent_weight = self.config.ent_weight
        ppo_clip = self.config.ppo_clip
        ppo_epochs = self.config.ppo_epochs
        
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values[:-1]) # we need to remove the last value
        returns = torch.cat(returns)
        states = torch.cat(states)
        actions = torch.cat(actions)
        
        # A(s, a) = r + γV(s') - V(s)
        advantages = returns - values.detach()
        
        for _ in range(ppo_epochs):
            sampler = self.random_indices(len(states))
            
            for batch_idx in sampler:
                sampled_state = states[batch_idx, :]
                sampled_action = actions[batch_idx]
                sampled_log_prob = log_probs[batch_idx, :]
                sampled_advantage = advantages[batch_idx, :]
                sampled_return = returns[batch_idx, :]

                _, new_log_prob, entropy = self.policy(sampled_state, sampled_action)
                new_value = self.value(sampled_state)
                
                ratio = (new_log_prob - sampled_log_prob.detach()).exp()
                
                surrogate_ratio = ratio * sampled_advantage
                surrogate_clip = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * sampled_advantage
                min_surrogate = torch.min(surrogate_ratio, surrogate_clip)
                
                policy_loss = (-min_surrogate - ent_weight * entropy).mean()
                value_loss = F.mse_loss(new_value, sampled_return)
                
                self.update(policy_loss, value_loss)
        
        return policy_loss, value_loss