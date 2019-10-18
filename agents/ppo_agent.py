import numpy as np
import torch
import torch.nn.functional as F

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    """Proximal Policy Optimization Agent (https://arxiv.org/abs/1707.06347)
    Reuses lots of logic from A2C algorithm. Only difference is in the
    way it learns.
    
    Args:
        config (Config): holds the list of configuration items
    
    Attributes:
        name {str}: type/name of agent
            Default: 'ppo'
    
    """
    
    name = 'ppo'
    
    def __init__(self, config):
        super().__init__(config)
    
    def random_indices(self, len_states, shuffle=True):
        """Helper method, generator of random indices to sample mini batches"""
        
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
            # yields the left-overs
            yield indices[-rest:]
        
        
    
    def learn(self, returns):
        """Computes the losses of the policy and value according to PPO algorithm.
        Will sample ppo_epochs times a mini-batch of previously collected 
        trajectories. Then calculates the surrogate function and clip it to
        avoid swaying two far from the previous policy.
        
        
        Args:
            returns {List of torch.Tensor}: computed returns
        """
        
        ent_weight = self.config.ent_weight
        val_loss_weight = self.config.val_loss_weight
        ppo_clip = self.config.ppo_clip
        ppo_epochs = self.config.ppo_epochs
        
        log_probs = self.data.get('log_probs')
        values = self.data.get('values')
        states = self.data.get('states')
        actions = self.data.get('actions')
        
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions).detach()
        returns = torch.cat(returns)
        
        advantages = returns - values
        
        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        for _ in range(ppo_epochs):
            sampler = self.random_indices(len(states))
            
            for batch_idx in sampler:
                old_state = states[batch_idx, :]
                old_action = actions[batch_idx]
                old_log_prob = log_probs[batch_idx, :]
                old_advantage = advantages[batch_idx, :]
                old_return = returns[batch_idx, :]

#                (_, _, 
#                 new_log_prob, 
#                 entropy, 
#                 new_value) = self.policy(sampled_state, sampled_action)
                
                _, _, new_log_prob, entropy = self.policy(old_state, old_action)
                new_value = self.value(old_state)
                
                ratio = (new_log_prob - old_log_prob).exp()
                
                surrogate_ratio = ratio * old_advantage
                surrogate_clip = ratio.clamp(1.0 - ppo_clip, 
                                             1.0 + ppo_clip) * old_advantage
                
                surrogate = torch.min(surrogate_ratio, surrogate_clip)
                
                policy_loss = (-surrogate - ent_weight * entropy).mean()
                value_loss = val_loss_weight * (old_return - new_value).pow(2).mean()
                
                self.update(policy_loss, value_loss)
        
        return policy_loss, value_loss
    
    def summary(self):
        super().summary('PPO Agent')