import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic
#from .policy import Policy
from .device import device
from .data import Data
from .utils import get_time_elapsed

class A2CAgent:
    """Advantage Actor Critic Agent
    
    Args:
        config (Config): holds the list of configuration items
    
    Attributes:
        name {str}: type/name of agent
            Default: 'a2c'
    
    """
    
    name = 'a2c'
    
    def __init__(self, config):
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        
        self.config = config
        self.data = Data()
        
#        self.policy = Policy(config.state_size, config.action_size)

        self.policy = Actor(config.state_size, 
                            config.action_size, 
                            config.hidden_actor,
                            config.activ_actor)
        
        self.value = Critic(config.state_size, 
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
        self.steps_done = 0
        self.done = False
        
    def act(self, state):
        """Will make it easier and efficient to get an action based on a state"""
        
        self.policy.eval()
        with torch.no_grad():
#            _, action, _, _, _ = self.policy(state)
            _, action, _, _ = self.policy(state)
        self.policy.train()
        
        return action
    
    def collect_data(self):
        """Will take care of collecting trajectories for learning purposes"""
        
        steps = self.config.steps
        envs = self.config.envs
        state = self.state
        
        self.data.clear()
                
        for _ in range(steps):
            state = torch.FloatTensor(state).to(device)
            
#            _, action, log_prob, entropy, value = self.policy(state)
            _, action, log_prob, entropy = self.policy(state)
            value = self.value(state)
            
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)
            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(device)
            
            # Stores trajectory, plus some extra information such as 
            # log_probs, entropies, values and masks (is terminal)
            self.data.add(states=state, 
                          actions=action, 
                          log_probs=log_prob, 
                          entropies=entropy, 
                          values=value, 
                          rewards=reward, 
                          masks=mask)
            
            state = next_state
            
            self.steps_done += 1
            
            if done.any():
                self.done = True
                break
        
        self.state = state
        
        # Let's estimate the next value
#        _, _, _, _, next_value = self.policy(state)
        next_value = self.value(state)
        
        return next_value
    
    def compute_return(self, next_value):
        """We offer the possibility of computing returns using
        Generalized Advantage Estimation algorithm, which will be
        parametrized by use_gae.
        
        See: https://arxiv.org/abs/1506.02438
        
        Args:
            next_value (torch.Tensor): next value estimated
        """
        
        num_agents = self.config.num_agents
        use_gae = self.config.use_gae
        gamma = self.config.gamma
        lamda = self.config.lamda
        
        values = self.data.get('values') + [next_value]
        rewards = self.data.get('rewards')
        masks = self.data.get('masks')
        
        returns = []
        R = next_value.detach()
        GAE = torch.zeros((num_agents, 1)).to(device)
        
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            
            if use_gae:
                value = values[i].detach()
                value_next = values[i+1].detach()
                
                # δ = r + γV(s') - V(s)
                delta = reward + gamma * value_next * masks[i] - value
                
                # GAE = δ' + λγδ
                GAE = delta + lamda * gamma * GAE * masks[i]
                
                returns.insert(0, GAE + value)
            else:
                # R = r + γV(s')
                R = reward + gamma * R * masks[i]
                
                returns.insert(0, R)
        
        return returns
    
    def update(self, policy_loss, value_loss):
        """Updates the weights by backpropagating the losses of the two
        (or combined) networks.
        
        Will also clip gradients according to grad_clip_actor/critic 
        configuration item,.
        
        Args:
            policy_loss (torch.FloatTensor)
            value_loss (torch.FloatTensor)
        """
        
#        grad_clip = self.config.grad_clip
        grad_clip_actor = self.config.grad_clip_actor
        grad_clip_critic = self.config.grad_clip_critic
        
#        loss = policy_loss + value_loss
#        
#        self.optim.zero_grad()
#        
#        los.backward()
#        
#        if grad_clip is not None:
#            nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
#        
#        self.optim.step()
        
        self.optim_policy.zero_grad()
        policy_loss.backward()
        
        if grad_clip_actor is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip_actor)
        
        self.optim_policy.step()
        
        self.optim_value.zero_grad()
        value_loss.backward()
        
        if grad_clip_critic is not None:
            nn.utils.clip_grad_norm_(self.value.parameters(), grad_clip_critic)
        
        self.optim_value.step()
    
    def learn(self, returns):
        """Computes the losses of the policy and value.
        We're using the advantage as a base line function.
        
        Args:
            returns {List of torch.Tensor}: computed returns
        """
        
        ent_weight = self.config.ent_weight
        val_loss_weight = self.config.val_loss_weight
        
        log_probs = self.data.get('log_probs')
        entropies = self.data.get('entropies')
        values = self.data.get('values')
        
        # List to torch.Tensor
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)
        returns = torch.cat(returns)
        
        # A(s, a) = r + γV(s') - V(s)
        advantages = returns - values.detach()
        
        # I noticed an improvement by normilizing the advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-3)
        
        policy_loss = (-log_probs * advantages - ent_weight * entropies).mean()
        value_loss = val_loss_weight * (returns - values).pow(2).mean()
        
        self.update(policy_loss, value_loss)
        
        return policy_loss, value_loss
    
    def step(self):
        """
        Performs a step in the A2C algorithm:
            1. Collect trajectories
            2. Compute returns/GAE
            3. Calculate losses and update
        """
        
        next_value = self.collect_data()
        returns = self.compute_return(next_value)
        policy_loss, value_loss = self.learn(returns)
        
        return policy_loss, value_loss
    
    def train(self):
        """Runs the training with evaluation and some debugging
        based on configuration items. Will also save checkpoints
        (policy and value weights) every time the best score is beaten
        """
        
        num_episodes = self.config.num_episodes
        times_solved = self.config.times_solved
        env_solved = self.config.env_solved
        envs = self.config.envs
        
        start = time.time()
        
        scores = []
        best_score = -np.inf
        
        for i_episode in range(1, num_episodes+1):
            self.reset()
            while True:
                policy_loss, value_loss = self.step()
                if self.done:
                    break
            
            score = self.eval_episode(1) # we evaluate only once
            scores.append(score)
            
            print('\rEpisode {}\tPolicy loss: {:.3f}\tValue loss: {:.3f}\tAvg Score: {:.3f}'\
                  .format(i_episode, 
                          policy_loss, 
                          value_loss, 
                          score), end='')
            
            if score > best_score:
                best_score = score
                print('\nBest score so far: {:.3f}'.format(best_score))
                
                torch.save(self.policy.state_dict(), '{}_actor_checkpoint.ph'.format(self.name))
                torch.save(self.value.state_dict(), '{}_critic_checkpoint.ph'.format(self.name))
                
            if score >= env_solved:
                # For speed reasons I'm gonna do a full evaluation after the env has been
                # solved the first time.
                
                print('\nRunning full evaluation...')
                
                # We now evaluate times_solved-1 (it's been already solved once), 
                # since the condition to consider the env solved is to reach the target 
                # reward at least an average of times_solved times consecutively
                avg_score = self.eval_episode(times_solved-1)
                
                if avg_score >= env_solved:
                    time_elapsed = get_time_elapsed(start)
                    
                    print('Environment solved {} times consecutively!'.format(times_solved))
                    print('Avg score: {:.3f}'.format(avg_score))
                    print('Time elapsed: {}'.format(time_elapsed))
                    break;
                else:
                    print('No success. Avg score: {:.3f}'.format(avg_score))
        
        envs.close()
        
        return scores
    
    def eval_episode(self, times_solved):
        """Evaluation method. 
        Will run times_solved times and avarage the total reward obtained
        
        Args:
            times_solved (int): how many times we run an episode
        """
        
        envs = self.config.envs
        
        total_reward = 0
        
        for _ in range(times_solved):
            state = envs.reset()
            while True:
                action = self.act(state)
                state, reward, done, _ = envs.step(action.cpu().numpy())
                
                avg_reward = (reward * (1 - done)).mean()
                total_reward += avg_reward
    
                if done.all():
                    break
                
        return total_reward / times_solved
    
    def summary(self, agent_name='A2C Agent'):
        print('{}:'.format(agent_name))
        print('==========')
        print('')
        print('Policy Network:')
        print('---------------')
        print(self.policy)
        print('')
        print('Value Network:')
        print('--------------')
        print(self.value)