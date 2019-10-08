import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic
from .device import device

class A2CAgent:
    def __init__(self, config):
        
        super(A2CAgent, self).__init__()
        self.config = config

        self.policy = Actor(config.state_dim, 
                            config.action_dim, 
                            config.activ_actor)
        
        self.value = Critic(config.state_dim, 
                            config.activ_critic)
        
        self.optim_policy = config.optim_actor(self.policy.parameters(), 
                                               lr=config.lr_actor)
        self.optim_value = config.optim_critic(self.value.parameters(), 
                                               lr=config.lr_critic)
        
        self.reset()
    
    def reset(self):
        self.state = self.config.envs.reset()
        self.total_steps = 0
        self.all_done = False
    
    def act(self, state):
        self.policy.eval()
        with torch.no_grad():
            action, _, _ = self.policy([state])
        self.policy.train()
        
        return action.item()
    
    def collect_data(self):
        steps = self.config.steps
        envs = self.config.envs
        num_envs = self.config.num_envs
        state = self.state
        
        log_probs = []
        values = []
        entropies = []
        rewards = []
        masks = []
                
        for _ in range(steps):
            action, log_prob, entropy = self.policy(state)
            value = self.value(state)
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            rewards.append(torch.FloatTensor(reward).unsqueeze(-1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(device))
            
            state = next_state
            
            self.total_steps += num_envs
            
            if done.all():
                self.all_done = True
                break
        
        self.state = state
        next_value = self.value(state)
        values.append(next_value)
        
        return log_probs, values, entropies, rewards, masks
    
    def compute_return(self, values, rewards, masks):
        num_envs = self.config.num_envs
        use_gae = self.config.use_gae
        gamma = self.config.gamma
        lamda = self.config.lamda
        
        next_value = values[-1]
        returns = []
        
        R = next_value.detach()
        GAE = torch.zeros((num_envs, 1))
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            value = values[i].detach()
            
            if use_gae:
                value_next = values[i+1].detach()
                
                # δ = r + γV(s') - V(s)
                delta = reward + gamma * value_next * masks[i] - value
                
                # GAE = δ' + λδ
                GAE = delta + lamda * gamma * GAE * masks[i]
                
                returns.insert(0, GAE + value)
            else:
                # R = r + γV(s')
                R = reward + gamma * R * masks[i]
                
                returns.insert(0, R)
        
        return returns
    
    def learn(self, log_probs, entropies, values, returns):
        ent_weight = self.config.ent_weight
        grad_clip = self.config.grad_clip
        
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values[:-1])
        returns = torch.cat(returns)
        
        # A(s, a) = r + γV(s') - V(s)
        advantages = returns - values.detach()
        
        policy_loss = (-log_probs * advantages - ent_weight * entropies).mean()
        value_loss = F.mse_loss(values, returns)
        
        self.optim_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        self.optim_policy.step()
        
        self.optim_value.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), grad_clip)
        self.optim_value.step()
        
        return policy_loss, value_loss
    
    def step(self):
        
        log_probs, values, entropies, rewards, masks = self.collect_data()
        
        returns = self.compute_return(values, rewards, masks)

        policy_loss, value_loss = self.learn(log_probs, 
                                             entropies, 
                                             values, 
                                             returns)
        
        return value_loss, policy_loss
    
    def train(self):
        num_episodes = self.config.num_episodes
        max_steps = self.config.max_steps
        log_every = self.config.log_every
        env_solved = self.config.env_solved
        envs = self.config.envs
        
        for i_episode in range(1, num_episodes+1):
            self.reset()
            
            while self.total_steps <= max_steps:
                value_loss, policy_loss = self.step()
                
                if self.all_done:
                    break
            
            if i_episode % log_every == 0:
                score = self.eval_episode()
                
                print('Episode {}\tValue loss: {:.3f}\tPolicy loss: {:.3f}\tScore: {:.2f}'\
                      .format(i_episode, 
                              value_loss,
                              policy_loss, 
                              score))
                
                if score >= env_solved:
                    print('Environment solved with {:.2f}!'.format(score))
                    break
        
        envs.close()
    
    def eval_episode(self):
        env = self.config.eval_env
        render = self.config.render_eval
        num_evals = self.config.num_evals
        
        total_score = 0
        
        for i in range(num_evals):
            state = env.reset()
            
            is_last = i == num_evals - 1
            
            if render and is_last:
                env.render()
            
            while True:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                
                if render and is_last:
                    env.render()
                
                total_score += reward
                
                if done:
                    break
            
            if render and is_last:
                env.close()
        
        return total_score / num_evals
    
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