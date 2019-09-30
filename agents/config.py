import torch.nn.functional as F
from torch.optim import RMSprop

class Config:
    num_envs = 5
    env_name = 'CartPole-v0'
    envs = []
    eval_env = None
    num_episodes = 2000
    rollout = 5
    max_steps = 1000
    state_dim = 0
    action_dim = 0
    hidden_units = (64, 64)
    activ = F.relu
    optim = RMSprop
    lr = 0.001
    gamma = 0.99
    ent_weight = 0.01
    val_loss_weight = 0.5
    grad_clip = 0.5
    log_every = 100
    