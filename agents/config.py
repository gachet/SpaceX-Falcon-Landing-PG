import torch.nn.functional as F
from torch.optim import RMSprop

class Config:
    seed = 101
    num_envs = 5
    env_name = 'CartPole-v0'
    solved_with = 195
    envs = []
    eval_env = None
    num_episodes = 2000
    steps = 5
    max_steps = 1000
    state_dim = 0
    action_dim = 0
    hidden_units = (64, 64)
    hidden_actor = (64, 64)
    hidden_critic = (64, 64)
    activ = F.relu
    activ_actor = F.relu
    activ_critic = F.relu
    optim = RMSprop
    optim_actor = RMSprop
    optim_critic = RMSprop
    lr = 0.001
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.99
    ent_weight = 0.01
    val_loss_weight = 0.5
    grad_clip = 0.5
    log_every = 100
    use_gae = False
    lamda = 0.95
    