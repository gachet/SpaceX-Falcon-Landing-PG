import gym
import torch
import torch.nn.functional as F
import rocket_lander_gym

from torch.optim import RMSprop, Adam
from utils import multi_env
from agents.config import Config
from agents.a2c_agent import A2CAgent

config = Config()

config.num_envs = 5
config.env_name = 'LunarLander-v2' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.solved_with = 200
config.envs  = multi_env(config.env_name, config.num_envs)
config.eval_env = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.num_episodes = 5000
config.rollout = 5
config.max_steps = 1000
config.hidden_units = (16, 16, 16)
config.activ = F.tanh
config.optim = RMSprop
config.lr = 0.001
config.gamma = 0.99
config.ent_weight = 0.01
config.val_loss_weight = 0.25
config.grad_clip = 5
config.log_every = 100

agent = A2CAgent(config)

agent.train()

torch.save(agent.policy.state_dict(), 'policy_weights.pth')