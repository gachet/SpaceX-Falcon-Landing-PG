import gym
import torch
import torch.nn.functional as F
import rocket_lander_gym

from torch.optim import RMSprop, Adam
from utils import multi_env
from agents.config import Config
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent

config = Config()

config.env_name = 'RocketLander-v0' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.seed = 0
config.num_agents = 5
config.envs = multi_env(config.env_name, config.num_agents)
config.num_episodes = 1000
config.steps = 1000
config.state_size = config.envs.observation_space.shape[0]
config.action_size = config.envs.action_space.shape[0]
config.activ_actor = F.relu
config.lr_actor = 3e-4
config.hidden_actor = (512, 512)
config.optim_actor = Adam
config.grad_clip_actor = 5
config.activ_critic = F.relu
config.lr_critic = 3e-4
config.hidden_critic = (512, 512)
config.optim_critic = Adam
config.grad_clip_critic = 5
config.gamma = 0.99
config.ppo_clip = 0.2
config.ppo_epochs = 10
config.ppo_batch_size = 32
config.ent_weight = 0.01
config.val_loss_weight = 1
config.use_gae = True
config.lamda = 0.95
config.env_solved = 1.0
config.times_solved = 10

#agent = A2CAgent(config)
agent = PPOAgent(config)

agent.train()