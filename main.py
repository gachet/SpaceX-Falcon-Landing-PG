import gym
import torch
import torch.nn.functional as F
import rocket_lander_gym

from torch.optim import RMSprop, Adam
from utils import multi_env
from agents.config import Config
from agents.a2c_agent import A2CAgent

config = Config()

#####################################################################

config.num_envs = 1
config.env_name = 'RocketLander-v0' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.env_solved = 1
config.envs  = multi_env(config.env_name, config.num_envs)
config.eval_env = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.num_episodes = 20000
config.rollout = 1000
config.max_steps = 1000
config.hidden_units = (64, 64)
config.activ = torch.tanh
config.optim = RMSprop
config.lr = 0.001
config.gamma = 0.99
config.ent_weight = 0.25
config.val_loss_weight = 0.05
config.grad_clip = 1
config.log_every = 100

agent = A2CAgent(config)

agent.train()

torch.save(agent.policy.state_dict(), 'policy_weights.pth')

#####################################################################

config.env_name = 'RocketLander-v0' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.envs = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.hidden_units = (64, 64)
config.activ = torch.tanh

agent = A2CAgent(config)

agent.policy.load_state_dict(torch.load('policy_weights.pth'))

agent.run_episode()
