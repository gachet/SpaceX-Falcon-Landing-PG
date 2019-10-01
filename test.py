import gym
import torch
import torch.nn.functional as F
import rocket_lander_gym

from agents.config import Config
from agents.a2c_agent import A2CAgent

config = Config()

################################ Running ################################

config.env_name = 'LunarLander-v2' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.envs = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.hidden_units = (64, 64)
config.activ = torch.tanh

agent = A2CAgent(config)

agent.policy.load_state_dict(torch.load('policy_weights.pth'))

agent.run_episode()
