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

config.num_envs = 5
config.env_name = 'LunarLander-v2' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.env_solved = 200
config.envs  = multi_env(config.env_name, config.num_envs)
config.eval_env = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.num_episodes = 2000
config.steps = 5
config.activ_actor = F.relu
config.activ_critic = F.relu
config.optim_actor = Adam
config.optim_critic = Adam
config.lr_actor = 0.001
config.lr_critic = 0.001
config.gamma = 0.99
config.ppo_clip = 0.2
config.ppo_epochs = 10
config.ppo_batch_size = 5
config.ent_weight = 0.01
config.grad_clip_actor = None
config.grad_clip_critic = None
config.use_gae = True
config.lamda = 0.95
config.log_every = 100
config.render_eval = False
config.num_evals = 10

#agent = A2CAgent(config)
agent = PPOAgent(config)

agent.train()

torch.save(agent.policy.state_dict(), 'policy_weights.pth')