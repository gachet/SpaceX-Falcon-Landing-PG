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
config.env_name = 'RocketLander-v0' # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
config.solved_with = 0.5
config.envs  = multi_env(config.env_name, config.num_envs)
config.eval_env = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.num_episodes = 5000
config.steps = 5
config.max_steps = 1000
config.hidden_units = (128,)
config.hidden_actor = (128,)
config.hidden_critic = (128,)
config.activ = torch.tanh
config.activ_actor = torch.tanh
config.activ_critic = torch.tanh
config.optim = Adam
config.optim_actor = Adam
config.optim_critic = Adam
config.lr = 1e-3
config.lr_actor = 1e-3
config.lr_critic = 1e-3
config.gamma = 0.99
config.ent_weight = 1e-3
config.val_loss_weight = 1e-3
config.grad_clip = 5
config.use_gae = True
config.lamda = 0.95
config.log_every = 100
config.render_eval = False

agent = A2CAgent(config)

agent.train()

torch.save(agent.policy.state_dict(), 'policy_weights.pth')