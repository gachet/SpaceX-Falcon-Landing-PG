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
config.env_name = 'RocketLander-v0' # RocketLander-v0 | MountainCar-v0 | CartPole-v0
config.envs  = multi_env(config.env_name, config.num_envs)
config.eval_env = gym.make(config.env_name)
config.state_dim = config.envs.observation_space.shape[0]
config.action_dim = config.envs.action_space.n
config.num_episodes = 2000
config.rollout = 100 # 5
config.max_steps = 1000
config.hidden_units = (64, 64)
config.activ = F.relu
config.optim = Adam
config.lr = 0.001
config.gamma = 0.99
config.ent_weight = 0.01
config.val_loss_weight = 0.5
config.grad_clip = 0.5
config.log_every = 100

agent = A2CAgent(config)

agent.train()

#####################################################################

torch.save(agent.policy.state_dict(), 'policy_weights.pth')
#
agent.policy.load_state_dict(torch.load('policy_weights.pth'))

#def get_action(state):
#    
#    actiogn, _, _ = agent.act(state)
#    return action.cpu().numpy()
#
#run_env(env_name, get_action=get_action)

#env = gym.wrappers.Monitor(env, "recording")

#agent.policy.eval()
#state = env.reset()
#env.render()
#
#while True:
#    action, _, _, _ = agent.policy([state])
#    action = action.item()
##    action = env.action_space.sample()
#    state, reward, done, _ = env.step(action)
#    env.render()
#
#    if done: 
#        break
#
#env.close()
