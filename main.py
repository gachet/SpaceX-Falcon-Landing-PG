import torch
import rocket_lander_gym
from utils import multi_env, run_env
from agents.a2c_agent import A2CAgent

num_envs = 5
env_name = 'RocketLander-v0' # RocketLander-v0 | MountainCar-v0 | CartPole-v0

envs = multi_env(env_name, num_envs)

state_size = envs.observation_space.shape[0]
action_size = envs.action_space.n

agent = A2CAgent(state_size, action_size, hidden_size=256)

scores = agent.train(envs)

#####################################################################

torch.save(agent.policy.state_dict(), 'poli   cy_weights.pth')

agent.policy.load_state_dict(torch.load('policy_weights.pth'))

#def get_action(state):
#    
#    action, _, _ = agent.act(state)
#    return action.cpu().numpy()
#
#run_env(env_name, get_action=get_action)

import gym

env = gym.make(env_name)
#env = gym.wrappers.Monitor(env, "recording")

agent.policy.eval()
state = env.reset()
env.render()

while True:
    action, _, _ = agent.act([state])
    action = action.item()
#    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()

    if done: 
        break

env.close()
