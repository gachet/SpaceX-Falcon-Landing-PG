import torch
import gym
import gym.spaces
import rocket_lander_gym

from agents.pg_agent import PGAgent
from agents.actor_critic_agent import ActorCriticAgent
from agents.multi_processing_env import SubprocVecEnv
from utils import make_env

num_envs = 16
env_name = "RocketLander-v0" # RocketLander-v0 | CartPole-v0

envs = [make_env(env_name) for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_size = envs.observation_space.shape[0]
action_size = envs.action_space.n

agent = ActorCriticAgent(state_size, action_size)

scores = agent.train(envs)

#import matplotlib.pyplot as plt
#plt.plot(scores)

#env.reset()
#
#while True:
#    env.render()
#    action, _ = agent.act()
#    state, reward, done, info = env.step(action)
#
#    if PRINT_DEBUG_MSG:
#        print("Action Taken  ", action)
#        print("Observation   ", observation)
#        print("Reward Gained ", reward)
#        print("Info          ", info, end='\n\n')
#
#    if done:
#        print("Simulation done.")
#        break
#
#env.close()