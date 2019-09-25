import torch
import gym
import gym.spaces
import rocket_lander_gym

from agents.PGAgent import PGAgent

#env = gym.make('RocketLander-v0')
env = gym.make('CartPole-v0')
env.seed(0)

#env.reset()
#
#PRINT_DEBUG_MSG = True
#
#while True:
#    env.render()
#    action = env.action_space.sample()
#    observation,reward,done,info = env.step(action)
#
#    if PRINT_DEBUG_MSG:
#        print("Action Taken  ",action)
#        print("Observation   ",observation)
#        print("Reward Gained ",reward)
#        print("Info          ",info,end='\n\n')
#
#    if done:
#        print("Simulation done.")
#        break
#
#env.close()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = PGAgent(state_size, 
                action_size, 
                hidden_size=(16,), 
                activ=torch.tanh,
                lr=1e-2)

scores = agent.train(env, n_episodes=1000)

import matplotlib.pyplot as plt
plt.plot(scores)

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