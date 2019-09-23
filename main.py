import gym
import gym.spaces
import rocket_lander_gym
from agents.PGAgent import PGAgent

env = gym.make('RocketLander-v0')

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
hidden_size = (64, 64)

agent = PGAgent(state_size=state_size, 
                hidden_size=hidden_size, 
                action_size=action_size)

scores = agent.train(env, n_episodes=2000)

#import matplotlib.pyplot as plt
#plt.plot(scores)