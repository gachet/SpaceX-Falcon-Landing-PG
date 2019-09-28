import torch
import rocket_lander_gym
from utils import multi_env, run_env
from agents.actor_critic_agent import ActorCriticAgent

num_envs = 10
env_name = 'RocketLander-v0' # RocketLander-v0 | MountainCar-v0

envs = multi_env(env_name, num_envs)

state_size = envs.observation_space.shape[0]
action_size = envs.action_space.n

agent = ActorCriticAgent(state_size, action_size, hidden_size=16, lr=1e-6)

scores = agent.train(envs, 
                     n_episodes=20000, 
                     max_t=1000, 
                     coef_val=0.5, 
                     coef_ent=0.001)





#torch.save(agent.policy.state_dict(), 'policy_weights.pth')
agent.policy.load_state_dict(torch.load('policy_weights.pth'))
#
def get_action(state):
    
    action, _, _ = agent.act(state)
    return action.cpu().numpy()
#
#run_env(env_name, get_action=get_action)

import gym

env = gym.make(env_name)
#env = gym.wrappers.Monitor(env, "recording")
#    
state = env.reset()
env.render()

for t in range(1000):
    action, _, _ = agent.act([state])
#    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()

    if done: 
        break

env.close()
