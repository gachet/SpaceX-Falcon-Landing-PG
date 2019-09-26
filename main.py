import rocket_lander_gym
from utils import multi_env, run_env
from agents.actor_critic_agent import ActorCriticAgent

num_envs = 5
env_name = "MountainCar-v0" # RocketLander-v0 | MountainCar-v0

envs = multi_env(env_name, num_envs)

state_size = envs.observation_space.shape[0]
action_size = envs.action_space.n

agent = ActorCriticAgent(state_size, action_size)

scores = agent.train(envs)

#def get_action(state):
#    action, _, _ = agent.act(state)
#    return action.cpu().numpy()
#
#run_env(env_name, get_action=get_action)