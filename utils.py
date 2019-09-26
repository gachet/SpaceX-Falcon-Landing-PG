import gym
from multi_processing_env import SubprocVecEnv

def make_env(env_name):
    return lambda: gym.make(env_name)

def multi_env(env_name, num_envs):
    envs = [make_env(env_name) for _ in range(num_envs)]
    return SubprocVecEnv(envs)

def run_env(env_name, get_action=None, close_env=True):
    env = gym.make(env_name)
    
    if get_action is None:
        get_action = lambda _: env.action_space.sample()
        
    state = env.reset()
    env.render()
    
    while True:
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
    
        if done: break
    
    if close_env:
        env.close()