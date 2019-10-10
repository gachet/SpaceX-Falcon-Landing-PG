import gym
from multi_processing_env import SubprocVecEnv

def make_env(env_name):
    return lambda: gym.make(env_name)

def multi_env(env_name, num_envs):
    envs = [make_env(env_name) for _ in range(num_envs)]
    return SubprocVecEnv(envs)