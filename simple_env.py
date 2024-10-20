from transformers import pipeline, AutoProcessor
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from pettingzoo.mpe import simple_v3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transformers import pipeline
# Convert the PettingZoo environment to a Gymnasium environment
class CustomSimpleEnv(gym.Env):
    def __init__(self, env, render_mode='rgb_array'):
        self.env = env
        self.action_space = env.action_space(env.possible_agents[0])
        self.observation_space = env.observation_space(env.possible_agents[0])
        self.render_mode = render_mode

    def reset(self, seed=None, **kwargs):
        self.env.reset()
        return self.env.observe(self.env.possible_agents[0]), {}

    def step(self, action):
        self.env.step(action)
        obs = self.env.observe(self.env.possible_agents[0])
        reward = self.env.rewards[self.env.possible_agents[0]]
        terminated = self.env.terminations[self.env.possible_agents[0]]
        truncated = self.env.truncations[self.env.possible_agents[0]]
        info = self.env.infos[self.env.possible_agents[0]]
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        return self.env.render()

    @staticmethod
    def display_image(image: np.ndarray):
        plt.imshow(image)
        plt.axis('off')
        plt.show()
