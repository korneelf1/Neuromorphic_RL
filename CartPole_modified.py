from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box


class CartPole_fake:
    def __init__(self, config=None):
        self.env = gym.make("CartPole-v1", max_episode_steps=5000)
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space
        self.env.tau = 0.05
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info,sth = self.env.step(action)
        return obs, rew, done, info,sth

    def set_state(self, state):
        # self.env = deepcopy(state)
        # obs = np.array(list(self.env.unwrapped.state))
        self.env.reset()
        self.env.state = self.env.unwrapped.state = state
        return state

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def set_dt(self, dt):
        '''default dt is 0.2 sec
        '''
        self.env.tau = dt


class MountainCart_fake:
    def __init__(self, config=None):
        self.env = gym.make("MountainCar-v0")
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info,sth = self.env.step(action)
        return obs, rew, done, info,sth

    def set_state(self, state):
        # self.env = deepcopy(state)
        # obs = np.array(list(self.env.unwrapped.state))
        self.env.reset()
        self.env.state = self.env.unwrapped.state = state
        return state

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()