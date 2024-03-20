from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
import torch
import stable_baselines3 as sb3

class CartPole_fake:
    def __init__(self, config=None, dt = 0.05):
        self.env = gym.make("CartPole-v1", max_episode_steps=5000)
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space
        self.env.tau = dt
        self.env.force_mag = self.env.force_mag*dt/0.02 # scale force 
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

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # print(env_og.tau)
    # env = CartPole_fake(dt=0.02)
    env.reset()
    action_size = env.action_space
    state_size = env.observation_space.shape[0]
    model = sb3.A2C("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save("a2c_cartpole")
    model = sb3.A2C.load("a2c_cartpole")

    # Evaluate the agent and save states and actions
    num_episodes = 25
    dataset = []
    min_length = 1000000
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        local_dataset = []
        while not done:
            action = model.predict(state)[0]
            next_state, _, done,_, _ = env.step(action)

            local_dataset.append(torch.cat((torch.tensor(state), torch.tensor(action).unsqueeze(0))))
            state = next_state
        dataset.append(local_dataset)
        if len(local_dataset) < min_length:
            min_length = len(local_dataset)
        print(f"Episode {episode} finished, number of steps taken: {len(local_dataset)}")

    for i in range(len(dataset)):
        dataset[i] = torch.stack(dataset[i][:min_length])
    dataset = torch.stack(dataset)
    print(dataset.shape)
    torch.save(dataset, "/Users/korneel/coding/A3C/experiments/dataset.pt")
    # Save the dataset
