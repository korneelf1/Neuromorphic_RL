import gym
from gym import spaces
# import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from gym.utils import play

import gymnasium
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}

        return np.concatenate(self._agent_location, self._target_location)

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2
            )

        observation = self._get_obs()
        # observation_agent = observation['agent']
        # observation_target = observation['target']
        # observation = np.concatenate([observation_agent,observation_target])
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        # observation_agent = observation['agent']
        # observation_target = observation['target']
        # observation = np.concatenate([observation_agent,observation_target])
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# from gym.envs.registration import register

# register(
#     id='gym_examples/GridWorld-v0',
#     entry_point='gym_examples.envs:GridWorldEnv',
#     max_episode_steps=300,
# )
class Discret():
    def __init__(self,val) -> None:
        self.n = val
class SimpleGrid():

    def __init__(self, size = 3, start_loc = [0,0], target = [2,2]):
        self.size = size
        self.field = np.zeros((size,size))
        self.field[target[0], target[1]] = 1
        self.state = np.array([*start_loc,*target])

        self.actions = {'0' : [0,-1],
                        '1' : [0,1],
                        '2' : [-1,0],
                        '3' : [1,0]}
        self.action_space = Discret(4)
        self.time_tracker = 0
        self.x_t = np.random.randint(0,self.size)
        self.y_t = np.random.randint(0,self.size)

    def reset(self):
        self.time_tracker = 0
        x = np.random.randint(0,self.size)
        y = np.random.randint(0,self.size)
        self.x_t = np.random.randint(0,self.size)
        self.y_t = np.random.randint(0,self.size)
        self.state = np.array([x,y,self.x_t,self.y_t])
        return self.state, []
    
    def step(self,action):
        dir = self.actions[str(action)]

        new_x = min(max(0,self.state[0]+ dir[0]), self.size-1)
        new_y = min(max(0,self.state[1]+ dir[1]), self.size-1)
        self.time_tracker +=1
        terminal = False
        reward = -self.time_tracker
        if (new_x == self.x_t) & (new_y == self.y_t):
            reward += 200 
            terminal = True
        self.state = np.array([new_x,new_y,self.x_t,self.y_t])
        return self.state, reward, terminal, 0,0
    

class SimpleDrone(gym.Env):
    def __init__(self, render_mode=None, mass = .5, landing_velocity=0.1, dt = 0.01):
        self.mass = mass
        self.g = 9.81
        self.dt = dt

        self._agent_location = 5
        self._agent_velocity = 0

        self._target_location = 0
        self._target_velocity = 0

        self.reward = 0 # -height for each timestep -100 for crash + 250 for landing under 0.1m/s
        self.landing_velocity = -landing_velocity

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([[0],[-.5]]), high=np.array([[2],[.5]]), shape=(2,1))

        self.max_thrust = 50
        self.trajectory = []
        self.thrust = []
        self.times = []
        self.velocities = []
        self.counter = 0
        self.render_mode = render_mode
#  Create a figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        self.line_trajectory, = self.ax.plot([], [], label='Trajectory')
        self.line_velocities, = self.ax.plot([], [], label='Velocities')
        self.line_thrust, = self.ax.plot([], [], label='Thrust')

        # Set up the legend
        self.ax.legend(loc='upper right')
        

    def _get_obs(self):
        # return {"agent": [self._agent_location, self._agent_velocity], "target": self._target_location}
        # [self._agent_location, self._agent_velocity], [self._target_location, self._target_velocity]
        obs = np.array([self._agent_location, self._agent_velocity]).reshape(2,1)
        return torch.tensor(obs)
    
    def reset(self, seed=None, options=None):
        # self._agent_location = np.random.randint(2,10)
        self._agent_location = 2
        # self._agent_velocity = np.random.randint(-5,5)*.1
        self._agent_velocity = 0
        self.reward = 0

        self.trajectory = []
        self.times = []
        self.thrust = []
        self.velocities = []
        self.counter = 0
        self.thrust_last = 0

        self.line_trajectory.set_data([], [])
        self.line_velocities.set_data([], [])
        self.line_thrust.set_data([], [])
        return self._get_obs(),{}
    
    def step(self, action, hover=False):
        '''Instead of learning total thrust, learn delta thrust wrt hover'''
        terminal = False
        truncated = False
        done = False
        self.thrust_last = action

        acceleration = -self.g + self.max_thrust*action/self.mass
        if hover:
            acceleration += self.g
        self._agent_velocity += acceleration*self.dt
        self._agent_location += self._agent_velocity*self.dt

        crash = False
        if self._agent_location < 0:
            if self._agent_velocity > self.landing_velocity:
                terminal = True
                truncated = True
                done = True
                crash = False

                print(self._agent_location)
                print("Landed!\nResults: ", self._agent_location, self._agent_velocity)
            else:
                terminal = True
                truncated = True
                crash = True
                print('Crash')
        elif self._agent_location > 2.2:
            terminal = True
            truncated = True
            print('Too High')

            # Calculate shaped rewards
        distance_to_target = abs(self._agent_location - self._target_location)
        distance_reward = 1/(1+distance_to_target)

        smooth_landing_reward = 250*np.exp(-self._agent_velocity**2)

        energy_penalty = 0.01*abs(action)

        if self.counter % 10 == 0:
            progress_reward = 1
        else:
            progress_reward = 0

        if abs(self._agent_velocity) < 0.1:
            hover_reward = 0.1
        else:
            hover_reward = 0

        if crash:
            self.reward -= 100

        self.reward += distance_reward + smooth_landing_reward - energy_penalty + progress_reward + hover_reward
        
        self.counter +=1
        self.times.append(self.counter)
        self.trajectory.append(self._agent_location)
        self.velocities.append(self._agent_velocity)
        self.thrust.append(action)
        info = {}
        info['distance'] = self._agent_location

        return self._get_obs(), self.reward, terminal, truncated, info
    

    def render(self, mode='human'):
        # self.add_data_point(self.counter, self._agent_location, self._agent_velocity, self.thrust_last)
        if mode == 'post':
            plt.plot(self.times, self.trajectory)
            plt.plot(self.times, self.velocities)
            plt.plot(self.times, self.thrust)
            plt.show()
                

        elif mode == 'human':
            self.line_trajectory.set_data(self.times, self.trajectory)
            self.line_velocities.set_data(self.times, self.velocities)
            self.line_thrust.set_data(self.times, self.thrust)

            # Adjust the plot's axis limits if needed
            self.ax.relim()
            self.ax.autoscale_view()

            # Display the updated plot
            plt.draw()
            plt.pause(0.01)  # Add a small delay to update the plot

    def add_data_point(self, time, trajectory, velocities, thrust):
        self.times.append(time)
        self.trajectory.append(trajectory)
        self.velocities.append(velocities)
        self.thrust.append(thrust)
            

    def close(self):
        pass

class SimpleDrone_Discrete(gym.Env):
    def __init__(self, render_mode=None, mass = .5, landing_velocity=0.1, dt = 0.01):
        self.mass = mass
        self.g = 9.81
        self.dt = dt

        self._agent_location = 5
        self._agent_velocity = 0

        self._target_location = 0
        self._target_velocity = 0

        self.reward = 0 # -height for each timestep -100 for crash + 250 for landing under 0.1m/s
        self.landing_velocity = -landing_velocity

        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = spaces.Box(low=np.array([[0],[-.5]]), high=np.array([[2.2],[.5]]), shape=(2,1))

        self.max_thrust = 15
        self.max_acc = self.max_thrust/self.mass
        self.trajectory = []
        self.thrust = []
        self.times = []
        self.velocities = []
        self.accelerations = []
        self.counter = 0
        self.render_mode = render_mode
#  Create a figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        self.line_trajectory, = self.ax.plot([], [], label='Trajectory')
        self.line_velocities, = self.ax.plot([], [], label='Velocities')
        self.line_thrust, = self.ax.plot([], [], label='Thrust')

        # Set up the legend
        self.ax.legend(loc='upper right')
        

    def _get_obs(self):
        # return {"agent": [self._agent_location, self._agent_velocity], "target": self._target_location}
        # [self._agent_location, self._agent_velocity], [self._target_location, self._target_velocity]
        # obs = np.array([self._agent_location, self._agent_velocity]).reshape(2,1).squeeze()
        obs = np.array([self._agent_location, self._agent_velocity]).reshape(2,1)
        return obs
    def reward_function(self,z, vz,thrust, done):

        reward = 0

        # Penalize for deviation from target height
        reward -= abs(z)

        # Penalize for speed
        reward -= np.abs(vz)

        # Big penalty for crashing 
        if np.abs(z) > 2.2 or (np.abs(z) < 0.1 and np.abs(vz) > 0.1):
            reward -= 100

        # Big reward for soft landing
        if np.abs(vz) < 0.1 and z < 0.1: 
            reward += 100

        return reward
    def reset(self, seed=None, options=None):
        # self._agent_location = np.random.randint(2,10)
        self._agent_location = 2.
        # self._agent_velocity = np.random.randint(-5,5)*.1
        self._agent_velocity = 0.
        self.reward = 0

        self.trajectory = []
        self.times = []
        self.thrust = []
        self.velocities = []
        self.counter = 0
        self.thrust_last = 0

        self.line_trajectory.set_data([], [])
        self.line_velocities.set_data([], [])
        self.line_thrust.set_data([], [])
        
        return self._get_obs(),{}
    

    def action_to_acc(self, action, hover = True):
        '''Converts discrete action to thrust value'''
        steps = self.action_space.n
        return self.max_acc*(action - steps/2)/steps 


    def step(self, action):
        '''Instead of learning total thrust, learn delta thrust wrt hover'''
        terminal = False
        truncated = False
        self.thrust_last = action

        acceleration = self.action_to_acc(action, hover=False) # learn wrt hover or wrt zero acc
        self.accelerations.append(acceleration)
        acceleration_low_passed = 0.4*acceleration + 0.3*self.accelerations[-1]+ 0.3*self.accelerations[-2]  if len(self.accelerations)>1 else acceleration
        
        self._agent_velocity += acceleration_low_passed*self.dt

        self._agent_location += self._agent_velocity*self.dt

  
        self.reward = self.reward_function(self._agent_location, self._agent_velocity, action, terminal)
        if np.abs(self._agent_location) > 2.2:
            terminal = True
            truncated = True
            # self.reward -= 50
            # print('Too High')

            # Calculate shaped rewards
        elif np.abs(self._agent_location)<0.1:
            if np.abs(self._agent_velocity) < 0.1:
                terminal = True
                truncated = True
                # self.reward +=100
                print('Landed')
            else:
                terminal = True
                truncated = True
                # print('Crash')
        self.counter +=1
        self.times.append(self.counter)
        self.trajectory.append(self._agent_location)
        self.velocities.append(self._agent_velocity)
        self.thrust.append(action)
        info = {}
        info['distance'] = self._agent_location

        return self._get_obs(), self.reward, terminal, truncated, info
        

    def render(self, mode='human'):
        # self.add_data_point(self.counter, self._agent_location, self._agent_velocity, self.thrust_last)
        if mode == 'post':
            plt.figure()
            plt.plot(self.times, self.trajectory)
            plt.plot(self.times, self.velocities)
            plt.plot(self.times, self.thrust)
            plt.show()
                

        elif mode == 'human':
            self.line_trajectory.set_data(self.times, self.trajectory)
            self.line_velocities.set_data(self.times, self.velocities)
            self.line_thrust.set_data(self.times, self.thrust)

            # Adjust the plot's axis limits if needed
            self.ax.relim()
            self.ax.autoscale_view()

            # Display the updated plot
            plt.draw()
            plt.pause(0.01)  # Add a small delay to update the plot

    def add_data_point(self, time, trajectory, velocities, thrust):
        self.times.append(time)
        self.trajectory.append(trajectory)
        self.velocities.append(velocities)
        self.thrust.append(thrust)
            

    def close(self):
        pass


gym.register("SimpleDroneDiscrete-v0", entry_point=SimpleDrone_Discrete) 
# # gym.register("SimpleDrone-v0", entry_point=SimpleDrone) 
from stable_baselines3 import A2C


# # Create the environment
env = gym.make("SimpleDroneDiscrete-v0")
env.reset()
done = False
while not done:
    _,_,done,_,_ = env.step(np.array(0))
env.render(mode='post')
# Create the model 
model = A2C("MlpPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=1e6) 

# Save the trained agent
model.save("a2c_drone")

# Test the trained agent
obs,_ = env.reset()
# obs = np.array(obs).reshape(2,1)
for i in range(1000):
  action, _states = model.predict(obs)
  obs, rewards, terminal,truncated, info = env.step(action)
  env.render()
