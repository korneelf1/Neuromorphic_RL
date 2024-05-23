import gym
from gym import spaces
# import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt

'''
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
# )'''  


class Discret():
    def __init__(self,val) -> None:
        self.n = val



class SimpleDrone_Discrete(gym.Env):
    def __init__(self, render_mode=None, mass = .5, landing_velocity=0.4, dt = 0.02, max_episode_length = 200, train_vel = False, optic_flow=False):
        """
        optic_flow if true return optic flow else height"""
        self.mass = mass
        self.g = 9.81
        self.dt = dt
        self.max_episode_length = max_episode_length

        self._agent_location = 5
        self._agent_velocity = 0

        self._target_location = 0
        self._target_velocity = 0

        self.reward = 0 # -height for each timestep -100 for crash + 250 for landing under 0.1m/s
        self.landing_velocity = -landing_velocity

        self.action_space = gym.spaces.Discrete(7)
        # self.observation_space = spaces.Box(low=np.array([[0],[-.5]]), high=np.array([[2.2],[.5]]), shape=(2,1))
        self.observation_space = spaces.Box(low=np.array([[0]]), high=np.array([[2.5]]), shape=(1,1))
        self.train_vel = train_vel

        self.max_thrust = 5
        self.max_acc = 1
        self.accelerations_actions = np.linspace(-self.max_acc,self.max_acc,7)
        self.trajectory = []
        self.thrust = []
        self.times = []
        self.velocities = []
        self.accelerations = []
        self.counter = 0
        self.render_mode = render_mode

        self.done = False
        self.prev_shaping = None
        
        self.optic_flow = optic_flow

    def _get_obs(self):
        # return {"agent": [self._agent_location, self._agent_velocity], "target": self._target_location}
        # [self._agent_location, self._agent_velocity], [self._target_location, self._target_velocity]
        # obs = np.array([self._agent_location, self._agent_velocity]).reshape(2,1).squeeze()
        if self.optic_flow:
            obs = torch.tensor([self._agent_velocity/self._agent_location], dtype=torch.float32).reshape(1,)
        else:
            obs = torch.tensor([self._agent_location]).reshape(1,)
        vel = torch.tensor([self._agent_velocity], dtype=torch.float32).reshape(1,)
        # obs = np.array([self._agent_location]).reshape(1,)
        if self.train_vel:
            return obs, vel
        else:
            return np.array([self._agent_location]).reshape(1,)
    



    def reward_function_snn(self,z, vz,thrust, done, normalized=False):
        touch_ground = False
        if abs(z)<0.1:
            touch_ground = True
        
        reward = 100
        if z>-.1:
            # reward += 100/(1+abs(z))
            reward -= z*100
        
        if vz<0:
            distance_from_target_factor = (2.5-z)/2.5
            # reward += 50*np.exp(-abs(vz)*5)

            reward += (vz*200)*distance_from_target_factor + (1-distance_from_target_factor)*200*-vz
        else:
            # reward -= 100*np.exp(abs(vz)*5e-1) # factor 100 so 0 reward if upward speed at height 0
            reward -= (vz+1) *100

        if z < -.1:
            reward = 0 
        
        # landing
        if touch_ground and 0 > (vz) > -self.landing_velocity:
            reward += 100
            print('Landed')
            # print(reward)
        return reward

        

    def reward_function(self,z, vz,thrust, done, normalized=False):
        touch_ground = False
        if abs(z)<0.1:
            touch_ground = True
        state = [z, vz, touch_ground]

        reward = -1
        if vz<0:
            reward += 5

        
        shaping = (
            -200 * np.abs(state[0])
            - 100 * np.abs(state[1])
            + 20 * state[2]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= 1 # we want fast landing
        # reward += -state[1]*10
        terminated = False
        crash = False
        if touch_ground and abs(vz) > self.landing_velocity:
            crash = True

        if abs(state[0]) >= 2.2:
            terminated = True
            reward -= 400
            # reward -= abs((state[1])*25)
        # elif touch_ground and abs(vz) < self.landing_velocity:
        elif touch_ground:
            terminated = True
            # bonus for soft landing
            reward += abs((1/(state[1]+1e-3)*100))
            # print(reward,(1/(state[1]+1e-3)*25))
        if self.counter>= self.max_episode_length-1:
            terminated = True
            reward -= 200
        return reward
    
    def plot_reward_function(self):
        # Define the range of z and vz values
        z_values = np.linspace(0, 2.5, 100)
        vz_values = np.linspace(-1, 1, 100)

        # Create a grid of z and vz values
        Z, VZ = np.meshgrid(z_values, vz_values)

        # Calculate the reward for each combination of z and vz
        reward = np.zeros_like(Z)
        for i in range(len(z_values)):
            for j in range(len(vz_values)):
                reward[i, j] = self.reward_function_snn(Z[i, j], VZ[i, j], 0, False)

        # Create a surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Z, VZ, reward, cmap=cm.coolwarm)

        # Set labels and title
        ax.set_xlabel('z')
        ax.set_ylabel('vz')
        ax.set_zlabel('Reward')
        ax.set_title('Reward Function (SNN)')

        # Show the plot
        plt.show()
    def reset(self, seed=None, options=None):
        # self._agent_location = np.random.randint(2,10)
        self._agent_location = np.random.randint(5,20)*.1
        
        # self._agent_location = 2
        # self._agent_velocity = np.random.randint(-5,5)*.1
        self._agent_velocity = np.random.randint(-3,3)*.1
        if self.train_vel:
            self._agent_velocity = 0
            self._agent_location = np.random.randint(10,20)*.1
        # self._agent_velocity = 0
        self.reward = 0

        self.mass = 0.5 + (np.random.rand()-0.5)*0.01
        self.trajectory = []
        self.times = []
        self.thrust = []
        self.velocities = []
        self.accelerations = []
        self.counter = 0
        self.thrust_last = 0
        self.done = False

        self._agent_location = 0.8
        
        return self._get_obs(),{}
    

    def action_to_acc(self, action, hover = True):
        '''Converts discrete action to thrust value'''
        
        return self.accelerations_actions[action] 


    def step(self, action):
        '''Instead of learning total thrust, learn delta thrust wrt hover'''
        if self.train_vel:
            self.done = False
        
        if self.done:
            return self._get_obs(), self.reward, True, True, {}
        terminal = False
        truncated = False
        self.thrust_last = action

        low_pass_len = .1 # sec
        index = int(low_pass_len//self.dt)
        acceleration = self.action_to_acc(action) # learn wrt hover or wrt zero acc
        self.accelerations.append(acceleration)
        acceleration_low_passed = 0.4*acceleration + 0.6*np.mean(self.accelerations[-index:-1])  if len(self.accelerations)>index else acceleration

        self._agent_velocity += (acceleration_low_passed)*self.dt

        self._agent_location += self._agent_velocity*self.dt

        info = {}

        info['landed'] = False
        info['end_condition'] = 'none'
        self.reward = self.reward_function_snn(self._agent_location, self._agent_velocity, action, terminal)
        if np.abs(self._agent_location) > 2.2:
            terminal = True
            truncated = True
            info['landed'] = False
            # self.reward -= 50
            # print('Too High')

            # Calculate shaped rewards
        elif self._agent_location<0.1:
            self.done = True

            if np.abs(self._agent_velocity) < self.landing_velocity:
                terminal = True
                truncated = True
                # self.reward +=100
                info['landed'] = True
                info['end_condition'] = 'landed'
                print('Landed!\nResults: ', self._agent_location, self._agent_velocity)

            else:
                terminal = True
                truncated = True
                # print('Crash')
                info['end_condition'] = 'crash'
                info['landed'] = False
        elif self._agent_location<0:
            terminal = True
            truncated = True
            info['end_condition'] = 'too low'
        
        elif self._agent_location>2.2:
            terminal = True
            truncated = True    
            info['end_condition'] = 'too high'

        self.counter +=1
        self.times.append(self.counter)
        self.trajectory.append(self._agent_location)
        self.velocities.append(self._agent_velocity)
        self.thrust.append(action)
        info['distance'] = self._agent_location
        
        if self.counter>= self.max_episode_length-1:
            terminal = True
            truncated = True
            # self.reward -= -200
        if self.train_vel:
            terminal, truncated = (False,False)
        return self._get_obs(), self.reward, terminal, truncated, info
        

    def render(self, mode='post'):
        # Set up the legend
        self.ax.legend(loc='upper right')
        # self.add_data_point(self.counter, self._agent_location, self._agent_velocity, self.thrust_last)
        if mode == 'post':
            plt.figure()
            plt.plot(self.times, self.trajectory)
            plt.plot(self.times, self.velocities)
            plt.plot(self.times, self.thrust)
            plt.show()
                


    def add_data_point(self, time, trajectory, velocities, thrust):
        self.times.append(time)
        self.trajectory.append(trajectory)
        self.velocities.append(velocities)
        self.thrust.append(thrust)
            

    def close(self):
        pass