import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
# from ..CartPole_modified import CartPole_fake

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import snntorch
from snntorch import surrogate
import wandb

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
INTERACTION_MAX_LENGTH = 500
TBPTT_LENGTH = 50
PADDING_MODE = 'end'
GRADIENT_FREQ = 50   # frequency of gradient updates per rollout interaction
SPIKING = True
PLOTTING = 'wandb' # local or wandb or none

if PLOTTING=='wandb':
    # set up wandb
    wandb.init(project='cartpole', config={'algorithm': 'DSQN', 'learning_rate': 0.01, 'architecture': 'SNN'})

env = gym.make("CartPole-v1")
# env = CartPole_fake()
if PLOTTING=='local':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# set random seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
# env.seed(seed)
np.random.seed(seed)

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory_Full_Seq(object):

    def __init__(self, capacity, interaction_max_length=200, padding='end', tbptt_length=10):
        '''
        Padding can be end or start. 
        End padding will padd the end of the sequence with zeros
        Start padding will pad the start of the sequence with zeros
        tbptt_length is the truncated backpropagation through time length, which defines the length of the sequences being stored. if None, the full sequence is stored'''
        self.memory = deque([], maxlen=capacity)
        self.max_length = interaction_max_length
        self.padding = padding
        self.tbptt_length = tbptt_length

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        states = torch.zeros((self.max_length, 4))
        actions = torch.zeros((self.max_length, 1))
        next_states = torch.zeros((self.max_length, 4))
        rewards = torch.zeros((self.max_length))
        
        #states.fill_(float('inf'))
        #next_states.fill_(float('inf'))

        if state.shape[0] > self.max_length \
            or next_state.shape[0] > self.max_length \
            or action.shape[0] > self.max_length\
            or reward.shape[0] > self.max_length:
            print(state, action, next_state, reward)
            raise ValueError('Interaction length exceeds maximum length')
        else:
            if self.padding == 'end':
                states[:state.shape[0]] = state
                actions[:action.shape[0]] = action
                next_states[:next_state.shape[0]] = next_state
                rewards[:reward.shape[0]] = reward
            elif self.padding == 'start':
                states[self.max_length - state.shape[0]:] = state
                action[self.max_length - state.shape[0]:] = action
                next_states[self.max_length - state.shape[0]:] = next_state
                rewards[self.max_length - state.shape[0]:] = reward
            else:
                raise ValueError('Padding must be either start or end')
        if self.tbptt_length is not None:
            for i in range(0, self.max_length, self.tbptt_length):
                if torch.all(torch.all(states[i:] == 0, dim=0)==1):
                # if torch.isinf(next_state_batch).all(dim=-1):
                    break
                self.memory.append(Transition(states[i:i+self.tbptt_length], actions[i:i+self.tbptt_length], next_states[i:i+self.tbptt_length], rewards[i:i+self.tbptt_length]))
        else:
            self.memory.append(Transition(states, actions, next_states, rewards))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.n_actions = n_actions

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_batch):
        '''
        x shape: (batch_size, observation_length, feature_size)'''
        values = []
        for i in range(state_batch.shape[1]):
            if torch.isinf(state_batch[:,i,:]).all():
                # find way to mask the infinity entries but not the other ones
                # raise Exception('Infinite value in state batch detected')
                # return torch.zeros((state_batch.shape[0], state_batch.shape[1], self.n_actions))
                values.append(torch.zeros((state_batch.shape[0], self.n_actions)))
        # x = F.relu(self.layer1(x))
            x = F.relu(self.layer1(state_batch[:,i,:]))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            values.append(x)
        # return x
        return torch.stack(values, dim=1)
        
class DSQN(nn.Module):
    def __init__(self, n_observations, n_actions, bias=True):
        '''
        Output is population coded with linear layer to actions'''
        super(DSQN, self).__init__()
        spike_fn = surrogate.fast_sigmoid()

        self.layer1 = nn.Linear(n_observations, 128, bias=bias)
        betas = torch.rand(128)
        self.lif1 = snntorch.Leaky(spike_grad=spike_fn, beta=betas, learn_beta=True)

        self.layer2 = nn.Linear(128, 128, bias= bias)
        betas = torch.rand(128)
        self.lif2 = snntorch.Leaky(spike_grad=spike_fn, beta=betas, learn_beta=True)

        self.layer3 = nn.Linear(128, n_actions, bias= bias)
        self.n_actions = n_actions

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def step_forward(self, x):
        x = self.layer1(x)
        spk1, self.mem1 = self.lif1(x,self.mem1)
        x = self.layer2(spk1)
        spk2, self.mem2 = self.lif2(x,self.mem2)
        x = self.layer3(spk2)
        return x
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_batch):
        '''
        x shape: (batch_size, observation_length, feature_size)'''
        self.reset()

        values = []
        for i in range(state_batch.shape[1]):

        # x = F.relu(self.layer1(x))
            x = self.layer1(state_batch[:,i,:])
            spk1, self.mem1 = self.lif1(x,self.mem1)
            x = self.layer2(spk1)
            spk2, self.mem2 = self.lif2(x,self.mem2)
            x = self.layer3(spk2)
            values.append(x)
        # return x
        return torch.stack(values, dim=1)

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the duration of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#



# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset(seed=seed)

n_observations = len(state)

if SPIKING:
    policy_net = DSQN(n_observations, n_actions).to(device)
    target_net = DSQN(n_observations, n_actions).to(device)
else:
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
if PLOTTING=='wandb':
    wandb.watch(policy_net)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory_Full_Seq(10000, padding=PADDING_MODE, interaction_max_length=INTERACTION_MAX_LENGTH, tbptt_length=TBPTT_LENGTH)


steps_done = 0


def select_action(state, spiking=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if spiking:
                return policy_net.step_forward(state.unsqueeze(0)).squeeze(0).max(1).indices.view(1, 1)
            else:
                return policy_net(state.unsqueeze(0)).squeeze(0).max(1).indices.view(1, 1) # unsqueeze to get batch size of 1
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network is updated at every step with a 
# `soft update <https://arxiv.org/pdf/1509.02971.pdf>`__ controlled by 
# the hyperparameter ``TAU``, which was previously defined.
#
NR_OPTIMIZATIONS = 0

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # transitions = [Transition(state=torch.tensor([[-0.0373, -0.0251,  0.0144,  0.0172]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.1064, -0.0352,  0.1266,  0.2401]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), reward=torch.tensor([1.]))]
# 
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state).to(device) # (batch_size, n_observations, observation_length)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    next_state_batch = torch.stack(batch.next_state).to(device)


    # Find the indices of the last non-zero element in the state_batch
    indices = torch.nonzero(torch.isinf(next_state_batch).all(dim=2),as_tuple=False).to(device=device)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(2, action_batch.to(torch.int64)).squeeze(2)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # for i in range(indices.shape[0]):
    #     next_state_batch[indices[i]] = 0
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(2).values #strange that we take the max, cause this max is not necessarily the value of the action taken, which is used for the action-state values
    next_state_values[indices[:,0], indices[:,1]] = 0
    # placeholders for targets and predictions which are padded with zeros rather then the biased outputs
    policy_placeholder = torch.zeros_like(state_action_values,device=device)
    expected_state_action_values_placeholder = torch.zeros_like(next_state_values,device=device)
    if PADDING_MODE == 'end':
        for i in range(BATCH_SIZE):
            if i in indices[:,0]:
                j = indices[indices[:,0]==i,1]
                policy_placeholder[i, :j+1] = state_action_values[i, :j+1]
                # expected_state_action_values[indices[j,0], indices[j,1]] = 0
                expected_state_action_values_placeholder[i,:j] = next_state_values[i,:j]
            # rollout did not stop early
            else:
                policy_placeholder[i] = state_action_values[i]
                expected_state_action_values_placeholder[i] = next_state_values[i]
        # policy_placeholder[:indices[:,0], indices[:,1]] = state_action_values[:indices[:,0], indices[:,1]]
        # expected_state_action_values[indices[:,0], indices[:,1]] = 0
        # expected_state_action_values_placeholder[:indices[:,0], indices[:,1]] = expected_state_action_values[:indices[:,0], indices[:,1]]
    
    else:
        raise ValueError('Padding mode not implemented')

    # Compute the expected Q values
    expected_state_action_values = (expected_state_action_values_placeholder * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    # loss = criterion(state_action_values, expected_state_action_values)
    loss = criterion(policy_placeholder, expected_state_action_values)
    # check correct padding 



    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    global NR_OPTIMIZATIONS
    NR_OPTIMIZATIONS += 1

######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and obtain the initial ``state`` Tensor. Then, we sample
# an action, execute it, observe the next state and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set to 600 if a GPU is available, otherwise 50 
# episodes are scheduled so training does not take too long. However, 50 
# episodes is insufficient for to observe good performance on CartPole.
# You should see the model constantly achieve 500 steps within 600 training 
# episodes. Training RL agents can be a noisy process, so restarting training
# can produce better results if convergence is not observed.
#
def collect_rollout(env, memory, device=device, spiking=False):
    if spiking:
        policy_net.reset()
        target_net.reset()
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    actions = []
    rewards = []
    next_states = []
    states = [] 
    for t in count():
        # print(state)
        action = select_action(state,spiking=spiking)
        # print(action)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            # next_state = torch.ones(state.shape, device=device)*10000
            next_state = torch.tensor(float('inf'), device=device).expand(state.shape)
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)

        

        # Move to the next state
        state = next_state

        # # Perform one step of the optimization (on the policy network)
        # optimize_model()

        # # Soft update of the target network's weights
        # # θ′ ← τ θ + (1 −τ )θ′
        # target_net_state_dict = target_net.state_dict()
        # policy_net_state_dict = policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        # target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            # print(states)
            memory.push(torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.cat(rewards))
            if PLOTTING=='local':
                plot_durations()
            elif PLOTTING=='wandb':
                wandb.log({'episode_duration': t+1})
            break


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 600

for i_episode in range(num_episodes):
    # collect a rollout
    collect_rollout(env, memory, device,spiking=SPIKING)

    for _ in range(GRADIENT_FREQ):
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

print('NR_OPTIMIZATIONS:', NR_OPTIMIZATIONS)
print('Complete')
if PLOTTING=='local':
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. The "older" target_net is also used in optimization to compute the
# expected Q values. A soft update of its weights are performed at every step.
#

########### DEBUG ############
# CURRENT MEMORY:
# deque([Transition(state=tensor([[-0.0373, -0.0251,  0.0144,  0.0172]]), action=tensor([[0]]), next_state=tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), reward=tensor([1.])), Transition(state=tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), action=tensor([[1]]), next_state=tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), reward=tensor([1.])), Transition(state=tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), action=tensor([[0]]), next_state=tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), reward=tensor([1.])), Transition(state=tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), action=tensor([[0]]), next_state=tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), reward=tensor([1.])), Transition(state=tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), action=tensor([[0]]), next_state=tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), reward=tensor([1.])), Transition(state=tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), action=tensor([[1]]), next_state=tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), reward=tensor([1.])), Transition(state=tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), action=tensor([[0]]), next_state=tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), reward=tensor([1.])), Transition(state=tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), action=tensor([[1]]), next_state=tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), reward=tensor([1.])), Transition(state=tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), action=tensor([[1]]), next_state=tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), reward=tensor([1.])), Transition(state=tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), action=tensor([[1]]), next_state=tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), reward=tensor([1.])), Transition(state=tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), action=tensor([[0]]), next_state=tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), reward=tensor([1.])), Transition(state=tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), action=tensor([[1]]), next_state=tensor([[-0.1064, -0.0352,  0.1266,  0.2401]]), reward=tensor([1.]))], maxlen=10000)
# CURRENT BATCH:
# [Transition(state=tensor([[-0.0373, -0.0251,  0.0144,  0.0172]]), action=tensor([[0]]), next_state=tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), reward=tensor([1.])), Transition(state=tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), action=tensor([[1]]), next_state=tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), reward=tensor([1.])), Transition(state=tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), action=tensor([[0]]), next_state=tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), reward=tensor([1.])), Transition(state=tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), action=tensor([[0]]), next_state=tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), reward=tensor([1.])), Transition(state=tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), action=tensor([[0]]), next_state=tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), reward=tensor([1.])), Transition(state=tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), action=tensor([[1]]), next_state=tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), reward=tensor([1.])), Transition(state=tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), action=tensor([[1]]), next_state=tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), reward=tensor([1.])), Transition(state=tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), action=tensor([[0]]), next_state=tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), reward=tensor([1.])), Transition(state=tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), action=tensor([[0]]), next_state=tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), reward=tensor([1.])), Transition(state=tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), action=tensor([[1]]), next_state=tensor([[-0.1064, -0.0352,  0.1266,  0.2401]]), reward=tensor([1.])), Transition(state=tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), action=tensor([[1]]), next_state=tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), reward=tensor([1.])), Transition(state=tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), action=tensor([[1]]), next_state=tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), reward=tensor([1.]))]
# CURRENT LOSS:
# 0.5024523138999939
# policy_net.layer3.bias.grad
# tensor([-0.4953, -0.5000])