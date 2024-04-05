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
import time
# DQN Components
from replay import ReplayMemory, ReplayMemory_Full_Seq, Transition, Transition_Spiking
from agents import DQN, DSQN

DEBUG = False
# start timing
t0 = time.time()

BATCH_SIZE = 12 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor as mentioned in the previous section
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.05 # final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 1e-4 # learning rate of the ``AdamW`` optimizer
INTERACTION_MAX_LENGTH = 500 # maximum length of an interaction
TBPTT_LENGTH = int(2) # truncated backpropagation through time length
PADDING_MODE = 'end' # padding mode for the replay buffer
GRADIENT_FREQ = 50   # frequency of gradient updates per rollout interaction
SPIKING = True # use spiking or non-spiking network
PLOTTING = 'local' # local or wandb or none
ITERATIONS = int(1e3) # number of training iterations (corresponding to collections of rollouts)
WARMUP = 0 # first warmup steps where no optimization is performeds
seed = 7

# set random seeds for reproducibility
random.seed(seed)
torch.manual_seed(seed)
# env.seed(seed)
np.random.seed(seed)

if PLOTTING=='wandb':
    # set up wandb
    wandb.init(project='cartpole', config={'algorithm': 'DSQN', 
                                           'SPIKING': SPIKING,
                                           'BATCH_SIZE':BATCH_SIZE,
                                           'TBPTT_LENGTH': TBPTT_LENGTH,
                                           'PADDING_MODE': PADDING_MODE,
                                           'EPS_START': EPS_START,
                                           'EPS_END': EPS_END,
                                           'EPS_DECAY': EPS_DECAY,
                                           'GAMMA': GAMMA,
                                           'TAU': TAU,
                                           'LR': LR,
                                           'GRADIENT_FREQ': GRADIENT_FREQ,
                                           'INTERACTION_MAX_LENGTH': INTERACTION_MAX_LENGTH,
                                           'seed': seed,})
    

env = gym.make("CartPole-v1")
env.action_space.seed(14)

if PLOTTING=='local':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

# # Check that MPS is available
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(device)
# if DEBUG:
# set random seeds for reproducibility




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
    policy_net = DSQN(n_observations, n_actions,device=device,BATCH_SIZE=BATCH_SIZE).to(device)
    target_net = DSQN(n_observations, n_actions,device=device,BATCH_SIZE=BATCH_SIZE).to(device)
else:
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
if PLOTTING=='wandb':
    wandb.watch(policy_net)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory_Full_Seq(10000, padding=PADDING_MODE, interaction_max_length=INTERACTION_MAX_LENGTH, tbptt_length=TBPTT_LENGTH, keep_hidden_states=SPIKING)


steps_done = 0


def select_action(state, spiking=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # if DEBUG:
    #    eps_threshold = 0
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        if spiking:
            out, hidden_state = policy_net.step_forward(state.unsqueeze(0))
            out =  out.squeeze(0).max(1).indices.view(1, 1)
        else:
            out = policy_net(state.unsqueeze(0)).squeeze(0).max(1).indices.view(1, 1) # unsqueeze to get batch size of 1
    if sample > eps_threshold:
        if spiking:
            return out,hidden_state
        return out
    else:
        if spiking:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), hidden_state
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result DSQN')
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
PRINT_TIMES = False
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return False
    assert WARMUP<TBPTT_LENGTH, 'Warmup steps must be less than TBPTT_LENGTH'
    transitions = memory.sample(BATCH_SIZE)

    # transitions = [Transition(state=torch.tensor([[-0.0373, -0.0251,  0.0144,  0.0172]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0378, -0.2205,  0.0147,  0.3144]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0967, -0.2254,  0.1049,  0.4255]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.1012, -0.0319,  0.1134,  0.1676]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0471, -0.4164,  0.0280,  0.6250]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0422, -0.0256,  0.0210,  0.0263]]), action=torch.tensor([[0]]), next_state=torch.tensor([[-0.0427, -0.2210,  0.0215,  0.3256]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.1018, -0.2285,  0.1168,  0.4938]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.1064, -0.0352,  0.1266,  0.2401]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0554, -0.6119,  0.0405,  0.9263]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0677, -0.4173,  0.0591,  0.6467]]), reward=torch.tensor([1.])), Transition(state=torch.tensor([[-0.0760, -0.6132,  0.0720,  0.9574]]), action=torch.tensor([[1]]), next_state=torch.tensor([[-0.0883, -0.4192,  0.0911,  0.6881]]), reward=torch.tensor([1.]))]
# 
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    if SPIKING:
        batch = Transition_Spiking(*zip(*transitions))
    else:
        batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state).to(device) # (batch_size, n_observations, observation_length)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    next_state_batch = torch.stack(batch.next_state).to(device)
    if SPIKING:
        hidden_states = torch.stack(batch.hidden_states).to(device)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        state_action_values = policy_net(state_batch,hidden_states).gather(2, action_batch.to(torch.int64)).squeeze(2)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = target_net(next_state_batch,hidden_states).max(2).values #strange that we take the max, cause this max is not necessarily the value of the action taken, which is used for the action-state values

    else:
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        state_action_values = policy_net(state_batch).gather(2, action_batch.to(torch.int64)).squeeze(2)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = target_net(next_state_batch).max(2).values #strange that we take the max, cause this max is not necessarily the value of the action taken, which is used for the action-state values

    # Find the indices of the last non-zero element in the state_batch
    # indices = torch.ones(BATCH_SIZE,dtype=torch.int64, device=device)*TBPTT_LENGTH # create tensor of indices with the last index
    indices = torch.nonzero(torch.isinf(next_state_batch).all(dim=2),as_tuple=False).to(device=device)
    # indices[final_states[:,0]] = final_states[:,1]

    # placeholders for targets and predictions which are padded with zeros rather then the biased outputs

    if PADDING_MODE == 'end':
        
     
        for row,col in indices:
            state_action_values[row, col+1:] = 0
            next_state_values[row, col:] = 0
   
    else:
        raise ValueError('Padding mode not implemented')
    # policy_placeholder = state_action_values
    # expected_state_action_values_placeholder = next_state_values

    # policy_placeholder = torch.zeros_like(state_action_values,device=device)
    # expected_state_action_values_placeholder = torch.zeros_like(next_state_values,device=device)
    # state_action_values[indices[:,0], indices[:,1]+1:] = 0
    # next_state_values[indices[:,0], indices[:,1]:] = 0
    # if PADDING_MODE == 'end':
    #     for i in range(BATCH_SIZE):
    #         if i in indices[:,0]:
    #             j = indices[indices[:,0]==i,1]
    #             # policy_placeholder[i, :j+1] = state_action_values[i, :j+1]
    #             expected_state_action_values_placeholder[i,:j] = next_state_values[i,:j]
    #         # rollout did not stop early
    #         else:
    #             # policy_placeholder[i] = state_action_values[i]
    #             expected_state_action_values_placeholder[i] = next_state_values[i]

    # else:
    #     raise ValueError('Padding mode not implemented')

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)
    # loss = criterion(policy_placeholder, expected_state_action_values)
    # check correct padding 



    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    global NR_OPTIMIZATIONS
    NR_OPTIMIZATIONS += 1

    return True

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
    hidden_states = []
    for t in count():
        # print(state)
        action = select_action(state,spiking=spiking)
        if spiking:
            action, hidden_state = action
            hidden_states.append(hidden_state)

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

        if done or len(memory)+t == BATCH_SIZE-1:
            episode_durations.append(t + 1)
            # print(states)
            if spiking:
                memory.push(torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.cat(rewards), torch.stack(hidden_states))
            else:
                memory.push(torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.cat(rewards))
            # if PLOTTING=='local':
            #     plot_durations()
            # elif PLOTTING=='wandb':
            #     wandb.log({'episode_duration': t+1})
            #     durations_t = torch.tensor(episode_durations, dtype=torch.float)

            #     if len(durations_t) >= 100:
            #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            #         means = torch.cat((torch.zeros(99), means))
            #         wandb.log({'mean_duration': means[-1]})
            return t+1

######################################################################
import time as t

print('Starting training at time:', t.time())
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = ITERATIONS
    # num_episodes = 30
else:
    num_episodes = 5


for i_episode in range(num_episodes):
    # collect a rollout
    time_episode = collect_rollout(env, memory, device,spiking=SPIKING)
    if PLOTTING=='local':
        plot_durations()
    elif PLOTTING=='wandb':
        wandb.log({'episode_duration': time_episode+1})
        durations_t = torch.tensor(episode_durations, dtype=torch.float)

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            wandb.log({'mean_duration': means[-1]})
    
    # time_episode=3
    for _ in range(time_episode):
        # Perform one step of the optimization (on the policy network)

        optimized = optimize_model()
        if optimized:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

print('NR_OPTIMIZATIONS:', NR_OPTIMIZATIONS)

print('Complete, duration:', time.time()-t0)
if PLOTTING=='local':
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
elif PLOTTING=='wandb':
    wandb.log({'NR_OPTIMIZATIONS': NR_OPTIMIZATIONS})
    wandb.finish()
