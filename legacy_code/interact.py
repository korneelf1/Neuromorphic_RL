# main file combining all components of the A3C algorithm
from helper_functions import A3Cnet as a3c
from helper_functions import Memory, Transition, ActorCritic, ActorCriticSNN, plot_activity, plot_potentials, ActorCriticSNN_LIF

# environment wrapper with tunable states:
from CartPole_modified import MountainCart_fake
from environments import SimpleGrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading

import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# training environment
import gym

def interact(model, env):
    if args.spiking:
        model.init_mem() 

    T +=1
    print('Run: ',T, '\t time: ',time_finish)
    T_lst.append(time_finish)
   
    # retrieve parameters of global nets
    model_params  = model.state_dict()


    model.load_state_dict(model_params)

    # get initial state
    state, info = env.reset()
    # state = env.set_state(constant_state)

    # set up replay memory
    memory = deque([])

    t_sim = 0
    terminal = False
    reward = 0

    values = []
    log_probs = []
    rewards = []
    entropies = []
    val_spk_lst = deque(list(np.ones(T_SNN_VAL_WINDOW)),T_SNN_VAL_WINDOW)
    weights_vals = list(range(T_SNN_VAL_WINDOW))
    weights_sum = sum(weights_vals)
    # print('Interacting with environment')
    while not terminal and t_sim < t_sim_max:

        # state to tensor
        state = torch.from_numpy(state).to(device)

        # get network outputs on given state
        value, policy = model(state.unsqueeze(0))

        # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)

        logprob = F.log_softmax(policy, dim=-1)

        # calculate entropy
        entropy = -(logprob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

        # get log probs of the actions taken
        log_prob = logprob.gather(1,action)
        
        # perform action
        obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))

        # !!!!!!!!!!!
        reward = max(min(reward, 1), -1)

        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)

        
        if not terminal:
            state = obs

        t_sim += 1

    time_finish = t_sim   
 
    R = 0
    if not terminal:
        R, _ = model(torch.from_numpy(state).to(device))
        R = R.detach()

        
    # save current R (value) for the gneralized advantage DeltaT
    prev_val = R
    values.append(R)