# main file combining all components of the A3C algorithm
from helper_functions import Feedforward as f
from helper_functions import A3Cnet as a3c
from helper_functions import Memory, Transition, ActorCritic, ActorCriticSNN

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

# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make("CartPole-v1")

# init plt
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()
# plt.figure()
# plt.xlabel('Episode')
# plt.ylabel('Duration')

show_result = True
# global parameters
T = 0
th = 0
th_v = 0
GAMMA = .9
LAMBDA_G = .9
ENTROPY_COEF = .01
VALUE_LOSS_COEF = 0.5


T_max = 10000
# T_max = 10
t_sim_max = 200
T_SNN_VAL_WINDOW = 5

# critic_global = f(2,1,3,200)
# actor_global = f(2,3,3,200)

# # multithreading purposes
# nr_threads = threading.active_count()

# # create 4 agents
# actor1 = f(2,3,3,120)

# # create 4 critics
# critic1 = f(2,1,3,120)


# optimizer_actor1 = torch.optim.Adam(actor1.parameters(), lr = 1e-4)
# optimizer_critic1 = torch.optim.Adam(critic1.parameters(), lr = 1e-4)

# model = a3c(4,2,1,3,200)
# model = ActorCritic(2,env.action_space)
# model = ActorCritic(2,env.action_space)

model = ActorCriticSNN(4, env.action_space, T_SNN_VAL_WINDOW)

# model = ActorCritic(2, env.action_space)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss = 0
time_finish = 0
T_lst = []
while T<T_max:
    model.init_mem()
    T +=1
    if T==1:
        
        # torch.save(actor1.state_dict(), 'actor_1.txt')
        # torch.save(critic1.state_dict(), 'critic_1.txt')
        torch.save(model.state_dict(), 'model.txt')


    if T == 500:
        
        # torch.save(actor1.state_dict(), 'actor_500.txt')
        # torch.save(critic1.state_dict(), 'critic_500.txt')
        torch.save(model.state_dict(), 'model_500.txt')

    print('Run: ',T, '\t time: ',time_finish)
    T_lst.append(time_finish)
    # reset gradients d theta and d theta_v to zero
    # critic_global.zero_grad()
    # actor_global.zero_grad()

    # optimizer_actor1.zero_grad()
    # optimizer_critic1.zero_grad()
    # optimizer.zero_grad()

    # dth   = 0
    # dth_v = 0
    
    # retrieve parameters of global nets
    # critic_params = critic_global.state_dict()
    # actor_params  = actor_global.state_dict()
    model_params  = model.state_dict()

    # create thread specific actors and critics (assume 4 threads for now)
    # actor1.load_state_dict(actor_params)
    # critic1.load_state_dict(critic_params)
    model.load_state_dict(model_params)

    # get initial state
    state, info = env.reset()

    # set up replay memory
    # memory = Memory()
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
        t_sim += 1
        
        # state to tensor
        state = torch.from_numpy(state)
        # get network outputs on given state
        value, policy = model(state.unsqueeze(0))
        # val_spk_lst.append(value_spike)
        # value = np.dot(val_spk_lst.detach(), weights_vals) / weights_sum
        # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)
        # print(prob)
        logprob = F.log_softmax(policy, dim=-1)

        # calculate entropy
        entropy = -(logprob * prob).sum(1, keepdim=True)
        entropies.append(entropy)

        # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
        # print(action)
        # print(value, action, policy)
        # get log probs of the actions taken
        log_prob = logprob.gather(1,action)
        
        # perform action
        obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))
        reward = max(min(reward, 1), -1)

        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)
        # transition = Transition(reward, log_prob, entropy, value)
        
        if not terminal:
            # memory.appendleft(transition)
            state = obs
    time_finish = t_sim   
    # plt.clf()
    # plt.plot(time_finish)
    # plt.pause(.001)
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())

    # print('MaxReward: ', max(0,reward))
    R = 0
    if not terminal:
        R, _ = model(torch.from_numpy(state))
        R = R.detach()
    # save current R (value) for the gneralized advantage DeltaT
    prev_val = R
    values.append(R)
    # on github code, they append this to the transitions, 
    # however, they substract values again to calculate advantage, 
    # does this mean they basically neglect this step for the advantage function?
    # my intuition: advantage tells how much better reward was than expected
    # therefore, you have to substract the value from the reward of the correct step.
    
    # print('Optimizing agents and critics...\n\n')
    policy_loss = 0
    value_loss  = 0
    g = torch.zeros(1,1)
    len_counter = 0
    # note cycle through memory sequentially as we are appending left
    for i in reversed(range(len(rewards))):

        # reward = torch.tensor(getattr(trans,'reward'))
        # logprob = torch.tensor(getattr(trans,'logprob')) # depends on actor
        # value  = torch.tensor(getattr(trans,'value'))
        # entropy  = torch.tensor(getattr(trans,'entropy')) # depends on critic

        
        R = rewards[i] + GAMMA*R

        # accumulate gradients wrt thread specific theta
        advantage = R - values[i] # note that according to the git they are out of sync by one step 
        # # detach from graph for use in actor updates
        # advantage_no_grad = advantage.clone().detach()
        
        # this loss should ONLY influence actor
        # policy_loss += torch.tensor(torch.log(action)*advantage_no_grad, requires_grad=True) # inplace doesnt allow gradient -> calculate backward pass and then 


    #     # this if from the paper: 
    #     # question: doesn't this iterative adding make steps too big? all the gradients are applied but they were all calculated to the same initial net? 
    #     # every step now changes the weights but this gradient is wrt the initial weights


        len_counter+=1

    # alternative training loop
        if len_counter>5:
            value_loss = value_loss + 0.5*advantage.pow(2)

            # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
            delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
            g = g*GAMMA*LAMBDA_G + delta_t

            policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF
        
        else:
            value_loss = 0
            policy_loss = 0


    optimizer.zero_grad()

    loss = policy_loss + value_loss * VALUE_LOSS_COEF 
    # loss = loss/len_counter # normalize?
    # print('\n\nLoss: ', loss)  
    loss.backward()
    # print('Backward pass completed!')
    # torch.nn.utils.clip_grad_norm_(model.parameters(), .max_grad_norm)

    optimizer.step()


    # equal the parameters of these nets to the global one
    # actor_global.load_state_dict(actor1.state_dict())
    # critic_global.load_state_dict(critic1.state_dict())


torch.save(model.state_dict(), 'model_mountaincart.txt')
plt.ioff()
plt.plot(range(len(T_lst)),T_lst)
plt.show()