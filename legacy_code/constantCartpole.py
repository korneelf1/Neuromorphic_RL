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
from CartPole_modified import CartPole_fake
# training environment
# import gym

# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v1', render_mode="human")
# env = gym.make("CartPole-v1")
env = CartPole_fake()
constant_state, info = env.reset()
print('\nCONSTANT STATE:\n'+str(constant_state)+'\n\n')
# init plt
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()
# plt.figure()
# plt.xlabel('Episode')
# plt.ylabel('Duration')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
show_result = True
# global parameters
T = 0
th = 0
th_v = 0
GAMMA = .9
LAMBDA_G = .9
ENTROPY_COEF = .01
VALUE_LOSS_COEF = 1.4
LEARNING_RATE = 1e-4

T_max = 100000 
# T_max = 10
t_sim_max = 200
T_SNN_VAL_WINDOW = 5


model = ActorCriticSNN(4, env.action_space, T_SNN_VAL_WINDOW).to(device)
# model = ActorCritic(4, env.action_space).to(device)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss = 0
time_finish = 0
T_lst = []
loss_lst = []
print('Lets try to learn one iteration')
while T<T_max:
    model.init_mem()
    T +=1
    
    print('Run: ',T, '\t time: ',time_finish)
    T_lst.append(time_finish)
    
    model_params  = model.state_dict()

    # create thread specific actors and critics (assume 4 threads for now)
   
    model.load_state_dict(model_params)

    # get initial state
    state = env.set_state(constant_state)
    # print(state)
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
        state = torch.from_numpy(state).to(device)
        # get network outputs on given state
        value, policy = model(state.unsqueeze(0))
        # if T%1000 ==0 or T==1:
        # print('value:',value,'\tpolicy: ', policy)
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

        # !!!!!!!!!!!
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
        R, _ = model(torch.from_numpy(state).to(device))
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
    g = torch.zeros(1,1).to(device)
    len_counter = 0
    # note cycle through memory sequentially as we are appending left
    for i in reversed(range(len(rewards))):
        len_counter+=1

        if len_counter>-1:

        # reward = torch.tensor(getattr(trans,'reward'))
        # logprob = torch.tensor(getattr(trans,'logprob')) # depends on actor
        # value  = torch.tensor(getattr(trans,'value'))
        # entropy  = torch.tensor(getattr(trans,'entropy')) # depends on critic

        
            R = rewards[i] + GAMMA*R

        # accumulate gradients wrt thread specific theta
            advantage = R - values[i] # note that according to the git they are out of sync by one step 
            # if (T%500 == 0) or (T == 1):
            #     print(R, values[i])
        # # detach from graph for use in actor updates
        # advantage_no_grad = advantage.clone().detach()
        
        # this loss should ONLY influence actor
        # policy_loss += torch.tensor(torch.log(action)*advantage_no_grad, requires_grad=True) # inplace doesnt allow gradient -> calculate backward pass and then 


    #     # this if from the paper: 
    #     # question: doesn't this iterative adding make steps too big? all the gradients are applied but they were all calculated to the same initial net? 
    #     # every step now changes the weights but this gradient is wrt the initial weights



    # alternative training loop
            value_loss = value_loss + 0.5*advantage.pow(2)

            # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
            delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
            g = g*GAMMA*LAMBDA_G + delta_t

            policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF
            if (T%500 == 0) or (T == 1):
                print(R, values[i])
           
        else:
            value_loss = 0
            policy_loss = 0

    # if (T%500 == 0) or (T == 1):
    #     print('Policy loss: ', policy_loss)
    #     print('Value loss: ', value_loss)
    optimizer.zero_grad()

    loss = (policy_loss + value_loss * VALUE_LOSS_COEF )
    loss_lst.append(loss.to('cpu').detach().squeeze(0))
    if not terminal:
        print(t_sim, loss)
    # loss = loss/len_counter # normalize?
    # print('\n\nLoss: ', loss)  
    loss.backward()
    # print('Backward pass completed!')
    # torch.nn.utils.clip_grad_norm_(model.parameters(), .max_grad_norm)

    optimizer.step()
    # print(model.action_mat)

    # equal the parameters of these nets to the global one
    # actor_global.load_state_dict(actor1.state_dict())
    # critic_global.load_state_dict(critic1.state_dict())


torch.save(model.state_dict(), 'model_mountaincart.txt')
plt.ioff()
plt.figure()
plt.title('times')
plt.plot(range(len(T_lst)),T_lst)
# print(loss_lst)
plt.figure()
plt.title('losses')
plt.plot(range(len(loss_lst)),loss_lst)
plt.show()