from environments import SimpleDrone_Discrete
from actor_critics import ActorCritic_ANN,ActorCriticSNN_LIF_drone
import torch
import torch.nn.functional as F
from typing import Deque
import numpy as np
import matplotlib.pyplot as plt

env = SimpleDrone_Discrete(dt=0.02, max_episode_length=500, train_vel=True)

state_size = env.observation_space.shape[0]
action_size = env.action_space
model = ActorCriticSNN_LIF_drone(state_size, action_size, hidden1=32, hidden2=32, inp_min=torch.tensor([0]),inp_max=torch.tensor([2.5]))
torch.load('drone_snn_vel_syn_lif_1e4_3232.pt')


for _ in range(1):

    minibatch_counter = 0
    loss_epoch = []
    pos, true_vel = env.reset()[0]
    mem_lst = []
    true_vel_lst = []
    obs_lst =[]

    done = False
    model.init_mem()
    action_low_pass = Deque(maxlen=25)
    actions = []
    action_last=torch.tensor([0])
    while not done and len(obs_lst) < 200:
        # pos = pos.to(device)
        # true_vel = true_vel.to(device)
        # print(pos.shape, action_last.shape)
        prev_act_pos = torch.cat((action_last, pos))
        _,_,mem = model(pos)
        action_fluct = np.random.randint(0,7)
        action_low_pass.append(action_fluct)
        action = np.mean(action_low_pass) 
        action_last = torch.tensor([action])

        obs,_,done,_,_ = env.step(int(action))

        pos, true_vel = obs[0], obs[1]

        # mem_lst.append(mem)
        true_vel_lst.append(true_vel)
        obs_lst.append(pos)
        mem_lst.append(mem)
        actions.append(action)
        print(len(obs_lst))


# plot mem_lst and true_vel_lst to see if the model is learning
mem_lst  = torch.stack(mem_lst).detach().numpy()
true_vel_lst = torch.stack(true_vel_lst).detach().numpy()
plt.plot(mem_lst)
plt.plot(true_vel_lst)
plt.show()

    # scheduler.step() # update learning rate
