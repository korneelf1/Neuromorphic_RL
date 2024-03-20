from actor_critics import  ActorCriticSNN_LIF_drone, ActorCriticSNN_LIF_SYN_drone, ActorCriticSNN_SYN_LIF_drone
from environments import SimpleDrone_Discrete

import torch

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from typing import Deque
env = SimpleDrone_Discrete(dt=0.02, max_episode_length=500, train_vel=True)
# env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'
model = ActorCriticSNN_LIF_drone(state_size, action_size, hidden1=128, hidden2=128).to(device)
# model = ActorCriticSNN_LIF_SYN_drone(state_size, action_size).to(device)
# model = ActorCriticSNN_SYN_LIF_drone(state_size, action_size).to(device)

loss_hist = []

num_iter = int(1e4) # train for x iterations

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# try learning velocity
# run a single forward-pass

with tqdm.trange(num_iter) as pbar:
    for _ in pbar:

        minibatch_counter = 0
        loss_epoch = []
        pos, true_vel = env.reset()[0]
        mem_lst = []
        true_vel_lst = []
        done = False
        model.init_mem()
        action_low_pass = Deque(maxlen=25)
        actions = []
        action_last=torch.tensor([0])
        while not done:
            pos = pos.to(device)
            true_vel = true_vel.to(device)
            # print(pos.shape, action_last.shape)
            prev_act_pos = torch.cat((action_last, pos))
            _,_,mem = model(pos)
            action_fluct = np.random.randint(0,7)
            action_low_pass.append(action_fluct)
            action = np.mean(action_low_pass) 
            action_last = torch.tensor([action])

            obs,_,done,_,_ = env.step(int(action))

            pos, true_vel = obs[0], obs[1]

            mem_lst.append(mem)
            true_vel_lst.append(true_vel)
            actions.append(action)

        loss_val = loss_function(torch.stack(mem_lst), torch.stack(true_vel_lst))
        optimizer.zero_grad() # zero out gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights

        # store loss
        loss_hist.append(loss_val.item())
        loss_epoch.append(loss_val.item())
        minibatch_counter += 1

        avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
        pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print lo_1e6ss p/batch

        # scheduler.step() # update learning rate

#save the model
torch.save(model.state_dict(), 'drone_snn_vel_syn_lif_1e4_128128.pt')

# plot

# mem_lst = np.array(mem_lst)
# true_vel_lst = np.array(true_vel_lst)
plt.subplot(3,1,1)

plt.plot(torch.stack(mem_lst).detach().numpy(), label="Output")
plt.plot(torch.stack(true_vel_lst).detach().numpy(), '--', label="Target")
plt.title("Velocity estimate, input: position ")
plt.xlabel("Time")
plt.ylabel("Membrane Potential")
# set log scale for y axis in the second subplot
plt.yscale('linear')
plt.legend(loc='best')
plt.subplot(3,1,2)
plt.plot(loss_hist, label="Loss  lif-lif random parameters")
plt.yscale('log')
plt.subplot(3,1,3)
plt.plot(actions, label="Actions")

plt.show()