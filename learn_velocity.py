from actor_critics import  ActorCriticSNN_LIF_drone, ActorCriticSNN_LIF_SYN_drone, ActorCriticSNN_SYN_LIF_drone, ActorCriticSNN_LIF_withbuffer
from environments import SimpleDrone_Discrete

import torch

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from typing import Deque
from torch.utils.data import DataLoader, TensorDataset

TRAIN = False
GATHER_DATA = True
INTERACTION_LENGTH = 200
env = SimpleDrone_Discrete(dt=0.02, max_episode_length=500, train_vel=True)
# env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = 'cpu'
model = ActorCriticSNN_LIF_drone(state_size, action_size, hidden1=32, hidden2=32, inp_min=torch.tensor([0], device=device),inp_max=torch.tensor([2.5], device=device)).to(device)
model = ActorCriticSNN_LIF_withbuffer(state_size, action_size, hidden1=32, hidden2=32, inp_min=torch.tensor([0], device=device),inp_max=torch.tensor([2.5], device=device), device=device).to(device)
# model = ActorCriticSNN_LIF_SYN_drone(state_size, action_size).to(device)
# model = ActorCriticSNN_SYN_LIF_drone(state_size, action_size).to(device)

  
print('I am learning actions or something i think, velocity is not smooth at all...')
loss_hist = []

num_iter = int(2e3) # train for x iterations

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# try learning velocity
# run a single forward-pass
dataset_observations = []
dataset_velocities = []

with tqdm.trange(num_iter) as pbar:
    for _ in pbar:

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
        action_last=torch.tensor([0], device=device)
        while not done and len(obs_lst) < INTERACTION_LENGTH:
            pos = pos.to(device)
            true_vel = true_vel.to(device)
            # print(pos.shape, action_last.shape)
            prev_act_pos = torch.cat((action_last, pos))
            # _,_,mem = model(pos)
            action_fluct = np.random.randint(0,7)
            action_low_pass.append(action_fluct)
            action = np.mean(action_low_pass) 
            action_last = torch.tensor([action],device=device)

            obs,_,done,_,_ = env.step(int(action))

            pos, true_vel = obs[0], obs[1]

            # mem_lst.append(mem)
            true_vel_lst.append(true_vel.to(device))
            obs_lst.append(pos.to(device))
            actions.append(action)

        if TRAIN:
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
        elif GATHER_DATA:
            if done:
                print(len(obs_lst))
                continue # the interacion has different length and cant be used for training
            dataset_velocities.append(torch.tensor(true_vel_lst,device = device))
            dataset_observations.append(torch.tensor(obs_lst, device=device))
        # scheduler.step() # update learning rate

obs_tensor = torch.stack(dataset_observations)

vel_tensor = torch.stack(dataset_velocities)
# Create a TensorDataset
dataset = TensorDataset(obs_tensor, vel_tensor)

# Create a DataLoader
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train model on this dataset
if GATHER_DATA:
    for epoch in range(20):
        epoch_avg_loss = 0
        for batch in dataloader:
            # Process the batch
            # Your code here
            obs, vel = batch
            # print(obs.shape, vel.shape)
            obs = obs.swapaxes(0,1)
            vel = vel.swapaxes(0,1)
            mem_lst = []
            true_vel_lst = []
            loss_epoch = []
            minibatch_counter = 0
            model.init_mem()
            mem_lst = []
            for i in range(obs.shape[0]):
                pos = obs[i].unsqueeze(-1)
                # true_vel = vel[i]
                # print(pos.shape)
                _,_,mem = model(pos)
                mem_lst.append(mem)

            # print(torch.stack(mem_lst).shape, vel.shape)
            loss_val = loss_function(torch.stack(mem_lst), vel.unsqueeze(-1))
            optimizer.zero_grad() # zero out gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights

        # store loss
        loss_hist.append(loss_val.item())
        loss_epoch.append(loss_val.item())
        minibatch_counter += 1

        avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
        pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print lo_1e6ss p/batch

        print(avg_batch_loss)
    # scheduler.step() # update learning rate

torch.save(model.state_dict(), 'drone_snn_vel_syn_lif_1e4_3232_with_buffer.pt')



# plot mem_lst and true_vel_lst to see if the model is learning
mem_lst  = torch.stack(mem_lst)[:,0].to('cpu').detach().numpy()
true_vel_lst =  vel.unsqueeze(-1)[:,0].to('cpu').detach().numpy()
plt.plot(mem_lst)
plt.plot(true_vel_lst)
plt.show()

# scheduler.step() # update learning rate

plt.subplot(3,1,1)

plt.plot(torch.stack(mem_lst).to('cpu').detach().numpy(), label="Output")
plt.plot(torch.stack(true_vel_lst).to('cpu').detach().numpy(), '--', label="Target")
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