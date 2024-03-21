import gymnasium as gym
import numpy as np
import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn

from snntorch import functional as SF
from snntorch import utils

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import itertools
import random
import statistics
import tqdm
from matplotlib import pyplot as plt

# Seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# Create a simple SNN model
class SNN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(SNN, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
   
        beta = 0.95

        self.lin1 = nn.Linear(state_size, 64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=self)
        
        self.lin2 = nn.Linear(64, action_size)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism='none',spike_grad=self)
        
        self.init_mem()

    def forward(self, x):
        x = self.lin1(x)
        x, self.mem1 = self.lif1(x, self.mem1)
        x = self.lin2(x)
        x, self.mem2 = self.lif2(x, self.mem2)
        return x

    def init_mem(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        
class Net(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden):
        super().__init__()

        self.timesteps = timesteps # number of time steps to simulate the network
        self.hidden = hidden # number of hidden neurons
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

        # randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # layer 1
        self.fc_in = torch.nn.Linear(in_features=4, out_features=self.hidden,bias=False)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(self.hidden)
        thr_hidden = torch.rand(self.hidden)

        # layer 2
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden,bias=False)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=2,bias=False)
        self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")


    def init_mem(self):
        self.mem_1 = self.lif_in.init_leaky()
        self.mem_2 = self.lif_hidden.init_leaky()
        self.mem_3 = self.li_out.init_leaky()


    def forward_single(self, x):
        """Forward pass for a single time step."""

        # layer 1
        cur_in = self.fc_in(x)
        spk_in, mem_1 = self.lif_in(cur_in, self.mem_1)

        # layer 2
        cur_hidden = self.fc_hidden(spk_in)
        spk_hidden, mem_2 = self.lif_hidden(cur_hidden, self.mem_2)

        # layer 3
        cur_out = self.fc_out(spk_hidden)
        _, mem_3 = self.li_out(cur_out, self.mem_3)

        return F.softmax(mem_3, dim=-1)
    

    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()

        # Empty lists to record outputs
        mem_3_rec = []
        probs = []
        seq_length = x.shape[0]
        # Loop over
        
        for step in range(seq_length):
            x_timestep = x[step, :, :]

            cur_in = self.fc_in(x_timestep)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            cur_out = self.fc_out(spk_hidden)
            _, mem_3 = self.li_out(cur_out, mem_3)
            
            prob = F.softmax(mem_3, dim=-1)

            # logprob = F.log_softmax(mem_3, dim=-1)

            # choose the action and detach from computational graph
            # probs = prob.multinomial(num_samples=1).detach()

            mem_3_rec.append(mem_3)
            probs.append(prob)

        return torch.stack(probs)

env = gym.make("CartPole-v1")

def eval_agent(agent, iterations):
    # evaluate with gym interaction
    env.reset()

    times = []
    for i in range(int(iterations)):
        agent.init_mem()
        state, _ = env.reset()
        done = False
        j=0
        while not done and j <1000:
            

            prob = agent.forward_single(torch.tensor(state).to(device).unsqueeze(0))
                # find probabilities of certain actions
            # prob = F.softmax(policy, dim=-1)



                # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
            
            state, reward, done, truncated, info = env.step(action.item())
            j+=1

        # print(f"Episode {i} finished, number of steps taken: {j}")
        times.append(j)

    # print(f"Average number of steps taken: {sum(times)/len(times)}")
    return sum(times)/len(times)

# from snnTorch
class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list

        # generate linear functions one by one
        for idx in range(num_samples):
            end = float(torch.rand(1)) # random final point
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            feature_lst.append(feature) # add sample to list

        self.features = torch.stack(feature_lst, dim=1) # convert list to tensor

        # option to generate linear function or square-root function
        if mode == "linear":
            self.labels = self.features * 1

        elif mode == "sqrt":
            slope = float(torch.rand(1))
            self.labels = torch.sqrt(self.features * slope)

        elif mode == "moving_avg":
            self.labels = torch.zeros_like(self.features)
            for i in range(10, timesteps):
                self.labels[i] = torch.mean(self.features[i-10:i], dim=0)**2

        else:
            raise NotImplementedError("'linear', 'sqrt'")

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        return self.features[:, idx, :], self.labels[:, idx, :]
    

num_steps = 500
num_samples = 1
mode = "moving_avg" # 'linear' or 'sqrt'

# Create a simple SNN model
hidden = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = Net(timesteps=num_steps, hidden=hidden).to(device)


# generate a single data sample
# dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)

# load sequences from cartpole
datatensor = torch.load('experiments/dataset.pt')
# print(dataset.shape)

# features = dataset[:, :, 0]
# labels = dataset[:, :, 1]

class CartPoleDataset(torch.utils.data.Dataset):
    def __init__(self, datatensor):
        self.datatensor = datatensor
        self.features = datatensor[:, :500, :4].swapaxes(0, 1)
        labels = datatensor[:, :500, 4:].swapaxes(0, 1)
        placeholder = torch.zeros((labels.shape[0], labels.shape[1], 2))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j, 0] < 0.5:
                    placeholder[i, j, 0] = 1
                else:
                    placeholder[i, j, 1] = 1
        self.labels = placeholder

    def __len__(self):
        return self.datatensor.shape[0]
    def __getitem__(self, idx):
        return self.features[ :,idx, :], self.labels[:,idx, :]
dataset = CartPoleDataset(datatensor)
# plot
sample = dataset.labels[:, 0, 0]


batch_size = 10 # only one sample to learn
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)
train_batch = iter(dataloader)

# run a single forward-pass
with torch.no_grad():
    for feature, label in train_batch:
        print(feature.shape, label.shape)
        feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
        label = torch.swapaxes(input=label, axis0=0, axis1=1)
        feature = feature.to(device)
        label = label.to(device)
        mem = model(feature)


num_iter = 500 # train for 500 iterations
eval_interval = 50 # evaluate every 100 iterations
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

loss_hist = [] # record loss

n_sequences = 1
sequence_length = 500
print('Number of sequences:', n_sequences)
# training loop
printed = False
counter = 0
performance_lst = []
with tqdm.trange(num_iter) as pbar:
    
    for _ in pbar:
        counter+=1
        if counter % eval_interval == 0:
            performance_lst.append(eval_agent(model, 3))

        train_batch = iter(dataloader)
        minibatch_counter = 0
        loss_epoch = []

        for feature, label in train_batch:
            # prepare data
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)

            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)

            # forward pass
            mem = model(feature.reshape(sequence_length,-1,4))
            label = label.reshape(sequence_length,-1,2) # now no burn in period
            loss_val = loss_function(mem, label) # calculate loss
            if not printed:
                print(feature.reshape(sequence_length,-1,4).shape)
                printed = True
            optimizer.zero_grad() # zero out gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights

            # store loss
            loss_hist.append(loss_val.item())
            loss_epoch.append(loss_val.item())
            minibatch_counter += 1

            avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
            pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print loss p/batch
    
loss_function = torch.nn.L1Loss() # Use L1 loss instead
print('Finished training')
print(performance_lst)
 # pause gradient calculation during evaluation
with torch.no_grad():
    model.eval()

    test_batch = iter(dataloader)
    minibatch_counter = 0
    rel_err_lst = []

    # loop over data samples
    for feature, label in test_batch:

        # prepare data
        feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
        label = torch.swapaxes(input=label, axis0=0, axis1=1)
        feature = feature.to(device)
        label = label.to(device)

        # forward-pass
        mem = model(feature)

        # calculate relative error
        rel_err = torch.linalg.norm(
            (mem - label), dim=-1
        ) / torch.linalg.norm(label, dim=-1)
        rel_err = torch.mean(rel_err[1:, :])

        # calculate loss
        loss_val = loss_function(mem, label)

        # store loss
        loss_hist.append(loss_val.item())
        rel_err_lst.append(rel_err.item())
        minibatch_counter += 1

    mean_L1 = statistics.mean(loss_hist)
    mean_rel = statistics.mean(rel_err_lst)

print(f"{'Mean L1-loss:':<{20}}{mean_L1:1.2e}")
print(f"{'Mean rel. err.:':<{20}}{mean_rel:1.2e}")


# print the betas of the hidden layers
print('Betas of the hidden layers')
print(torch.mean(model.lif_in.beta))
print(torch.mean(model.lif_hidden.beta))
print(torch.mean(model.li_out.beta))

# plot performance_lst
plt.figure()
plt.plot(performance_lst)
plt.title("Performance")
plt.xlabel("Evaluation")
plt.ylabel("Average steps")
plt.show()


