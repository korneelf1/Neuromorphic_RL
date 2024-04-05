import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import snntorch
from snntorch import surrogate


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
            # if torch.isinf(state_batch[:,i,:]).all():
            #     # find way to mask the infinity entries but not the other ones
            #     # raise Exception('Infinite value in state batch detected')
            #     # return torch.zeros((state_batch.shape[0], state_batch.shape[1], self.n_actions))
            #     values.append(torch.zeros((state_batch.shape[0], self.n_actions)))
        # x = F.relu(self.layer1(x))
            x = F.relu(self.layer1(state_batch[:,i,:]))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            values.append(x)
        # return x
        return torch.stack(values, dim=1)
        
class DSQN(nn.Module):
    def __init__(self, n_observations, n_actions, bias=True, BATCH_SIZE=1, device='cpu'):
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

        self.batch_size = BATCH_SIZE
        self.device = device
        

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def step_forward(self, x):
        x = self.layer1(x)
        spk1, self.mem1 = self.lif1(x,self.mem1)
        x = self.layer2(spk1)
        spk2, self.mem2 = self.lif2(x,self.mem2)
        x = self.layer3(spk2)
        # torch.cat([self.mem1,self.mem2], dim=1)
        return x, torch.cat([self.mem1,self.mem2], dim=1).squeeze(0)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_batch, hidden_states=None):
        '''
        x shape: (batch_size, observation_length, feature_size)'''
        if hidden_states is None:
            hidden_states = torch.zeros((self.batch_size,1,2,128), device=self.device)
        self.reset()
        self.mem1 = hidden_states[:,0,0,:]
        self.mem2 = hidden_states[:,0,1,:]
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
