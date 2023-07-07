# this file creates the agent class used for A3C
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

# spiking packages
import snntorch as snn
from snntorch import utils
from torchvision import transforms
from snntorch import surrogate
from snntorch import functional

from collections import namedtuple, deque
class ActorCriticSNN(torch.nn.Module):
    def __init__(self, num_inputs, action_space, value_window):
        super(ActorCriticSNN, self).__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=25)

        self.cont_to_spike_layer = nn.Linear(num_inputs,num_inputs)
        self.lif_in = snn.Leaky(beta = .65, spike_grad=self.spike_grad, threshold=.25)

        self.layer1 = nn.Linear(num_inputs, 32)
        self.lif1 = snn.Leaky(beta = .65, spike_grad=self.spike_grad, threshold=.25)
        self.layer2 = nn.Linear(32, 32)
        self.lif2 = snn.Leaky(beta=0.85, spike_grad=self.spike_grad)
        self.layer3 = nn.Linear(32, 32)
        self.lif3 = snn.Leaky(beta=0.85, spike_grad=self.spike_grad)
        self.layer4 = nn.Linear(32, 32)
        self.lif4 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.layer5 = nn.Linear(32, 32)
        self.lif5 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.layer6 = nn.Linear(32, 32)
        self.lif6 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.layer7 = nn.Linear(32, 32)
        self.lif7 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        

        num_outputs = action_space.n
        # try to represent value as 100 neurons, sum spikes to have score /100
        # self.critic_linear = nn.Linear(32, 100)
        self.critic_linear = nn.Linear(32, 1)
        self.lif_critic = snn.Leaky(beta=0.95,threshold=100, learn_beta=True, spike_grad=self.spike_grad)
        self.value_mat = torch.ones(100, requires_grad=False)/100
        # Try same as value
        self.actor_linear = nn.Linear(32, 100*num_outputs)
        # self.actor_linear = nn.Linear(32, num_outputs)

        self.lif_actor = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.action_mat = torch.zeros(100*num_outputs,num_outputs, requires_grad=False) 
        for row in range(num_outputs):
            self.action_mat[row*100:(row+1)*100,row] = torch.ones(100, requires_grad=False)/100


        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

       # membranes at t = 0
        self.mem_in     = self.lif_in.init_leaky()
        self.mem1       = self.lif1.init_leaky()
        self.mem2       = self.lif2.init_leaky()
        self.mem3       = self.lif3.init_leaky()
        self.mem4       = self.lif4.init_leaky()
        self.mem5       = self.lif5.init_leaky()
        self.mem6       = self.lif6.init_leaky()
        self.mem7       = self.lif7.init_leaky()
        self.mem_critic = self.lif_critic.init_leaky()
        self.mem_actor  = self.lif_actor.init_leaky()

        self.train()

        self.critic_vals = deque([],value_window)
    def init_mem(self):
        self.mem_in     = self.lif_in.init_leaky()
        self.mem1       = self.lif1.init_leaky()
        self.mem2       = self.lif2.init_leaky()
        self.mem3       = self.lif3.init_leaky()
        self.mem4       = self.lif4.init_leaky()
        self.mem5       = self.lif5.init_leaky()
        self.mem6       = self.lif6.init_leaky()
        self.mem7       = self.lif7.init_leaky()
        self.mem_critic = self.lif_critic.init_leaky()
        self.mem_actor  = self.lif_actor.init_leaky()


    def forward(self, inputs):
        # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
        cur_in = self.cont_to_spike_layer(inputs)
        spikes_in, self.mem_in = self.lif_in(cur_in, self.mem_in)

        cur1 = self.layer1(spikes_in)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        # print('Spikes in layer 1:\n', spk1)
        cur2 = self.layer2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)
        # print('Spikes in layer 2:\n', spk2)'
        cur3 = self.layer3(spk2)
        spk3, self.mem3 = self.lif3(cur3, self.mem3)

        cur4 = self.layer4(spk3)
        spk4, self.mem4 = self.lif4(cur4, self.mem4)

        cur5 = self.layer5(spk4)
        spk5, self.mem5 = self.lif5(cur5, self.mem5)

        cur6 = self.layer6(spk5)
        spk6, self.mem6 = self.lif6(cur6, self.mem6)

        cur7 = self.layer7(spk6)
        spk7, self.mem7 = self.lif7(cur7, self.mem7)
        # simplest output for actor is spike in node 1 is action 1
        cur_actor = self.actor_linear(spk7)
        cur_critic = self.critic_linear(spk7)

        spk_actor, self.mem_actor = self.lif_actor(cur_actor, self.mem_actor)
        spk_critic, self.mem_critic = self.lif_critic(cur_critic,self.mem_critic)
        # val = poisson_decode(spk_critic, 10)
        # val = torch.sum(spk_critic)/100
        val = self.mem_critic
        actions = torch.matmul(spk_actor,self.action_mat)

        # actions = spk_actor
        # self.critic_vals.append(spk_critic)
        # norm_weight = 0
        # value = 0
        # for i in range(len(self.critic_vals)):
        #     norm_weight += i
        #     value += self.critic_vals[i]*i
        # value = value/norm_weight

        return val, actions
    
class Feedforward(nn.Module):

    def __init__(self, nr_input_nodes, nr_output_nodes, layers, size_hiddenlayers):
        super(Feedforward,self).__init__()
        '''Agent is a simple FC FF net with a number of layers, which can be increased to represent more complex functions. A minimum of two layers is possible.'''
        fc1 = nn.Linear(nr_input_nodes, size_hiddenlayers)
        relu = nn.ReLU()
        
        # create modulelist
        self.modulelist = nn.ModuleList([fc1, relu])
        for _ in range(layers-2):
            self.modulelist.append(nn.Linear(size_hiddenlayers, size_hiddenlayers))
            self.modulelist.append(relu)

        fc_final = nn.Linear(size_hiddenlayers, nr_output_nodes)
        self.modulelist.append(fc_final)
  

    def forward(self, x):
        x = torch.tensor(x)
        for layer in self.modulelist:
            x = layer(x)

        return x

class A3Cnet(nn.Module):

    def __init__(self, nr_input_nodes, nr_output_nodes_actor, nr_output_nodes_critic, layers, size_hiddenlayers):
        super(A3Cnet,self).__init__()
        '''Agent is a simple FC FF net with a number of layers, which can be increased to represent more complex functions. A minimum of two layers is possible.'''
        fc1 = nn.Linear(nr_input_nodes, size_hiddenlayers)
        relu = nn.ReLU()
        
        # create modulelist
        self.modulelist = nn.ModuleList([fc1, relu])
        for _ in range(layers-2):
            self.modulelist.append(nn.Linear(size_hiddenlayers, size_hiddenlayers))
            self.modulelist.append(relu)

        self.fc_final_actor = nn.Linear(size_hiddenlayers, nr_output_nodes_actor)
        self.fc_final_critic =  nn.Linear(size_hiddenlayers, nr_output_nodes_critic)
        # self.modulelist.append(fc_final)
  

    def forward(self, x):
        '''Returns actor_out, critic_out'''
        x = torch.tensor(x)
        for layer in self.modulelist:
            x = layer(x)

        x_actor = self.fc_final_actor(x)
        x_critic = self.fc_final_critic(x)
        return x_actor, x_critic
Transition = namedtuple('transition',('reward', 'logprob', 'entropy','value'))

class Memory(object):
    '''Fifo memory'''
    def __init__(self):
        self.memory = deque([])

    def append(self, transition):
        self.memory.appendleft(transition)

    def reset(self):
        self.memory.clear()
        
    def __call__(self):
        return self.memory
    
    def __iter__(self):
        return self.memory
    
    # def reverse(self):
    #     return self.memory.reverse()
spike_tracker = []


min_rate = 0
max_rate = 10

min_value = 0
max_value = 5

t_max = 200
def poisson_decode(spike_in, step_window):
    global spike_tracker

    # estimate spike rate
    spike_tracker.append(spike_in)
    
    if len(spike_tracker) > step_window:
        spike_rate = sum(spike_tracker[len(spike_tracker)-step_window:-1])
        val_est = ((spike_rate-min_rate)/(max_rate - min_rate)*(max_value-min_value))/step_window*t_max + min_value
        return val_est
    else:
        return 0
    

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Linear(num_inputs, 32)
        self.conv2 = nn.Linear(32, 32)
        self.conv3 = nn.Linear(32, 32)
        self.conv4 = nn.Linear(32, 32)
        self.conv5 = nn.Linear(32, 32)
        self.conv6 = nn.Linear(32, 32)
        self.conv7 = nn.Linear(32, 32)

        # self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(32, 1)
        self.actor_linear = nn.Linear(32, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        # inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.elu(self.conv7(x))

        # x = x.view(-1, 32 * 3 * 3)
        # hx, cx = self.lstm(x, (hx, cx))
        # x = hx

        return self.critic_linear(x), self.actor_linear(x)

if __name__=='__main__':
    agent = Feedforward(2,3,3,2)
    input_test = torch.tensor([2.,3.])
    print(agent(input_test))