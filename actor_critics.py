from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# spiking packages
import snntorch as snn
import snntorch.surrogate as surrogate

from collections import namedtuple, deque

from fast_sigmoid_module import fast_sigmoid

class ActorCriticSNN(torch.nn.Module):
    def __init__(self, num_inputs, action_space, value_window, inp_min = torch.tensor([-2.4, -3, -0.419/2, -1.5]), inp_max=  torch.tensor([2.4, 3, 0.419/2, 1.5]), alpha = 0.05, beta = 0.9, threshold = 1):
        super(ActorCriticSNN, self).__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        num_outputs = action_space.n

        ALPHA = alpha
        BETA  = beta
        THRESHOLD = threshold
        # Initialize surrogate gradient
        # self.spike_grad1 = surrogate.fast_sigmoid()  # passes default parameters from a closure
        # self.spike_grad2 = surrogate.FastSigmoid.apply  # passes default parameters, equivalent to above
        # self.spike_grad3 = surrogate.fast_sigmoid(slope=50)  # custom parameters from a closure

        # make this layer bigger
        # Try to encode all to positive values (use from normal layer to bigger layer with relu)

        self.cont_to_spike_layer = nn.Linear(num_inputs,100*num_inputs, bias=False) # vs scaled identity
        self.lif_in = snn.Synaptic(alpha=ALPHA,beta = .6, spike_grad=self.spike_grad, reset_mechanism = 'subtract', learn_beta = True, learn_alpha = True, threshold = THRESHOLD)

        self.layer1 = nn.Linear(100*num_inputs, 320, bias=False)
        # self.lif1 = snn.Leaky(beta = .95, spike_grad=self.spike_grad)
        self.sn1 = snn.Synaptic(alpha=ALPHA,beta = BETA, spike_grad=self.spike_grad, reset_mechanism = 'subtract', threshold = THRESHOLD)
        self.layer2 = nn.Linear(320, 320, bias=False)
        # self.lif2 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.sn2 = snn.Synaptic(alpha=ALPHA,beta = BETA, spike_grad=self.spike_grad, reset_mechanism = 'subtract' ,threshold = THRESHOLD)
        self.layer3 = nn.Linear(320, 100*num_outputs, bias=False)
        # self.lif3 = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        self.sn3 = snn.Synaptic(alpha=ALPHA,beta = BETA, spike_grad=self.spike_grad, reset_mechanism = 'subtract', threshold = THRESHOLD)

        # try to represent value as 100 neurons, sum spikes to have score /100
        self.critic_linear = nn.Linear(100*num_outputs, 1, bias=True) 
        # self.critic_linear = nn.Linear(32, 1) # 
        # self.lif_critic = snn.Leaky(beta=0.95,threshold=100, learn_beta=True, spike_grad=self.spike_grad)
        # self.value_mat  = nn.Linear(100,1)
        # nn.init.normal_(self.value_mat.weight, 2,1) # we want initialize value roughly 100 -> lets say average of connections should be 2 (if 50 percent of neurons spike we have 100)
        # self.value_mat = torch.ones(100, requires_grad=False)/100
        # Try same as value
        self.actor_linear = nn.Linear(100*num_outputs, num_outputs, bias=True)
        # self.actor_linear = nn.Linear(32, num_outputs)

        # self.lif_actor = snn.Leaky(beta=0.95, spike_grad=self.spike_grad)
        # self.sn_actor = snn.Synaptic(alpha=0.9,beta = .85, spike_grad=self.spike_grad)
        # self.action_mat = nn.Linear(100*num_outputs, num_outputs)
        # self.critic_linear = nn.Linear(100*num_outputs, 1) 

        # maje it more controlled
        # self.action_mat = torch.zeros(100*num_outputs,num_outputs, requires_grad=False) 
        # for row in range(num_outputs):
        #     self.action_mat[row*100:(row+1)*100,row] = torch.ones(100, requires_grad=False)/100
        # self.action_mat = self.action_mat.clone().detach().requires_grad_(True)
        self.apply(weights_init)
        
        # self.cont_to_spike_layer.weight.data = normalized_columns_initializer(
        #     self.cont_to_spike_layer.weight.data, 0.01)
        
        # self.layer1.weight.data = normalized_columns_initializer(
        #     self.layer1.weight.data, 0.7)
        # self.layer2.weight.data = normalized_columns_initializer(
        #     self.layer2.weight.data, 0.7)
        # self.layer3.weight.data = normalized_columns_initializer(
        #     self.layer3.weight.data, 0.7)
        # torch.nn.init.xavier_uniform_(self.layer1.weight, gain=2.5)
        # torch.nn.init.xavier_uniform_(self.layer2.weight, gain=3.5)
        # torch.nn.init.xavier_uniform_(self.layer3.weight, gain=4.5)
        torch.nn.init.constant_(self.layer1.weight, 500) # if neuron doesnt spike it will never backprop and never be used
        torch.nn.init.constant_(self.layer2.weight, 500)
        torch.nn.init.constant_(self.layer3.weight, 500)
        # self.cont_to_spike_layer.bias.data.fill_(0)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

       # membranes at t = 0
        self.syn_in, self.mem_in = self.lif_in.init_synaptic()
        self.syn1, self.mem1     = self.sn1.init_synaptic()
        self.syn2, self.mem2     = self.sn2.init_synaptic()
        self.syn3, self.mem3     = self.sn3.init_synaptic()
       
        # self.mem_critic = self.lif_critic.init_leaky()
        # self.mem_actor  = self.lif_actor.init_leaky()
        # self.syn_actor, self.mem_actor = self.sn_actor.init_synaptic()

        self.train()
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential
        self.syn_in_rec = []  # Record the output trace of membrane potential
        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        self.syn1_rec = []  # Record the output trace of membrane potential
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential
        self.syn2_rec = []  # Record the output trace of membrane potential
        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        self.syn3_rec = []  # Record the output trace of membrane potential

        self.critic_vals = deque([],value_window)
        self.spike_counter = 0

        # for normalization purpose
        self.inp_min = inp_min
        self.inp_max = inp_max
        self.scale_factor = nn.Parameter(torch.tensor(10.))  # Make scale_factor a learnable parameter
    def init_mem(self):
        self.syn_in, self.mem_in     = self.lif_in.init_synaptic()
        self.syn1,   self.mem1       = self.sn1.init_synaptic()
        self.syn2,   self.mem2       = self.sn2.init_synaptic()
        self.syn3,   self.mem3       = self.sn3.init_synaptic()


        self.spike_counter = 0
        
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential
        self.syn_in_rec = []  # Record the output trace of membrane potential
        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        self.syn1_rec = []  # Record the output trace of membrane potential
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential
        self.syn2_rec = []  # Record the output trace of membrane potential
        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        self.syn3_rec = []  # Record the output trace of membrane potential

        # self.mem_critic = self.lif_critic.init_leaky()
        # self.mem_actor  = self.lif_actor.init_leaky()
        # self.syn_actor, self.mem_actor = self.sn_actor.init_synaptic()

    # def normalize(self, inp):
    #     if torch.min(inp, dim=0)< self.inp_min:
    #         self.inp_min =torch.min(inp, dim=0)
        
    #     if torch.max(inp, dim=0)> self.inp_max:
    #         self.inp_max =torch.max(inp, dim=0)

    #     norm_inp = (inp - self.inp_min)/(self.inp_max-self.inp_min)
    #     return norm_inp
    
    def soft_population(self, inp,device = 'cpu',nr_inp = 4, neurons_dim = 1000, width = 15):
        '''applies soft_population encoding
        neurons_dim is number of neurons per input channel
        scale_factors is list of scales for every input
        width is width of affected neurons'''
        out_fion = torch.zeros((1,nr_inp*neurons_dim)).to(device)
        array = torch.tensor(range(neurons_dim), requires_grad = False).to(device)

        normal_inp = (inp-self.inp_min.to(device))/(self.inp_max.to(device)-self.inp_min.to(device))*neurons_dim
        for i in range(nr_inp):
            normal_fion = gaussian_function(array, normal_inp[0][i],width/2)* self.scale_factor* width
            out_fion[0][i*1000:((i+1)*1000)] = normal_fion

        return out_fion
        
    def forward(self, inputs, device=None, print_spikes = False, normalize_in = True, nr_passes = 1):
        for i in range(nr_passes):
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)
            # print(cur_in.size())
            inputs = inputs.to(torch.float32)

            # soft population encoding
            # cur_in = self.soft_population(inputs, device = device)
            # # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur_in = self.cont_to_spike_layer(inputs) # avoid negative incoming currents
            # cur_in = self.cont_to_spike_layer(inputs)
            # print(cur_in.size())

            # print(cur_in[:,0])
            spikes_in, self.syn_in, self.mem_in = self.lif_in(cur_in, self.syn_in, self.mem_in)
            # print(spikes_in)
            cur1 = (self.layer1(spikes_in))
            # print(cur1)
            # spk1, self.mem1 = self.lif1(cur1, self.mem1)
            spk1, self.syn1, self.mem1 = self.sn1(cur1, self.syn1, self.mem1)
            # print(spk1.shape, self.mem1.shape)
            cur2 = (self.layer2(spk1))
            # spk2, self.mem2 = self.lif2(cur2, self.mem2)
            spk2, self.syn2, self.mem2 = self.sn2(cur2, self.syn2, self.mem2)
            cur3 = (self.layer3(spk2))
            # spk3, self.mem3 = self.lif3(cur3, self.mem3)
            spk3, self.syn3, self.mem3 = self.sn3(cur3, self.syn3, self.mem3)
            
            actions = torch.sigmoid(self.actor_linear(spk3))

            # cur_critic = self.critic_linear(spk3)
            # # val = torch.sigmoid(cur_critic)
            # val = cur_critic
            # spk_actor, self.mem_actor = self.lif_actor(cur_actor, self.mem_actor)
            # spk_actor, self.syn_actor, self.mem_actor = self.sn_actor(cur_actor, self.syn_actor, self.mem_actor)
            val = (self.critic_linear(spk3))
            # actions = self.action_mat(spk_actor)

        if print_spikes:
            print(self.mem1, self.mem2, self.mem3)
            # plot_activity(spikes_in, spk1, spk2)

        self.spk_in_rec.append(spikes_in)
        self.mem_in_rec.append(self.mem_in)
        self.syn_in_rec.append(self.syn_in)
        self.spk1_rec.append(spk1)
        self.mem1_rec.append(self.mem1)
        self.syn1_rec.append(self.syn1)
        self.spk2_rec.append(spk2)
        self.mem2_rec.append(self.mem2)
        self.syn2_rec.append(self.syn2)
        self.spk3_rec.append(spk3)
        self.mem3_rec.append(self.mem3)
        self.syn3_rec.append(self.syn3)

        

        return val, actions
    

class ActorCriticSNN_LIF(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        num_outputs = action_space.n

        beta = 0.95
        self.nr_passes = nr_passes
        # make this layer bigger
        self.cont_to_spike_layer = nn.Linear(num_inputs,100*num_inputs,bias = bias) # vs scaled identity
        self.lif_in = snn.Leaky(beta = .95, spike_grad=self.spike_grad, learn_beta=True)

        self.layer1 = nn.Linear(100*num_inputs, 320,bias = bias)
        self.lif1 = snn.Leaky(beta = beta, spike_grad=self.spike_grad)

        self.layer2 = nn.Linear(320,320,bias = bias)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)

        self.layer3 = nn.Linear(320, 100*num_outputs,bias = bias)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)

        # try to represent value as 100 neurons, sum spikes to have score /100
        # bias might be useful in last non spiking layers
        self.critic_linear = nn.Linear(100*num_outputs, 1,bias = True) 

        # Try same as value
        self.actor_linear = nn.Linear(100*num_outputs, num_outputs,bias = True)
     
       # membranes at t = 0
        self.mem_in   = self.lif_in.init_leaky()
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem3     = self.lif3.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.mem_in   = self.lif_in.init_leaky()
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem3     = self.lif3.init_leaky()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        for i in range(self.nr_passes):
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)

            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur_in = self.cont_to_spike_layer(inputs)
            spikes_in, self.mem_in = self.lif_in(cur_in, self.mem_in)

            cur1 = self.layer1(spikes_in)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.layer2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            cur3 = self.layer3(spk1)
            spk3, self.mem3 = self.lif3(cur3, self.mem3)

        actions =  self.actor_linear(spk3)
        
        val = self.critic_linear(spk3)
        
        # add information for plotting purposes
        self.spk_in_rec.append(spikes_in)  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem_in)  # Record the output trace of membrane potential

        self.spk1_rec.append(spk1)  # Record the output trace of spikes
        self.mem1_rec.append(self.mem1)  # Record the output trace of membrane potential

        self.spk2_rec.append(spk2)  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2)  # Record the output trace of membrane potential

        self.spk3_rec.append(spk3)  # Record the output trace of spikes
        self.mem3_rec.append(self.mem3)  # Record the output trace of membrane potential


        return val, actions
    
class ActorCriticSNN_LIF_Small(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Small, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        num_outputs = action_space.n

        beta = 0.95
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, 320)
        self.lif1 = snn.Leaky(beta = .95, spike_grad=self.spike_grad, learn_beta=True)
        self.lin2 = nn.Linear(320, 320)
        self.lif2 = snn.Leaky(beta = beta, spike_grad=self.spike_grad, learn_beta=True)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(320, 1)
        self.actor_linear = nn.Linear(320, num_outputs)

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        for i in range(self.nr_passes):
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)

            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)



        actions =  self.actor_linear(spk1)
        
        val = self.critic_linear(spk1)
        
        # add information for plotting purposes
        self.spk_in_rec.append(spk1)  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1)  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2)  # Record the output trace of spikes
        self.mem1_rec.append(self.mem2)  # Record the output trace of membrane potential


        return val, actions
    
    
class ActorCriticSNN_SYN_Small(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_SYN_Small, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        num_outputs = action_space.n

        beta = 0.95
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, 320)
        self.syn1 = snn.Synaptic(alpha = 0.6,beta = .95, spike_grad=self.spike_grad, learn_beta=True)
        self.lin2 = nn.Linear(320, 320)
        self.syn2 = snn.Synaptic(alpha= 0.6,beta = beta, spike_grad=self.spike_grad, learn_beta=True)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(320, 1)
        self.actor_linear = nn.Linear(320, num_outputs)

     
       # membranes at t = 0
        self.mem1,self.syncur1     = self.syn1.init_synaptic()
        self.mem2,self.syncur2     = self.syn2.init_synaptic()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.mem1, self.syncur1     = self.syn1.init_synaptic()
        self.mem2, self.syncur2     = self.syn2.init_synaptic()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        for i in range(self.nr_passes):
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)

            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1, self.syncur1 = self.syn1(cur1, self.mem1,self.syncur1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2,self.syncur2 = self.syn2(cur2, self.mem2,self.syncur2)



        actions =  self.actor_linear(spk1)
        
        val = self.critic_linear(spk1)
        
        # add information for plotting purposes
        self.spk_in_rec.append(spk1)  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1)  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2)  # Record the output trace of spikes
        self.mem1_rec.append(self.mem2)  # Record the output trace of membrane potential


        return val, actions
    

class ActorCritic_ANN(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic_ANN, self).__init__()
        self.lin1 = nn.Linear(num_inputs, 320)
        self.lin2 = nn.Linear(320, 320)
 

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(320, 1)
        self.actor_linear = nn.Linear(320, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)

        x = F.elu(self.lin1(inputs))
        x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.actor_linear(x)
