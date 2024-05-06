from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# spiking packages
import snntorch as snn
import snntorch.surrogate as surrogate

from collections import namedtuple, deque

# from fast_sigmoid_module import fast_sigmoid
# import torch
'''
This manual implementation is required to be able to pickle 
the model for parallelization'''
def fast_sigmoid_forward(ctx, input_, slope):
    ctx.save_for_backward(input_)
    ctx.slope = slope
    out = (input_ > 0).float()
    return out

def fast_sigmoid_backward(ctx, grad_output):
    (input_,) = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
    return grad, None

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, slope=25):
        return fast_sigmoid_forward(ctx, input_, slope)

    @staticmethod
    def backward(ctx, grad_output):
        return fast_sigmoid_backward(ctx, grad_output)

def fast_sigmoid(slope=25):
    def inner(x):
        return FastSigmoid.apply(x, slope)
    return inner
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

        self.lin1 = nn.Linear(num_inputs, 128)
        self.lif1 = snn.Leaky(beta = .95, spike_grad=self.spike_grad, learn_beta=True)
        self.lin2 = nn.Linear(128, 128)
        self.lif2 = snn.Leaky(beta = beta, spike_grad=self.spike_grad, learn_beta=True)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
     
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
            inputs = torch.tensor(inputs).to(torch.float32)
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

class ActorCriticSNN_LIF_drone(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden1=64,hidden2=64, inp_min = torch.tensor([0]), inp_max=  torch.tensor([2.5]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_drone, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        self.num_outputs = action_space.n
        self.num_inputs = num_inputs
        beta = 0.95
        self.nr_passes = nr_passes

        # randomly initialize decay rate and threshold for layer 1
        beta_hidden = torch.rand(hidden1)
        thr_hidden = torch.rand(hidden1)

        self.lin1 = nn.Linear(num_inputs, hidden1)
        self.lif1 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(hidden2)
        thr_hidden = torch.rand(hidden2)

        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)
        # self.lif2 = snn.Synaptic(beta = .75, alpha = 0.5, spike_grad=self.spike_grad, learn_beta=False, learn_alpha=False)
        self.action_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        self.value_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_vel = torch.nn.Linear(in_features=hidden2, out_features=1)
        self.li_vel = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=self.spike_grad, reset_mechanism="none")
        self.critic_linear = nn.Linear(hidden2, 1)
        self.actor_linear = nn.Linear(hidden2, self.num_outputs)
     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []

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
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        
        self.inputs = []

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential

    def clip_hiddens(self):
        self.mem1 = torch.clamp(self.mem1, 0, 1)
        self.mem2 = torch.clamp(self.mem2, 0, 1)
        self.mem3 = torch.clamp(self.mem3, 0, 1)
        self.mem_act = torch.clamp(self.mem_act, 0, 1)
        self.mem_val = torch.clamp(self.mem_val, 0, 1)

    def forward(self, inputs, nr_passes = 1):
        for i in range(self.nr_passes):
            # inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)
            self.inputs.append(inputs)
            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
        
        actions =  self.actor_linear(spk2)
        _, self.mem_act = self.action_lif(actions, self.mem_act)
        
        val = self.critic_linear(spk2)
        _, self.mem_val = self.value_lif(val, self.mem_val)
        vel = self.fc_vel(spk2)
        _, self.mem3 = self.li_vel(vel, self.mem3)
        vel = self.mem3
        # add information for plotting purposes
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2.squeeze(0).detach())  # Record the output trace of spikes
        self.mem1_rec.append(actions.squeeze(0).detach())  # Record the output trace of membrane potential

        return self.mem_val.reshape((1,-1)), self.mem_act.reshape((1,-1)), vel
  
class ActorCriticSNN_LIF_drone_DSQN(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden1=64,hidden2=64, inp_min = torch.tensor([0]), inp_max=  torch.tensor([2.5]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_drone_DSQN, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        self.num_outputs = action_space.n
        self.num_inputs = num_inputs
        beta = 0.95
        self.nr_passes = nr_passes

        # randomly initialize decay rate and threshold for layer 1
        beta_hidden = torch.rand(hidden1)
        thr_hidden = torch.rand(hidden1)

        self.lin1 = nn.Linear(num_inputs, hidden1)
        self.lif1 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(hidden2)
        thr_hidden = torch.rand(hidden2)

        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)
        # self.lif2 = snn.Synaptic(beta = .75, alpha = 0.5, spike_grad=self.spike_grad, learn_beta=False, learn_alpha=False)
        self.action_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        self.value_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_vel = torch.nn.Linear(in_features=hidden2, out_features=1)
        self.li_vel = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=self.spike_grad, reset_mechanism="none")
        self.critic_linear = nn.Linear(hidden2, 1)
        self.actor_linear = nn.Linear(hidden2, self.num_outputs)
     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []

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
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        
        self.inputs = []

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential

    def reset(self):
        '''for compatibility reasons'''
        self.init_mem()
        
    def clip_hiddens(self):
        self.mem1 = torch.clamp(self.mem1, 0, 1)
        self.mem2 = torch.clamp(self.mem2, 0, 1)
        self.mem3 = torch.clamp(self.mem3, 0, 1)
        self.mem_act = torch.clamp(self.mem_act, 0, 1)
        self.mem_val = torch.clamp(self.mem_val, 0, 1)

    def forward(self, batch):
        self.init_mem()

        values = []
        for inputs in range(batch.shape[1]):
            for i in range(self.nr_passes):
                # inputs = torch.tensor(inputs).to(torch.float32)
                inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

                inputs = inputs.to(torch.float32)
                self.inputs.append(inputs)
                # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
                cur1 = self.lin1(inputs)
                spk1, self.mem1 = self.lif1(cur1, self.mem1)

                cur2 = self.lin2(spk1)
                spk2, self.mem2 = self.lif2(cur2, self.mem2)
            
            actions =  self.actor_linear(spk2)
            _, self.mem_act = self.action_lif(actions, self.mem_act)
            
            values.append(self.mem_act.reshape((1,-1)))

        # add information for plotting purposes
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2.squeeze(0).detach())  # Record the output trace of spikes
        self.mem1_rec.append(actions.squeeze(0).detach())  # Record the output trace of membrane potential

        return torch.stack(values, dim = 1)
  
    def step_forward(self, inputs):
        for i in range(self.nr_passes):
            # inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)
            self.inputs.append(inputs)
            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
        
        actions =  self.actor_linear(spk2)
        _, self.mem_act = self.action_lif(actions, self.mem_act)
        
        val = self.critic_linear(spk2)
        _, self.mem_val = self.value_lif(val, self.mem_val)
        vel = self.fc_vel(spk2)
        _, self.mem3 = self.li_vel(vel, self.mem3)
        vel = self.mem3
        # add information for plotting purposes
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2.squeeze(0).detach())  # Record the output trace of spikes
        self.mem1_rec.append(actions.squeeze(0).detach())  # Record the output trace of membrane potential

        return self.mem_act.reshape((1,-1))
  
  
class ActorCriticSNN_LIF_SYN_drone(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden1=64,hidden2=64, inp_min = torch.tensor([0]), inp_max=  torch.tensor([2.5]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_SYN_drone, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        self.num_outputs = action_space.n
        self.num_inputs = num_inputs
        beta = 0.95
        self.nr_passes = nr_passes

        # randomly initialize decay rate and threshold for layer 1
        beta_hidden = torch.rand(1)
        thr_hidden = torch.rand(1)

        self.lin1 = nn.Linear(num_inputs, hidden1)
        self.lif1 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(1)
        thr_hidden = torch.rand(1)
        alpha_hidden = torch.rand(1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.syn2 = snn.Synaptic(beta = beta_hidden,alpha=alpha_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True, learn_alpha=True)
        # self.lif2 = snn.Synaptic(beta = .75, alpha = 0.5, spike_grad=self.spike_grad, learn_beta=False, learn_alpha=False)
        self.action_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        self.value_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_vel = torch.nn.Linear(in_features=hidden2, out_features=1)
        self.li_vel = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=self.spike_grad, reset_mechanism="none")
        self.critic_linear = nn.Linear(hidden2, 1)
        self.actor_linear = nn.Linear(hidden2, self.num_outputs)
     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2, self.syncur2     = self.syn2.init_synaptic()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []

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
        self.mem2, self.syncur2     = self.syn2.init_synaptic()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        
        self.inputs = []

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
            # inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)
            self.inputs.append(inputs)
            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2, self.syncur2 = self.syn2(cur2, self.mem2, self.syncur2)



        actions =  self.actor_linear(spk2)
        _, self.mem_act = self.action_lif(actions, self.mem_act)
        
        val = self.critic_linear(spk2)
        _, self.mem_val = self.value_lif(val, self.mem_val)
        vel = self.fc_vel(spk2)
        _, self.mem3 = self.li_vel(vel, self.mem3)
        vel = self.mem3
        # add information for plotting purposes
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2.squeeze(0).detach())  # Record the output trace of spikes
        self.mem1_rec.append(actions.squeeze(0).detach())  # Record the output trace of membrane potential

        return self.mem_val.reshape((1,-1)), self.mem_act.reshape((1,-1)), vel

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()
        print(self.lin1.weight.data.shape)
        def animate(i):
            ax.clear()
            ax.set_xlim(-1, 250)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.num_inputs):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(64):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(64):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Blues(self.spk1_rec[i][j])))

            # plot spikes_2
            for j in range(self.num_outputs):
                ax.add_artist(plt.Circle((j, 3), 0.2, color=plt.cm.Greens(self.mem1_rec[i][j])))
            
            # for j in range(1):
            #     ax.add_artist(plt.Circle((j, 2.5), 0.2, color=plt.cm.Greens(self.mem3_rec[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()

class ActorCriticSNN_SYN_LIF_drone(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden1=64,hidden2=64, inp_min = torch.tensor([0]), inp_max=  torch.tensor([2.5]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_SYN_LIF_drone, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        self.num_outputs = action_space.n
        self.num_inputs = num_inputs
        beta = 0.95
        self.nr_passes = nr_passes

        # randomly initialize decay rate and threshold for layer 1
        beta_hidden = torch.rand(hidden1)
        thr_hidden = torch.rand(hidden1)
        alpha_hidden = torch.rand(hidden1)

        self.lin1 = nn.Linear(num_inputs, hidden1)
        self.syn1 = snn.Synaptic(beta = beta_hidden,alpha=alpha_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True, learn_alpha=True)


        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(hidden2)
        thr_hidden = torch.rand(hidden2)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta = beta_hidden, spike_grad=self.spike_grad, threshold=thr_hidden, learn_beta=True)
        # self.lif2 = snn.Synaptic(beta = .75, alpha = 0.5, spike_grad=self.spike_grad, learn_beta=False, learn_alpha=False)
        self.action_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        self.value_lif = snn.Leaky(beta = 0.95, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')
        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_vel = torch.nn.Linear(in_features=hidden2, out_features=1)
        self.li_vel = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=self.spike_grad, reset_mechanism="none")
        self.critic_linear = nn.Linear(hidden2, 1)
        self.actor_linear = nn.Linear(hidden2, self.num_outputs)
     
       # membranes at t = 0
        self.mem2     = self.lif2.init_leaky()
        self.mem1, self.syncur1     = self.syn1.init_synaptic()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.mem2     = self.lif2.init_leaky()
        self.mem1, self.syncur1     = self.syn1.init_synaptic()
        self.mem3     = self.li_vel.init_leaky()
        self.mem_act  = self.action_lif.init_leaky()
        self.mem_val  = self.value_lif.init_leaky()
        
        self.inputs = []

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
            # inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)
            self.inputs.append(inputs)
            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1, self.syncur1 = self.syn1(cur1, self.mem1, self.syncur1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)



        actions =  self.actor_linear(spk2)
        _, self.mem_act = self.action_lif(actions, self.mem_act)
        
        val = self.critic_linear(spk2)
        _, self.mem_val = self.value_lif(val, self.mem_val)
        vel = self.fc_vel(spk2)
        _, self.mem3 = self.li_vel(vel, self.mem3)
        vel = self.mem3
        # add information for plotting purposes
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2.squeeze(0).detach())  # Record the output trace of spikes
        self.mem1_rec.append(actions.squeeze(0).detach())  # Record the output trace of membrane potential

        return self.mem_val.reshape((1,-1)), self.mem_act.reshape((1,-1)), vel

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()
        print(self.lin1.weight.data.shape)
        def animate(i):
            ax.clear()
            ax.set_xlim(-1, 250)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.num_inputs):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(64):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(64):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Blues(self.spk1_rec[i][j])))

            # plot spikes_2
            for j in range(self.num_outputs):
                ax.add_artist(plt.Circle((j, 3), 0.2, color=plt.cm.Greens(self.mem1_rec[i][j])))
            
            # for j in range(1):
            #     ax.add_artist(plt.Circle((j, 2.5), 0.2, color=plt.cm.Greens(self.mem3_rec[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()


class ActorCriticSNN_LIF_Smallest(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size=246, inp_min = torch.tensor([0,-2]), inp_max=  torch.tensor([2,2]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        # self.spike_grad = surrogate.fast_sigmoid(slope=25)

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, hidden_size)
        self.lif1 = snn.Leaky(beta = .5, spike_grad=self.spike_grad, learn_beta=True)
        self.num_inputs = num_inputs

        # basically not spiking final layer
        self.num_outputs = action_space.n
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.lif_critic = snn.Leaky(beta = 0.25, spike_grad=self.spike_grad, learn_beta=True,reset_mechanism='none')

        self.actor_linear = nn.Linear(hidden_size, self.num_outputs)
        self.lif_actor = snn.Leaky(beta = 0.25, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_critic.init_leaky()
        self.mem3     = self.lif_actor.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []
        
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self, random= False):
        self.inputs = []
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_critic.init_leaky()
        self.mem3     = self.lif_actor.init_leaky()

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



        actions =  self.actor_linear(spk1)
        spk_actions,self.mem2 = self.lif_actor(actions, self.mem2)
        val = self.critic_linear(spk1)
        val, self.mem3 = self.lif_critic(val, self.mem3)
        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        self.mem3_rec.append(self.mem3.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        val = self.mem3
        actions = self.mem2
        return val, actions
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()
        print(self.lin1.weight.data.shape)
        def animate(i):
            ax.clear()
            ax.set_xlim(-1, 250)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.num_inputs):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(246):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(self.num_outputs):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            for j in range(1):
                ax.add_artist(plt.Circle((j, 2.5), 0.2, color=plt.cm.Greens(self.mem3_rec[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('activity.gif', writer=writer)
        plt.show()


class ActorCriticSNN_LIF_Smallest_Cont(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest_Cont, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        num_outputs = action_space.n

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, 100)
        self.lif1 = snn.Leaky(beta = .75, spike_grad=self.spike_grad, learn_beta=True)

        # basically not spiking final layer
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(100, 1)
        self.lif_critic = snn.Leaky(beta = .4, spike_grad=self.spike_grad, learn_beta=True,reset_mechanism='none')

        self.mean_linear = nn.Linear(100, num_outputs)
        self.lif_mean = snn.Leaky(beta = 0.4, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')

        self.log_std_linear = nn.Linear(100, num_outputs)
        self.lif_log_std = snn.Leaky(beta=0.4, spike_grad=self.spike_grad, learn_beta=True, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem_critic     = self.lif_critic.init_leaky()
        self.mem_mean     = self.lif_mean.init_leaky()
        self.mem_log_std     = self.lif_log_std.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []
        
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.inputs = []
        self.mem1     = self.lif1.init_leaky()
        self.mem_critic     = self.lif_critic.init_leaky()
        self.mem_mean     = self.lif_mean.init_leaky()
        self.mem_log_std     = self.lif_log_std.init_leaky()

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



        vals =  self.critic_linear(spk1)
        spk_actions,self.mem_critic = self.lif_mean(vals, self.mem_critic)

        mean = self.mean_linear(spk1)
        mean, self.mem_mean = self.lif_mean(mean, self.mem_mean)

        log_std = self.log_std_linear(spk1)
        log_std, self.mem_log_std = self.lif_log_std(log_std, self.mem_log_std)

        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem_critic.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        self.mem3_rec.append(self.mem_mean.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        return self.mem_critic, self.mem_mean, self.mem_log_std
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1, 150)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(4):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(100):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(2):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            for j in range(1):
                ax.add_artist(plt.Circle((j, 2.5), 0.2, color=plt.cm.Greens(self.mem3_rec[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()

        

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
        self.lin1 = nn.Linear(num_inputs, 128)
        self.lin2 = nn.Linear(128, 128)
 

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)

        x = F.relu(self.lin1(inputs))
        x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.actor_linear(x)
    


class ActorCritic_ANN_Smallest(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size = 246):
        super(ActorCritic_ANN_Smallest, self).__init__()
        self.lin1 = nn.Linear(num_inputs, hidden_size)
 

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)

        x = F.relu(self.lin1(inputs))
        # x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.actor_linear(x)
    

class ActorCritic_ANN_Cont(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic_ANN_Cont, self).__init__()
        self.lin1 = nn.Linear(num_inputs, 128)
        self.lin2 = nn.Linear(128, 128)
 
        self.num_inputs = num_inputs
        num_outputs = 1
        self.critic_linear = nn.Linear(128, 1)
        self.mean_linear = nn.Linear(128, num_outputs)
        self.log_std_linear = nn.Linear(128, num_outputs)

        self.train()

    def forward(self, inputs):

        inputs = inputs.to(torch.float32).reshape(self.num_inputs,)

        x = F.relu(self.lin1(inputs))
        x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.mean_linear(x), self.log_std_linear(x)
