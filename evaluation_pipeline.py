import torch
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from snntorch import surrogate
import gym
from tqdm import tqdm
from actor_critics import ActorCriticSNN_LIF_Smallest,ActorCriticSNN_LIF_Small, ActorCritic_ANN,ActorCritic_ANN_Smallest
from matplotlib import animation
from CartPole_modified import CartPole_fake as CartPole
# env = gym.make('CartPole-v1',render_mode = 'human')
env = gym.make('CartPole-v1')
env = CartPole()
env.tau = 0.05

state_list = [] # used to replicate for SNN
state_size = env.observation_space.shape[0]
action_size = env.action_space
spiking = False
class ActorCriticSNN_LIF_Smallest_pruned(torch.nn.Module):
    def __init__(self, nr_ins, nr_outs,hidden_size=50, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]), nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest_pruned, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        self.nr_ins = nr_ins
        self.nr_outs = nr_outs.n
        self.hidden_size = hidden_size

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(nr_ins, hidden_size)
        self.lif1 = snn.Leaky(beta = .45, spike_grad=self.spike_grad, learn_beta=True)
        
        # basically not spiking final layer
        num_outputs = nr_outs

        self.actor_linear = nn.Linear(hidden_size, self.nr_outs)
        self.lif_actor = snn.Leaky(beta = 0, spike_grad=self.spike_grad, learn_beta=False, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_actor.init_leaky()

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
        self.mem2     = self.lif_actor.init_leaky()

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

        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        actions = self.mem2
        return 0,actions
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1, self.hidden_size)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.nr_ins):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(self.hidden_size):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(self.nr_outs):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            
        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()


snn_model = ActorCriticSNN_LIF_Small(state_size, action_size,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    bias=False,nr_passes = 1)

snn_model.load_state_dict(torch.load('A3C/past_trainings/Figures/SNN_in128x2out_50e3_20hz_real.pt'))

snn_model_1 = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    bias=False,nr_passes = 1)

snn_model_1.load_state_dict(torch.load('A3C/past_trainings/Figures/SNN_in246out_25e3_0gain_higherleaks.pt'))

snn_model_pruned = ActorCriticSNN_LIF_Smallest_pruned(state_size, action_size, hidden_size=11,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    nr_passes = 1)

snn_model_pruned.load_state_dict(torch.load('A3C/past_trainings/Figures/SNN_in248out_25e3_0gain_noleak_pruned_full.pt'))

snn_model_2_pruned = ActorCriticSNN_LIF_Smallest_pruned(state_size, action_size, hidden_size=21,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    nr_passes = 1)

snn_model_2_pruned.load_state_dict(torch.load('A3C/past_trainings/Figures/SNN_in248out_25e3_0gain_higher_leaks_pruned_full.pt'))
print('SNN loaded')

ann_model = ActorCritic_ANN(state_size, action_size)
ann_model.load_state_dict(torch.load('A3C/past_trainings/Figures/ANN_in128x2out_50e3_20hz.pt'))

snn_model.eval()
ann_model.eval()


def evaluate_model(model, gain, spiking, nr_its = 250):
    times = []
    all_spikes = []
    for _ in range(nr_its):
        if spiking:
            model.init_mem()
        state,_ = env.reset()
        state_list.append(state)

        t_sim = 0
        terminal = False
        reward = 0
        noisy_states = []
        while not terminal and t_sim < 1000:
            t_sim+=1
            # state to tensor
            state = torch.from_numpy(state)
            
            state = add_noise(state, gain=gain)
            noisy_states.append(state)

            # get network outputs on given state
            _, policy = model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

            
            # perform action
            obs, _, terminal, _, _ = env.step(int(action.squeeze(0)))


            
            if not terminal:
                state = obs
        # monitor always spiking and never spiking in model.spk_in_rec
        times.append(t_sim)
        # BEGIN: qm3v9g7j8z9p
        if spiking:
            spikes = torch.stack(model.spk_in_rec)
            
            all_spikes.append(spikes)
    # concatenate all spike tensors in all_spikes into a single tensor
    # plot_noisy_data(noisy_states, gain)

    # average_times.append(np.mean(times))
    return np.mean(times), all_spikes

def add_noise(state, gain=0.1):
    noise = np.random.normal(0, gain, state.shape)
    return state + noise

def plot_noisy_data(states_noisy, gain):
    states_noisy_array = np.stack(states_noisy, axis=1)
    size = states_noisy_array.shape[0]
    plt.figure()
    for obs in range(size):
        plt.subplot(size,1, obs+1)
        plt.plot(list(range(states_noisy_array.shape[1])), states_noisy_array[obs,:])
    plt.suptitle(f'Noise: {gain}')
    plt.show()

def noise_analysis(ann_model, snn_model, snn_model_2, snn_model_pruned, snn_model_2_pruned):
    all_spikes = []
    times = []
    average_times_ann = []
    average_times_snn = []
    average_times_snn_pruned = []
    average_times_snn_2 = []
    average_times_snn_2_pruned = []
    noise_gains = np.linspace(0,1,60)
    for gain in tqdm(noise_gains):
        average_times_snn.append(evaluate_model(snn_model, gain, spiking=True)[0]) # added an output to the evaluate fion
        average_times_ann.append(evaluate_model(ann_model, gain, spiking=False)[0])
        # average_times_snn_2.append(evaluate_model(snn_model_2, gain, spiking=True)[0])
        # average_times_snn_pruned.append(evaluate_model(snn_model_pruned, gain, spiking=True)[0])
        # average_times_snn_2_pruned.append(evaluate_model(snn_model_2_pruned, gain, spiking=True)[0])
        # print('Last average times: ', average_times_ann[-1],average_times_snn[-1],average_times_snn_2[-1],average_times_snn_pruned[-1],average_times_snn_2_pruned[-1])
        # local_model.plot_spikes()
    plt.figure()
    plt.plot(noise_gains, average_times_ann, label='ANN, 2 hidden layers, both size 128')
    plt.plot(noise_gains, average_times_snn, label='SNN, 2 hidden layers, both size 128')
    # plt.plot(noise_gains, average_times_snn_pruned, label='SNN, 1 hidden layer, size 11')
    # plt.plot(noise_gains, average_times_snn_2, label='SNN, 1 hidden layer, size 246, leaky')
    # plt.plot(noise_gains, average_times_snn_2_pruned, label='SNN, 1 hidden layer, size 21, leaky')

    plt.xlabel('Noise gain')
    plt.ylabel('Average time')
    plt.title('Noise vs Average time')
    plt.legend()
    plt.show()

noise_analysis(ann_model, snn_model, snn_model_1, snn_model_pruned, snn_model_2_pruned)

def pruning(snn_model, hidden_size=246):
    avg_time, all_spikes = evaluate_model(snn_model, 0, spiking=True, nr_its=10000)
    
    spikes = torch.cat(all_spikes, dim=0)
    spikes = spikes.sum(dim=0)/spikes.shape[0] # percentage of time each neuron spikes

    all_zeros = (spikes == 0)
    all_ones = (spikes == 1)
    # snn_model.plot_spikes()

    indices_zeros = np.where(all_zeros)[0]
    indices_ones = np.where(all_ones)[0]

    dead_saturated = all_zeros|all_ones

    indices_spiking = np.where(~dead_saturated)[0]
    # print('NOT DELETING SATURATED NEURONS YET')
    # indices_spiking = np.where(~all_zeros)[0]
    print(avg_time)
    return np.where(dead_saturated), indices_zeros, indices_ones, indices_spiking
'''
indices_dead_sat, indices_zeros, indices_ones, indices_spiking = pruning(snn_model)

hidden_size = len(indices_spiking)
# pruning efforts:
# 1. remove all neurons that never spike or always spike
print(hidden_size)
weights_layer_1 = torch.nn.Parameter(snn_model.lin1.weight[indices_spiking,:])
bias_layer_1 = torch.nn.Parameter(snn_model.lin1.bias[indices_spiking])

weights_layer_2 = torch.nn.Parameter(snn_model.actor_linear.weight[:,indices_spiking])
bias_layer_2 = torch.nn.Parameter(snn_model.actor_linear.bias)
# add saturated neuron weights to bias
saturated_bias = torch.nn.Parameter(snn_model.actor_linear.bias + snn_model.actor_linear.weight[:, indices_ones].sum(dim=1))

    
class ActorCriticSNN_LIF_Smallest_pruned(torch.nn.Module):
    def __init__(self, nr_ins, nr_outs,hidden_size=50, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]), nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest_pruned, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        self.nr_ins = nr_ins
        self.nr_outs = nr_outs
        self.hidden_size = hidden_size

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(nr_ins, hidden_size)
        self.lif1 = snn.Leaky(beta = .45, spike_grad=self.spike_grad, learn_beta=True)
        
        # basically not spiking final layer
        num_outputs = nr_outs

        self.actor_linear = nn.Linear(hidden_size, num_outputs)
        self.lif_actor = snn.Leaky(beta = 0, spike_grad=self.spike_grad, learn_beta=False, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_actor.init_leaky()

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
        self.mem2     = self.lif_actor.init_leaky()

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

        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        actions = self.mem2
        return 0,actions
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1, self.hidden_size)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.nr_ins):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(self.hidden_size):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(self.nr_outs):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            
        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()

pruned_snn = ActorCriticSNN_LIF_Smallest_pruned(state_size, action_size.n,
                                                hidden_size=hidden_size,
                                                inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                inp_max=  torch.tensor([4.8, 10,0.418,2]),)

# # load in weights and biases
pruned_snn.lin1.weight = weights_layer_1
pruned_snn.lin1.bias = bias_layer_1
pruned_snn.actor_linear.weight = weights_layer_2
pruned_snn.actor_linear.bias = saturated_bias

# load neuron characteristics
pruned_snn.lif1.beta = nn.Parameter(snn_model.lif1.beta)
pruned_snn.lif1.reset_mechanism = snn_model.lif1.reset_mechanism
pruned_snn.lif1.threshold = nn.Parameter(snn_model.lif1.threshold)

pruned_snn.lif_actor.beta = nn.Parameter(snn_model.lif_actor.beta)
pruned_snn.lif_actor.reset_mechanism = snn_model.lif_actor.reset_mechanism
pruned_snn.lif_actor.threshold = nn.Parameter(snn_model.lif_actor.threshold)

print(evaluate_model(pruned_snn, 0, spiking=True, nr_its=100)[0])
# pruned_snn.plot_spikes()
torch.save(pruned_snn.state_dict(), 'A3C/past_trainings/Figures/SNN_in248out_25e3_0gain_pruned_full.pt')


'''