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

state_list = [] # used to replicate for SNN
state_size = env.observation_space.shape[0]
action_size = env.action_space
spiking = False
# snn_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
#                                         inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
#                                         inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                         bias=False,nr_passes = 1)
# snn_model.load_state_dict(torch.load('A3C/past_trainings/Smallest_model.pt'))
snn_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    bias=False,nr_passes = 1)
# snn_model.load_state_dict(torch.load('A3C/past_trainings/SNN_128_noise-.1.pt'))
snn_model.load_state_dict(torch.load('A3C/past_trainings/Figures/SNN_in248out_50e3_0gain.pt'))
print('SNN loaded')
# snn_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
#                                         inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
#                                         inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                         bias=False,nr_passes = 1)
# snn_model.load_state_dict(torch.load('A3C/past_trainings/Smallest_model.pt'))

ann_model = ActorCritic_ANN_Smallest(state_size, action_size)
ann_model.load_state_dict(torch.load('A3C/past_trainings/Figures/ANN_in248out_50e3_0gain.pt'))


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


def evaluate_model(model, gain, spiking):
    for _ in tqdm(range(100)):
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

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)

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
    return np.mean(times)
# local_model.plot_spikes()
snn_model.eval()
ann_model.eval()
all_spikes = []
times = []
average_times_ann = []
average_times_snn = []
noise_gains = np.linspace(0,2,100)
for gain in noise_gains:
    average_times_snn.append(evaluate_model(snn_model, gain, spiking=True))
    average_times_ann.append(evaluate_model(ann_model, gain, spiking=False))
    # local_model.plot_spikes()
plt.figure()
plt.plot(noise_gains, average_times_ann, label='ANN, 1 hidden layer, size 246')
plt.plot(noise_gains, average_times_snn, label='SNN, 1 hidden layer, size 246')

plt.xlabel('Noise gain')
plt.ylabel('Average time')
plt.title('Noise vs Average time')
plt.legend()
plt.show()

spikes = torch.cat(all_spikes, dim=0)
all_zeros = (spikes == 0).all(dim=0)
all_ones = (spikes == 1).all(dim=0)

result = spikes[:, all_zeros | all_ones]

indices_zeros = (all_zeros).nonzero(as_tuple=True)[0]
indices_ones = (all_ones).nonzero(as_tuple=True)[0]

all_indices = np.arange(0,100,1)
indices_both = np.setdiff1d(all_indices, np.union1d(indices_zeros, indices_ones))
indices_spiking = np.setdiff1d(all_indices, indices_zeros)


# pruning efforts:
# 1. remove all neurons that never spike or always spike
# print(local_model.lin1.weight.shape)
# local_model.lin1.weight = torch.nn.Parameter(local_model.lin1.weight[:, indices_both])

# print(local_model.lin1.weight.shape)

# # in second layer, always spiking is bias
# bias = local_model.actor_linear.bias + local_model.actor_linear.weight[:, indices_ones].sum(dim=1)
# local_model.actor_linear.weight = torch.nn.Parameter(local_model.actor_linear.weight[:, indices_both])

    
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


new_weights = local_model.lin1.weight.clone()
first_bias = local_model.lin1.bias.clone()

new_weights[indices_zeros, :] = 0
new_weights[indices_ones,:] = 0
first_bias[indices_zeros] = 0
first_bias[indices_ones] = 0


new_weights2 = local_model.actor_linear.weight.clone()
new_bias = new_weights2[:, indices_ones].sum(dim=1)

new_weights2[:, indices_zeros] = 0
new_weights2[:, indices_ones] = 0
new_bias += local_model.actor_linear.bias

pruned_weights1 = new_weights[indices_both,:]
pruned_weights2 = new_weights2[:,indices_both]
# print(indices_spiking)

# pruned_model = ActorCriticSNN_LIF_Smallest_pruned(4,2,len(indices_both), 
#                                            inp_min = torch.tensor([-4.8, -10,-0.418,-2]),
#                                            inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                            nr_passes = 1)
# pruned_model.lin1.weight = torch.nn.Parameter(pruned_weights1)
# pruned_model.lin1.bias = torch.nn.Parameter(local_model.lin1.bias[indices_both])
# pruned_model.actor_linear.weight = torch.nn.Parameter(pruned_weights2)


# pruned_model.actor_linear.bias = torch.nn.Parameter(new_bias)

pruned_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                        inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                        inp_max=  torch.tensor([4.8, 10,0.418,2]),
                                        nr_passes = 1)

pruned_model.load_state_dict(torch.load('A3C/past_trainings/Smallest_model.pt'))
pruned_model.lin1.weight = torch.nn.Parameter(new_weights)
pruned_model.lin1.bias = torch.nn.Parameter(first_bias)
pruned_model.actor_linear.weight = torch.nn.Parameter(new_weights2)
pruned_model.actor_linear.bias = torch.nn.Parameter(new_bias)

times_pruned = []

for i in tqdm(range(100)):
    # pruned_model.init_mem()
    local_model.init_mem()
    # state,_ = env.reset()
    state = state_list[i]

    env.set_state(state)
    state = np.array(state)
    t_sim = 0
    terminal = False
    reward = 0
    error = 0
    while not terminal and t_sim < 1000:
        # print(state)
        t_sim+=1
        # state to tensor
        state = torch.from_numpy(state)

        # get network outputs on given state
        _, policy = pruned_model(state.unsqueeze(0))
        # _, policy = local_model(state.unsqueeze(0))
        # _, policy2 = local_model(state.unsqueeze(0))
        # error += torch.sum(torch.abs(policy - policy2))
        # print(error)
        # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)

        logprob = F.log_softmax(policy, dim=-1)

        # calculate entropy
        entropy = -(logprob * prob).sum(1, keepdim=True)

        # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
        
        # perform action
        obs, _, terminal, _, _ = env.step(int(action.squeeze(0)))


        
        if not terminal:
            state = obs
    # monitor always spiking and never spiking in model.spk_in_rec
    times_pruned.append(t_sim)
    
    # spikes = torch.stack(pruned_model.spk_in_rec)
    # spikes_og = torch.stack(local_model.spk_in_rec)
    if (error/t_sim) !=0:
        print("average error: ", error/t_sim)
        # print(torch.count_nonzero(spikes_og[:,indices_zeros]))

    # all_spikes.append(spikes)
print("plotting spikes...\n\n")
print("average non-pruned time: ", np.mean(times))
print("average pruned time: ", np.mean(times_pruned))
pruned_model.plot_spikes()
local_model.plot_spikes()
print("Done!")