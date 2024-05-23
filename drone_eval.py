from environments import SimpleDrone_Discrete
import torch
from actor_critics import ActorCriticSNN_LIF_Smallest, ActorCriticSNN_LIF_drone
import torch.nn.functional as F
import numpy as np

env = SimpleDrone_Discrete(dt=0.02, max_episode_length=500)
# env.plot_reward_function()
env.reset()
action_size = env.action_space
state_size = env.observation_space.shape[0]

global_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                                #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   inp_min = torch.tensor([0, -.5]), 
                                                   inp_max=  torch.tensor([2.5, .5]), 
                                                   bias=False,nr_passes = 1)
global_model = ActorCriticSNN_LIF_drone(state_size, action_size, hidden1=32, hidden2=32)

global_model.load_state_dict(torch.load('DSQN/drone_snn_vel_syn_lif_1e4_3232.pt'))
# print(global_model.state_dict)
torch.set_printoptions(threshold=np.inf)

# print('layer 1 beta:', global_model.lif1.beta)
# print('layer 2 beta:', global_model.lif2.beta)
# print('layer action beta:', global_model.action_lif.beta)
# print('layer 1 thereshold:', global_model.lif1.threshold)
# print('layer 2 thereshold:', global_model.lif2.threshold)
# print('layer action thereshold:', global_model.action_lif.threshold)
# print('lin1 weights:')
# print(global_model.lin1.weight)
# print('lin1 bias:')
# print(global_model.lin1.bias)
# print('lin2 weights:')
# print(global_model.lin2.weight)
# print('lin2 bias:')
# print(global_model.lin2.bias)
# print('actor weights:')
# print(global_model.actor_linear.weight)
# print('actor bias:')
# print(global_model.actor_linear.bias)



global_model.eval()
iterations  = 1
success = 0
high = 0
spikes1 = []
spikes2 = []
inputs = torch.tensor([1.1000, 1.0999, 1.1001, 1.1004, 1.1007, 1.1009, 1.1009, 1.1009, 1.1009,
         1.1009, 1.1009, 1.1009, 1.1009, 1.1009, 1.1009, 1.1008, 1.1007, 1.1007,
         1.1006, 1.1006])
outputs = []
positions = []
j = 0
for i in range(int(iterations)):
    global_model.init_mem()
    state, _ = env.reset()
    done = False
    while not done:
        
        state = torch.from_numpy(state)
        positions.append(state)
        # state = inputs[j]
# get network outputs on given state
        value, policy,_ = global_model(state.unsqueeze(0))
        
            # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)



            # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
        outputs.append(action)
        state, reward, done, truncated, info = env.step(action.item())

        if done:
            if info['end_condition']=='crash':
                # print(env._agent_velocity)
                if env._agent_velocity>-0.5:
                    success += 1
                    break
            elif info['end_condition']=='high':
                high += 1
        j+=1
    spikes1.append(torch.stack(global_model.spk_in_rec)) 
    spikes2.append(torch.stack(global_model.spk1_rec) )
    # print('inputs:')
    # print(torch.stack(inputs).transpose(1,0))
    # print('outputs:')
    print(positions)  
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(torch.stack(positions).detach().numpy())
plt.subplot(2,1,2)
plt.scatter(range(len(outputs)),torch.stack(outputs).detach().numpy())
plt.show()
''' 
print('success rate:', success/iterations) 
print(success/iterations)
def pruning(all_spikes, hidden_size=32):    
    # print(all_spikes[0].shape, len(all_spikes))
    # all_spikes = torch.tensor(all_spikes)
    spikes = torch.cat(all_spikes, dim=0)
    spikes = spikes.sum(dim=0)/spikes.shape[0] # percentage of time each neuron spikes
    print(spikes)
    all_zeros = (spikes == 0)
    all_ones = (spikes == 1)
    # snn_model.plot_spikes()

    indices_zeros = np.where(all_zeros)[0]
    indices_ones = np.where(all_ones)[0]

    dead_saturated = all_zeros|all_ones

    indices_spiking = np.where(~dead_saturated)[0]
    # print('NOT DELETING SATURATED NEURONS YET')
    # indices_spiking = np.where(~all_zeros)[0]
    return np.where(dead_saturated), indices_zeros, indices_ones, indices_spiking

output_layer_1 = pruning(spikes1)
output_layer_2 = pruning(spikes2)

hidden_size_1 = output_layer_1[3].shape[0]
hidden_size_2 = output_layer_2[3].shape[0]
print('hidden size 1:', hidden_size_1)
print('hidden size 2:', hidden_size_2)

weights_layer_1 = torch.nn.Parameter(global_model.lin1.weight[output_layer_1[3],:])
bias_layer_1 = torch.nn.Parameter(global_model.lin1.bias[output_layer_1[3]])
print(weights_layer_1.shape, bias_layer_1.shape)

weights_layer_2 = torch.nn.Parameter(global_model.lin2.weight[:,output_layer_1[3]])
weights_layer_2 = torch.nn.Parameter(weights_layer_2[output_layer_2[3],:])
pre_bias_layer_2 = torch.nn.Parameter(global_model.lin2.weight[:, output_layer_1[2]])
pre_bias_layer_2 = torch.nn.Parameter(pre_bias_layer_2[output_layer_2[3],:])
bias_layer_2 = torch.nn.Parameter(global_model.lin2.bias[output_layer_2[3]])
bias_layer_2 = torch.nn.Parameter(bias_layer_2 + pre_bias_layer_2.sum(dim=1))
print(weights_layer_2.shape, bias_layer_2.shape)

weights_action = torch.nn.Parameter(global_model.actor_linear.weight[:,output_layer_2[3]])
bias_layer_action = torch.nn.Parameter(global_model.actor_linear.bias)
# add saturated neuron weights to bias
saturated_bias = torch.nn.Parameter(global_model.actor_linear.bias + global_model.actor_linear.weight[:, output_layer_2[2]].sum(dim=1))
print(weights_action.shape, saturated_bias.shape)
pruned_model = ActorCriticSNN_LIF_drone(state_size, action_size, hidden_size_1, hidden_size_2)
pruned_model.lin1.weight = weights_layer_1
pruned_model.lin1.bias = bias_layer_1

pruned_model.lin2.weight = weights_layer_2
pruned_model.lin2.bias = bias_layer_2

pruned_model.actor_linear.weight = weights_action
pruned_model.actor_linear.bias = saturated_bias

pruned_model.lif1.beta = global_model.lif1.beta
pruned_model.lif2.beta = global_model.lif2.beta
pruned_model.action_lif.beta = global_model.action_lif.beta
# pruned_model.lif2.alpha = global_model.lif2.alpha
# global_model.plot_spikes()
# print(success/iterations)
print('lin1 weights:')
print(pruned_model.lin1.weight)
print('lin1 bias:')
print(pruned_model.lin1.bias)
print('lin2 weights:')
print(pruned_model.lin2.weight)
print('lin2 bias:')
print(pruned_model.lin2.bias)
print('actor weights:')
print(pruned_model.actor_linear.weight)
print('actor bias:')
print(pruned_model.actor_linear.bias)
print('layer 1 beta:', pruned_model.lif1.beta)
print('layer 1 thereshold:', pruned_model.lif1.threshold)
print('layer 2 beta:', pruned_model.lif2.beta)
print('layer 2 thereshold:', pruned_model.lif2.threshold)
print('layer action beta:', pruned_model.action_lif.beta)
print('layer action thereshold:', pruned_model.action_lif.threshold)

success = 0
high = 0
print('\n\n\nEvaluation:')
for i in range(int(100)):
    global_model.init_mem()
    pruned_model.init_mem()
    state, _ = env.reset()
    done = False
    while not done:
        state = torch.from_numpy(state)
# get network outputs on given state
        # _, policy_global = global_model(state.unsqueeze(0))
        value, policy,_ = pruned_model(state.unsqueeze(0))
        # print(np.sqrt((policy_global - policy).pow(2).sum().item()))

            # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)



            # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

        state, reward, done, truncated, info = env.step(action)

        if done:
            if info['end_condition']=='crash':
                print(env._agent_velocity)
                if env._agent_velocity>-0.5:
                    success += 1
                    break
            elif info['end_condition']=='high':
                high += 1

print(success/100)
print(high/100)
'''