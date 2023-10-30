import torch
import helper_functions as hf
import gym
import snntorch
import numpy as np
neuron = snntorch.Leaky(0.8, reset_mechanism='none')
state = neuron.init_leaky()
def charge_neuron(state, input):
    input = torch.tensor(input)
    spike, state = neuron(input, state)
    return state
    

inputs = np.zeros(100)
inputs[1] = -.1
inputs[10] = -.5
inputs[50] = -1.5
inputs[60] = 1.5

states = []
for input in inputs:
    state = charge_neuron(state, input)
    states.append(state.detach().numpy())

import matplotlib.pyplot as plt
plt.plot(inputs)
plt.show()
plt.plot(states)
plt.show()