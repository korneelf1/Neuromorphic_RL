import torch
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gym

from actor_critics import ActorCriticSNN_LIF_Smallest

# env = gym.make('CartPole-v1',render_mode = 'human')
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space

local_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                        inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                        inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                        bias=False,nr_passes = 1)

local_model.load_state_dict(torch.load('past_trainings/Smallest_model.pt'))

local_model.eval()
local_model.init_mem()
state,_ = env.reset()

t_sim = 0
terminal = False
reward = 0

print('Interacting with environment')
while not terminal:

    # state to tensor
    state = torch.from_numpy(state)

    # get network outputs on given state
    value, policy = local_model(state.unsqueeze(0))

    # find probabilities of certain actions
    prob = F.softmax(policy, dim=-1)

    logprob = F.log_softmax(policy, dim=-1)

    # calculate entropy
    entropy = -(logprob * prob).sum(1, keepdim=True)

    # choose the action and detach from computational graph
    action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

    # get log probs of the actions taken
    log_prob = logprob.gather(1,action)
    
    # perform action
    obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))

    # !!!!!!!!!!!
    reward = max(min(reward, 1), -1)


    
    if not terminal:
        state = obs

print("plotting spikes...\n\n")
local_model.plot_spikes()
print("Done!")