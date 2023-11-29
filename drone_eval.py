from environments import SimpleDrone_Discrete
import torch
from actor_critics import ActorCriticSNN_LIF_Smallest
import torch.nn.functional as F


env = SimpleDrone_Discrete(dt=0.02, max_episode_length=500)
env.reset()
action_size = env.action_space
state_size = env.observation_space.shape[0]

global_model = ActorCriticSNN_LIF_Smallest(state_size, action_size,
                                                #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   inp_min = torch.tensor([0, -.5]), 
                                                   inp_max=  torch.tensor([2.5, .5]), 
                                                   bias=False,nr_passes = 1)

global_model.load_state_dict(torch.load('drone_snn_small_test_FURTHER.pt'))

global_model.eval()
iterations  = 1e3
success = 0
for i in range(int(iterations)):
    global_model.init_mem()
    state, _ = env.reset()
    done = False
    while not done:
        state = torch.from_numpy(state)
# get network outputs on given state
        value, policy = global_model(state.unsqueeze(0))

            # find probabilities of certain actions
        prob = F.softmax(policy, dim=-1)



            # choose the action and detach from computational graph
        action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

        state, reward, done, truncated, info = env.step(action)
        if info['landed']:
            success += 1
            break
        if done:
            if info['end_condition']=='crash':
                print(state)
print(success/iterations)