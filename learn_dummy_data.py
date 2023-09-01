from helper_functions import ActorCriticSNN, ActorCritic, plot_activity, A3Cnet, ActorCriticSNN_dummy
from dummydata import dummydata
import matplotlib.pyplot as plt

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

ITERATIONS = 5000


in_size     = 2
out_actions = 2
out_val     = 1

# data params
N         = 1
LENGTH    = 25
SCALE_IN  = 2
SCALE_ACT = 3
SCALE_VAL = 5
WARM_UP   = 5

# having the warmup vs not having a warmup of at least the depth of the network will decrease performance tremendously, 1e-6 vs 1e-15
class ActionSpace:
    def __init__(self, value) -> None:
        self.n = value

actionspace = ActionSpace(out_actions)
# model = ActorCritic(in_size, actionspace).to(device) # note layers are actually linear not convolutional

# model = ActorCriticSNN(in_size, actionspace,out_val,inp_min=torch.tensor([0,0]),inp_max=torch.tensor([1,1]), alpha=0.90,  beta = 0.5, threshold = 1).to(device)
model = ActorCriticSNN_dummy(in_size, actionspace,out_val,inp_min=torch.tensor([0,0]),inp_max=torch.tensor([1,1])).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4
                             )

# in_data, labels = dummydata(in_size, out_actions, out_val).create_random(N, LENGTH,SCALE_IN,SCALE_ACT,SCALE_VAL)
# in_data, labels = dummydata(in_size, out_actions, out_val).create_quadratic(N, LENGTH,SCALE_IN)
# in_data, labels = dummydata(in_size, out_actions, out_val).create_multip(N, LENGTH,SCALE_IN)
in_data, labels = dummydata(in_size, out_actions, out_val).create_xor(LENGTH)

N = 1
values = []
actions = []
losses = []
lossfn    = torch.nn.MSELoss()

for iteration in range(ITERATIONS):
    loss_vals = 0
    loss_act  = 0
    loss = 0
    optimizer.zero_grad()
    # model.zero_grad()
    model.init_mem()
    in_data, labels = dummydata(in_size, out_actions, out_val).create_xor(LENGTH)
    # print(labels[0][0])
    for i in range(LENGTH):
        # print(in_data[0][i])
        # print(in_data[0][i])
        model_val, model_act = model(in_data[0][i].to(device).unsqueeze(0))
        # print(model_val, model_act)
        if i >= WARM_UP:
            loss_vals = lossfn(model_val, labels[0][0][i].to(device))
            # loss_vals = 0
            # loss_act  = lossfn(model_act, labels[1][0][i].to(device))
            loss_act = 0
            
            # print(loss_act, loss_vals)
            loss += loss_vals + loss_act
            # print(loss)
    loss /= LENGTH
    # print(model_val)
    # plot_activity(torch.stack(model.spk1_rec),torch.stack(model.spk2_rec),torch.stack(model.spk3_rec))
    # spike_sparsity_loss = torch.sum(torch.stack(model.spk_in_rec)) + torch.sum(torch.stack(model.spk1_rec)) + torch.sum(torch.stack(model.spk2_rec)) + torch.sum(torch.stack(model.spk3_rec))
    # # print('spike_sparsity loss: ' + str(spike_sparsity_loss*.00005 + 1/spike_sparsity_loss*100))
    # # print('other loss: '+ str((policy_loss + value_loss * VALUE_LOSS_COEF )))
    # spikes_loss = spike_sparsity_loss*.05 + 1/(spike_sparsity_loss+1e-6)*100
    # loss += spikes_loss/10000
    # loss = loss
    
    # Print the gradients before calling backward()
    # print('Gradients before backward:')
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.norm())

    loss.backward()

    # Print the gradients after calling backward()
    # print('Gradients after backward:')
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.norm())

    # print(i, model.actor_linear.weight.grad)
    # print(model.layer3.weight.grad)
    optimizer.step()
    # to be sure
    # model.zero_grad()
    # optimizer.zero_grad()


    
    # loss = loss_vals + loss_act
    print('iteration: '+ str(iteration)+ '\t'+ str(loss.item()))
    losses.append(loss.detach().to('cpu').squeeze(0))
    # print('Loss:', loss.item())

    # optimizer.step()
# plot_activity(torch.stack(model.spk1_rec),torch.stack(model.spk2_rec),torch.stack(model.spk3_rec))
plt.plot(range(len(losses)),losses)
plt.show()