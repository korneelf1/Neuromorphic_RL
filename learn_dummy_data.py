from helper_functions import ActorCriticSNN, ActorCritic
from dummydata import dummydata
import matplotlib.pyplot as plt

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ITERATIONS = 1000


in_size     = 4
out_actions = 2
out_val     = 1

# data params
N         = 1
LENGTH    = 100
SCALE_IN  = 2
SCALE_ACT = 3
SCALE_VAL = 5
class ActionSpace:
    def __init__(self, value) -> None:
        self.n = value

actionspace = ActionSpace(out_actions)
# model = ActorCritic(in_size, actionspace).to(device)
model = ActorCriticSNN(in_size, actionspace,out_val,inp_min=torch.tensor([-2,-2,-2,-2]),inp_max=torch.tensor([2,2,2,2])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

in_data, labels = dummydata(in_size, out_actions, out_val).create_random(N, LENGTH,SCALE_IN,SCALE_ACT,SCALE_VAL)
# in_data, labels = dummydata(in_size, out_actions, out_val).create_quadratic(N, LENGTH,SCALE_IN)
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
    for i in range(LENGTH):
        
      
        model_val, model_act = model(in_data[0][i].to(device).unsqueeze(0), device=device)
        print(model_val, model_act)
        loss_vals = lossfn(model_val, labels[0][0][i].to(device))
        loss_act  = lossfn(model_act, labels[1][0][i].to(device))
        # loss_act = 0
        
        loss += loss_vals + loss_act
    losses.append(loss.detach().to('cpu').squeeze(0))
    print('Loss:', loss.item())

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

    optimizer.step()
    # to be sure
    # model.zero_grad()
    # optimizer.zero_grad()


    
    # loss = loss_vals + loss_act
    print('iteration: '+ str(iteration)+ '\t'+ str(loss.item()))

    # optimizer.step()

plt.plot(range(len(losses)),losses)
plt.show()