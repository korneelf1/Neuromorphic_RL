import numpy as np
import torch 

class dummydata:
    def __init__(self, in_size, out_actions, out_val) -> None:
        self.inp = in_size
        self.nact = out_actions
        self.vals = out_val

    def create_random(self, size,length, scale_factor_ins, scale_factor_actions, scale_factor_vals):
        inputs = torch.randn((size, length, self.inp), requires_grad = False)*scale_factor_ins
        outs   = [torch.randn((size, length, self.vals), requires_grad = False)*scale_factor_vals,torch.randn((size, length, self.nact), requires_grad = False)*scale_factor_actions]
        return inputs, outs
    
    def create_quadratic(self, size,length, scale_factor_ins):
        inputs = torch.randn((size, length, self.inp), requires_grad = False)*scale_factor_ins
        # print(inputs.size(),torch.sum(inputs,dim=-1).size(),torch.sqrt(inputs)[:,:,:2].size())
        outs   = [torch.sqrt(inputs)[:,:,:2],torch.sum(inputs,dim=-1)]
        return inputs, outs

    def create_multip(self, size,length, scale_factor_ins):
        inputs = torch.randn((size, length, self.inp), requires_grad = False)*scale_factor_ins

        out_val = torch.tensor(torch.sum(inputs,dim=-1), requires_grad=False)
        out_act = torch.tensor(np.column_stack((torch.sum(inputs,dim=-1),torch.sum(inputs,dim=-1))),requires_grad=False)

        out = [out_act, out_val]
        return inputs, out
    
    def create_xor(self, length):
        # inputs = torch.tensor([0,0],[0,1],[1,0],[1,1], dtype=torch.float32)
        # labels = torch.tensor([0,1,1,0], dtype=torch.float32)
        inputs1 = torch.zeros((length,2),requires_grad=False).unsqueeze(0)
        out_val1 = torch.zeros((length,1),requires_grad=False).unsqueeze(0)
        out_act1 = torch.zeros((length,2),requires_grad=False)
        out_act1[:,0] = 1
        out_act1 = out_act1.unsqueeze(0)
        
        inputs2 = torch.ones((length,2),requires_grad=False).unsqueeze(0)
        out_val2 = torch.zeros((length,1),requires_grad=False).unsqueeze(0)
        out_act2 = torch.zeros((length,2),requires_grad=False)
        out_act2[:,0] = 1
        out_act2 = out_act2.unsqueeze(0)

        inputs3 = torch.zeros((length,2),requires_grad=False)
        inputs3[:,1] = 1
        inputs3 = inputs3.unsqueeze(0)
        out_val3 = torch.ones((length,1),requires_grad=False).unsqueeze(0)
        out_act3 = torch.zeros((length,2),requires_grad=False)
        out_act3[:,1] = 1
        out_act3 = out_act3.unsqueeze(0)

        inputs4 = torch.zeros((length,2),requires_grad=False).unsqueeze(0)
        inputs4[:,0] = 1
        inputs4 = inputs4.unsqueeze(0)
        out_val4 = torch.ones((length,1),requires_grad=False).unsqueeze(0)
        out_act4 = torch.zeros((length,2),requires_grad=False)
        out_act4[:,1] = 1
        out_act4 = out_act4.unsqueeze(0)
        # out_val  = torch.tensor(torch.sum(inputs,dim=-1), requires_grad=False)
        # out_act = torch.tensor(np.column_stack((torch.sum(inputs,dim=-1),torch.sum(inputs,dim=-1))),requires_grad=False)
        index = np.random.randint(0,3)

        inputs_pos = [inputs1,inputs2,inputs3,inputs4]
        out_act_pos = [out_act1,out_act2,out_act3,out_act4]
        out_val_pos = [out_val1,out_val2,out_val3,out_val4]
        out = [out_val_pos[index], out_act_pos[index]]
        return inputs_pos[index], out
