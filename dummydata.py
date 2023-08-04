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
