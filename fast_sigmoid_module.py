import torch

def fast_sigmoid_forward(ctx, input_, slope):
    ctx.save_for_backward(input_)
    ctx.slope = slope
    out = (input_ > 0).float()
    return out

def fast_sigmoid_backward(ctx, grad_output):
    (input_,) = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
    return grad, None

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, slope=25):
        return fast_sigmoid_forward(ctx, input_, slope)

    @staticmethod
    def backward(ctx, grad_output):
        return fast_sigmoid_backward(ctx, grad_output)

def fast_sigmoid(slope=25):
    def inner(x):
        return FastSigmoid.apply(x, slope)
    return inner
