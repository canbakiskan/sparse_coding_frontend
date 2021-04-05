import torch
from .encoder import top_T, dropout, activation
from torch.nn.functional import dropout


class top_T_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        return top_T(x, T)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class top_T_backward_top_U(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, U):
        ctx.save_for_backward(top_T(x, U))
        return top_T(x, T)

    @staticmethod
    def backward(ctx, grad_output):
        top_U, = ctx.saved_tensors
        return grad_output*top_U.sign().abs(), None, None


class dropout_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        return dropout(x, p, training=True)

    @staticmethod
    def backward(ctx, grad_wrt_output):
        return grad_wrt_output, None


class one_module_backward_identity(torch.autograd.Function):
    def __init__(self):
        super(one_module_backward_identity, self).__init__()

    @staticmethod
    def forward(ctx, x, module):
        return module(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class activation_backward_smooth(torch.autograd.Function):
    steepness = 0.0

    def __init__(self, steepness):
        super(activation_backward_smooth, self).__init__()
        activation_backward_smooth.steepness = steepness

    @staticmethod
    def forward(ctx, x, l1_norms, jump):
        # x.shape: batchsize,nb_atoms,L,L
        ctx.save_for_backward(x / l1_norms.view(1, -1, 1, 1), jump)
        return activation(x, l1_norms, jump)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Use this if you want an approximation of the activation 
        function in the backward pass. Uses the derivative of 
        0.5*(tanh(backward_steepness*(x-jump))+tanh(backward_steepness*(x+jump)))
        """
        x, jump = ctx.saved_tensors
        grad_input = None
        steepness = activation_backward_smooth.steepness

        def sech(x):
            return 1 / torch.cosh(x)

        del_out_over_del_in = 0.5 * steepness * (
            sech(steepness * (x - jump)) ** 2
            + sech(steepness * (x + jump)) ** 2
        )

        grad_input = del_out_over_del_in * grad_output

        return grad_input, None, None


class activation_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l1_norms, jump):
        # x.shape: batchsize,nb_atoms,L,L
        return activation(x, l1_norms, jump)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
