import torch
from .encoders import take_top_T, take_top_T_dropout
from torch.nn.functional import dropout
from .ablation.gaussian_blur import gaussian_blur
from types import SimpleNamespace


class take_top_T_dropout_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, p):
        return take_top_T_dropout(x, T, p)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class take_top_T_dropout_BPDA_top_U(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, p, U):
        ctx.save_for_backward(take_top_T(x, U * T)*x)
        return take_top_T_dropout(x, T, p)

    @staticmethod
    def backward(ctx, grad_output):
        top_U, = ctx.saved_tensors
        return grad_output*top_U.sign(), None, None, None


class take_top_T_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        return take_top_T(x, T)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class take_top_T_BPDA_top_U(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, U):
        ctx.save_for_backward(take_top_T(x, U * T)*x)
        return take_top_T(x, T)

    @staticmethod
    def backward(ctx, grad_output):
        top_U, = ctx.saved_tensors
        return grad_output*top_U.sign(), None, None


class dropout_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        return dropout(x, p, training=True)

    @staticmethod
    def backward(ctx, grad_wrt_output):
        return grad_wrt_output, None


class one_module_BPDA_identity(torch.autograd.Function):
    def __init__(self):
        super(one_module_BPDA_identity, self).__init__()

    @staticmethod
    def forward(ctx, x, module):
        return module(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class one_module_BPDA_gaussianblur(torch.autograd.Function):
    blur_sigma = 0.0

    def __init__(self, args):
        super(one_module_BPDA_gaussianblur, self).__init__()
        one_module_BPDA_gaussianblur.blur_sigma = args.ablation_blur_sigma

    @ staticmethod
    def forward(ctx, x, module):
        ctx.save_for_backward(x)
        return module(x)

    @ staticmethod
    def backward(ctx, grad_outputs):

        # prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        args = SimpleNamespace()
        args.ablation_blur_sigma = one_module_BPDA_gaussianblur.blur_sigma
        blur = gaussian_blur(args).to("cuda")
        x, = ctx.saved_tensors
        grad_inputs, = torch.autograd.grad(
            blur(x), x, grad_outputs=grad_outputs)

        # torch._C.set_grad_enabled(prev)

        return grad_inputs, None


class activation_quantization_BPDA_smooth_step(torch.autograd.Function):
    steepness = 0.0

    def __init__(self, steepness):
        super(activation_quantization_BPDA_smooth_step, self).__init__()
        activation_quantization_BPDA_smooth_step.steepness = steepness

    @staticmethod
    def forward(ctx, x, l1_norms, jump):
        # x.shape: batchsize,nb_atoms,L,L

        x = x / l1_norms.view(1, -1, 1, 1)

        ctx.save_for_backward(x, jump)

        x = 0.5 * (torch.sign(x - jump) + torch.sign(x + jump))

        x = x * l1_norms.view(1, -1, 1, 1)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Use this if you want an approximation of the activation quantization 
        function in the backward pass. Uses the derivative of 
        0.5*(tanh(bpda_steepness*(x-jump))+tanh(bpda_steepness*(x+jump)))
        """
        x, jump = ctx.saved_tensors
        grad_input = None
        steepness = activation_quantization_BPDA_smooth_step.steepness

        def sech(x):
            return 1 / torch.cosh(x)

        del_out_over_del_in = 0.5 * steepness * (
            sech(steepness * (x - jump)) ** 2
            + sech(steepness * (x + jump)) ** 2
        )

        grad_input = del_out_over_del_in * grad_output

        return grad_input, None, None


class activation_quantization_BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l1_norms, jump):
        # x.shape: batchsize,nb_atoms,L,L

        x = x / l1_norms.view(1, -1, 1, 1)

        x = 0.5 * (torch.sign(x - jump) + torch.sign(x + jump))

        x = x * l1_norms.view(1, -1, 1, 1)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
