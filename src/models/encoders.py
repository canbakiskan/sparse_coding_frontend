import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout


def take_top_T(x, T):
    values, indices = torch.topk(x.abs(), T, dim=1)
    dropped = torch.zeros_like(x).scatter(1, indices, values)
    return dropped


def take_top_T_dropout(x, T, p, seed=None):
    if seed:
        torch.manual_seed(seed)

    x = dropout(take_top_T(x, T), p=p, training=True)
    x *= 1 - p

    return x


class encoder_base_class(nn.Module):
    def __init__(self, args):
        super(encoder_base_class, self).__init__()

        from ..utils.get_modules import get_dictionary

        dictionary = get_dictionary(args)
        self.set_l1_norms(dictionary)
        self.set_jump(args.activation_beta * args.defense_epsilon)
        self.conv = nn.Conv2d(
            3,
            args.dict_nbatoms,
            kernel_size=args.defense_patchsize,
            stride=args.defense_stride,
            padding=0,
            bias=False,
        )
        self.conv.weight.data = (
            dictionary.t()
            .reshape(
                args.dict_nbatoms, args.defense_patchsize, args.defense_patchsize, 3
            )
            .permute(0, 3, 1, 2)
        )
        self.conv.weight.requires_grad = False

    def __getattr__(self, key):
        if key == "dictionary":
            return self.conv.weight
        else:
            return super(encoder_base_class, self).__getattr__(key)

    def forward(self, x):
        return self.conv(x)

    def set_jump(self, jump):

        if jump is not None:
            if isinstance(jump, torch.Tensor):
                self.jump = nn.Parameter(jump.float())
            else:
                self.jump = nn.Parameter(torch.tensor(jump, dtype=torch.float))
            self.jump.requires_grad = False

    def set_l1_norms(self, dictionary):
        if dictionary is not None:
            if isinstance(dictionary, torch.Tensor):
                self.l1_norms = nn.Parameter(
                    dictionary.float().abs().sum(dim=0))
            else:
                self.l1_norms = nn.Parameter(
                    torch.tensor(
                        dictionary, dtype=torch.float).abs().sum(dim=0)
                )
            self.l1_norms.requires_grad = False


class quant_encoder(encoder_base_class):
    def __init__(self, args):
        super(quant_encoder, self).__init__(args)
        if args.attack_quantization_BPDA_steepness == 0.0:
            from .bpda import activation_quantization_BPDA_identity
            self.activation = activation_quantization_BPDA_identity().apply
        else:
            from .bpda import activation_quantization_BPDA_smooth_step
            self.activation = activation_quantization_BPDA_smooth_step(
                args.attack_quantization_BPDA_steepness).apply

    def forward(self, x):
        super(quant_encoder, self).forward(x)
        x = self.activation(x, self.l1_norms, self.jump)
        return x


class top_T_encoder(encoder_base_class):
    def __init__(self, args, BPDA_type="maxpool_like"):
        super(top_T_encoder, self).__init__(args)

        self.T = args.top_T
        self.set_BPDA_type(BPDA_type)

    def set_BPDA_type(self, BPDA_type):
        self.BPDA_type = BPDA_type
        from .bpda import take_top_T_BPDA_identity

        if self.BPDA_type == "maxpool_like":
            self.take_top_T = take_top_T
        elif self.BPDA_type == "identity":
            self.take_top_T = take_top_T_BPDA_identity().apply

    def forward(self, x):
        x = super(top_T_encoder, self).forward(x)
        x = self.take_top_T(x, self.T)
        return x


class top_T_quant_encoder(encoder_base_class):
    def __init__(self, args, BPDA_type="maxpool_like"):
        super(top_T_quant_encoder, self).__init__(args)

        self.T = args.top_T
        if args.attack_quantization_BPDA_steepness == 0.0:
            from .bpda import activation_quantization_BPDA_identity
            self.activation = activation_quantization_BPDA_identity().apply
        else:
            from .bpda import activation_quantization_BPDA_smooth_step
            self.activation = activation_quantization_BPDA_smooth_step(
                args.attack_quantization_BPDA_steepness).apply

        self.set_BPDA_type(BPDA_type)

    def set_BPDA_type(self, BPDA_type):
        self.BPDA_type = BPDA_type
        from .bpda import take_top_T_BPDA_identity

        if self.BPDA_type == "maxpool_like":
            self.take_top_T = take_top_T
        elif self.BPDA_type == "identity":
            self.take_top_T = take_top_T_BPDA_identity().apply

    def forward(self, x):
        x = super(top_T_quant_encoder, self).forward(x)
        x = self.take_top_T(x, self.T)
        x = self.activation(x, self.l1_norms, self.jump)
        return x


class top_T_dropout_encoder(encoder_base_class):
    def __init__(self, args, BPDA_type="maxpool_like"):
        super(top_T_dropout_encoder, self).__init__(args)
        self.T = args.top_T
        self.p = args.dropout_p
        self.set_BPDA_type(BPDA_type)
        self.fixed_seed = False

    def set_BPDA_type(self, BPDA_type):
        self.BPDA_type = BPDA_type
        from .bpda import take_top_T_dropout_BPDA_identity

        if self.BPDA_type == "maxpool_like":
            self.take_top_T_dropout = take_top_T_dropout
        elif self.BPDA_type == "identity":
            self.take_top_T_dropout = take_top_T_dropout_BPDA_identity().apply

    def fix_seed(self, is_fixed=True):
        self.fixed_seed = True

    def forward(self, x):
        x = super(top_T_dropout_encoder, self).forward(x)
        if self.fixed_seed:
            x = self.take_top_T_dropout(x, self.T, self.p, 20200605)
        else:
            x = self.take_top_T_dropout(x, self.T, self.p)

        return x


class top_T_dropout_quant_encoder(encoder_base_class):
    def __init__(self, args, BPDA_type="maxpool_like"):
        super(top_T_dropout_quant_encoder, self).__init__(args)
        self.T = args.top_T
        self.p = args.dropout_p
        if args.attack_quantization_BPDA_steepness == 0.0:
            from .bpda import activation_quantization_BPDA_identity
            self.activation = activation_quantization_BPDA_identity().apply
        else:
            from .bpda import activation_quantization_BPDA_smooth_step
            self.activation = activation_quantization_BPDA_smooth_step(
                args.attack_quantization_BPDA_steepness).apply

        self.set_BPDA_type(BPDA_type)
        self.fixed_seed = False

    def set_BPDA_type(self, BPDA_type):
        self.BPDA_type = BPDA_type
        from .bpda import take_top_T_dropout_BPDA_identity

        if self.BPDA_type == "maxpool_like":
            self.take_top_T_dropout = take_top_T_dropout
        elif self.BPDA_type == "identity":
            self.take_top_T_dropout = take_top_T_dropout_BPDA_identity().apply

    def fix_seed(self, is_fixed=True):
        self.fixed_seed = True

    def forward(self, x):
        x = super(top_T_dropout_quant_encoder, self).forward(x)
        if self.fixed_seed:
            x = self.take_top_T_dropout(x, self.T, self.p, 20200605)
        else:
            x = self.take_top_T_dropout(x, self.T, self.p)
        x = self.activation(x, self.l1_norms, self.jump)

        return x


encoder_dict = {
    "quant_encoder": quant_encoder,
    "top_T_encoder": top_T_encoder,
    "top_T_quant_encoder": top_T_quant_encoder,
    "top_T_dropout_encoder": top_T_dropout_encoder,
    "top_T_dropout_quant_encoder": top_T_dropout_quant_encoder,
}
