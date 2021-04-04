import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout


def top_T(x, T):
    values, indices = torch.topk(x.abs(), T, dim=1)
    dropped = torch.zeros_like(x).scatter(1, indices, values)
    dropped = dropped * x.sign()  # <-- this should be included
    return dropped


def dropout(x, p, seed=None):
    if seed:
        torch.manual_seed(seed)

    x = dropout(x, p=p, training=True)
    x *= 1 - p

    return x


def activation(x, l1_norms, jump):
    # x.shape: batchsize,nb_atoms,L,L

    x = x / l1_norms.view(1, -1, 1, 1)

    x = 0.5 * (torch.sign(x - jump) + torch.sign(x + jump))

    x = x * l1_norms.view(1, -1, 1, 1)

    return x


class noisy():
    def __init__(self, args):
        self.train_noise_level = args.defense.assumed_budget*args.defense.train_noise_gamma
        self.test_noise_level = args.defense.assumed_budget*args.defense.test_noise_gamma

    def add_noise(self, x, seed=None):
        if seed:
            torch.manual_seed(seed)

        noise = torch.rand_like(x)
        noise = noise / torch.norm(noise, p=1, dim=(2, 3)
                                   ).unsqueeze(-1).unsqueeze(-1)

        if self.training:
            noise = noise * self.train_noise_level * \
                self.l1_norms.view(1, -1, 1, 1)
        else:
            noise = noise * self.test_noise_level * \
                self.l1_norms.view(1, -1, 1, 1)

        return x+noise


class encoder(nn.Module, noisy):
    def __init__(self, args):
        super(encoder, self).__init__()

        self.frontend_arch = args.defense.frontend_arch
        self.seed = args.seed

        self.conv = nn.Conv2d(
            3,
            args.dictionary.nb_atoms,
            kernel_size=args.defense.patch_size,
            stride=args.defense.stride,
            padding=0,
            bias=False,
        )

        if not args.ablation.no_dictionary:
            from ..utils.get_modules import get_dictionary

            dictionary = get_dictionary(args)
            self.conv.weight.data = (
                dictionary.t()
                .reshape(
                    args.dictionary.nb_atoms, args.defense.patch_size, args.defense.patch_size, 3
                )
                .permute(0, 3, 1, 2)
            )
            self.conv.weight.requires_grad = False

        if "noisy" in self.frontend_arch:
            noisy.__init__(self, args)

        if "top_T" in self.frontend_arch:
            self.T = args.defense.top_T
            self.top_T_backward = args.defense.attack_top_T_backward

            if self.top_T_backward == "top_U":
                from .bpda import top_T_backward_top_U
                self.top_T = top_T_backward_top_U().apply
                self.U = args.adv_testing.top_U
            elif self.top_T_backward == "identity":
                from .bpda import top_T_backward_identity
                self.top_T = top_T_backward_identity().apply
            else:
                self.top_T = top_T

        if "dropout" in self.frontend_arch:
            self.p = args.defense.dropout_p
            self.dropout_backward = args.defense.attack_dropout_backward

            if self.dropout_backward == "identity":
                from .bpda import dropout_backward_identity
                self.top_T_dropout = dropout_backward_identity().apply
            else:
                self.top_T_dropout = dropout

        if "activation" in self.frontend_arch:
            self.set_jump(args.defense.activation_beta *
                          args.defense.assumed_budget)
            self.set_l1_norms(self.conv.weight.data.permute(
                0, 2, 3, 1).reshape(args.dictionary.nb_atoms, -1).t())
            self.activation_backward = args.defense.attack_activation_backward

            if self.activation_backward == "identity":
                from .bpda import activation_backward_identity
                self.activation = activation_backward_identity().apply
            elif self.activation_backward == "smooth":
                from .bpda import activation_backward_smooth
                self.activation = activation_backward_smooth(
                    args.adv_testing.activation_backward_steepness).apply
            else:
                self.activation = dropout

    def __getattr__(self, key):
        if key == "dictionary":
            return self.conv.weight
        else:
            return super(encoder, self).__getattr__(key)

    def forward(self, x):
        if self.conv.weight.requires_grad:  # this means ablation.no_dictionary=True
            self.update_l1_norms()

        x = self.conv(x)

        if "noisy" in self.frontend_arch:
            x = self.add_noise(x, self.seed)

        if "top_T" in self.frontend_arch:
            if self.top_T_backward == "top_U":
                x = self.top_T(x, self.T, self.U)
            else:
                x = self.top_T(x, self.T)

        if "dropout" in self.frontend_arch:
            x = self.top_T_dropout(x, self.T, self.p, self.seed)

        if "activation" in self.frontend_arch:
            x = self.activation(x, self.l1_norms, self.jump)

    def set_jump(self, jump):
        if jump is not None:
            if isinstance(jump, torch.Tensor):
                self.jump = nn.Parameter(jump.float())
            else:
                self.jump = nn.Parameter(torch.tensor(jump, dtype=torch.float))
            self.jump.requires_grad = False

    def set_l1_norms(self, dictionary):
        # nn.parameter allows torch.save to save on file
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

    def update_l1_norms(self):
        # for cases where the dictionary changes, we need to update l1 norms
        self.l1_norms.data = self.conv.weight.data.abs().sum(dim=(1, 2, 3))
