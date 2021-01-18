from torch.nn import Module
from .bpda import one_module_BPDA_identity, one_module_BPDA_gaussianblur


class Combined(Module):
    def __init__(self, module_inner, module_outer):
        super(Combined, self).__init__()
        self.module_inner = module_inner
        self.module_outer = module_outer

    def forward(self, input):
        return self.module_outer(self.module_inner(input))


class Combined_inner_BPDA_identity(Combined):
    def __init__(self, module_inner, module_outer):
        super(Combined_inner_BPDA_identity, self).__init__(
            module_inner, module_outer)
        self.frontend = one_module_BPDA_identity().apply

    def forward(self, input):
        return self.module_outer(self.frontend(input, self.module_inner))


class Combined_inner_BPDA_gaussianblur(Combined):
    def __init__(self, module_inner, module_outer, args):
        super(Combined_inner_BPDA_gaussianblur, self).__init__(
            module_inner, module_outer)
        self.frontend = one_module_BPDA_gaussianblur(args).apply

    def forward(self, input):
        return self.module_outer(self.frontend(input, self.module_inner))
