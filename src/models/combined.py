from torch.nn import Module
from .bpda import one_module_backward_identity


class Combined(Module):
    def __init__(self, module_inner, module_outer):
        super(Combined, self).__init__()
        self.module_inner = module_inner
        self.module_outer = module_outer

    def forward(self, input):
        return self.module_outer(self.module_inner(input))


class Combined_inner_backward_identity(Combined):
    def __init__(self, module_inner, module_outer):
        super(Combined_inner_backward_identity, self).__init__(
            module_inner, module_outer)
        self.frontend = one_module_backward_identity().apply

    def forward(self, input):
        return self.module_outer(self.frontend(input, self.module_inner))
