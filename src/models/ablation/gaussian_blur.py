import torch
import torch.nn as nn
from math import pi


class gaussian_blur(nn.Module):
    def __init__(self, args):
        super(gaussian_blur, self).__init__()
        self.kernel_size = 5
        self.sigma = args.ablation_blur_sigma

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(
            self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1)/2.
        variance = self.sigma**2.

        gaussian_kernel = (1./(2.*pi*variance)) *\
            torch.exp(
            -torch.sum((xy_grid - mean)**2., dim=-1) /
            (2*variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(
            1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=3, out_channels=3,
                                         kernel_size=self.kernel_size, groups=3, bias=False, padding=0)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

        self.padding = nn.ReflectionPad2d(self.kernel_size//2)

    def forward(self, input):
        return self.gaussian_filter(self.padding(input))

    def dictionary_update_off(self):
        pass

    def set_BPDA_type(self, x):
        pass
