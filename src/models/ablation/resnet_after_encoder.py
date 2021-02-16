from ..resnet import ResNet, ResidualUnit
from torch import nn
import torch.nn.functional as F


class ResNet_after_encoder(ResNet):
    def __init__(self, nb_filters, num_outputs=10):
        super(ResNet_after_encoder, self).__init__(num_outputs)

        filters = [16, 16, 32, 64]
        strides = [1, 2, 2]

        self.conv1 = nn.Conv2d(nb_filters, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):

        out = self.conv1(x)
        out = self.block1(out)

        # odd pixel size causes issues with skip connection adding, pad one side
        pad = nn.ZeroPad2d((0, 1, 0, 1))
        out = pad(out)

        out = self.block2(out)
        out = self.block3(out)
        out = F.leaky_relu(self.bn1(out), 0.1)
        out = F.avg_pool2d(out, out.size(-1)).squeeze()
        out = self.linear(out)

        return out
