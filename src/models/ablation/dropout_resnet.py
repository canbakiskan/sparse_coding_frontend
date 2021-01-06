from ..resnet import ResNet, ResidualUnit
from torch import nn
import torch.nn.functional as F


class dropout_ResNet(ResNet):
    def __init__(self,  dropout_p, nb_filters, num_outputs=10):
        super(dropout_ResNet, self).__init__(num_outputs)

        self.dropout_p = dropout_p

        filters = [nb_filters, 16, 32, 64]
        strides = [1, 2, 2]

        self.conv1 = nn.Conv2d(3, nb_filters, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.block1 = self.build_block(
            ResidualUnit, 5, filters[0], filters[1], strides[0], True)

    def forward(self, x):

        out = self.norm(x)
        out = self.conv1(out)
        out = F.dropout(out, p=self.dropout_p, training=True)
        out *= 1 - self.dropout_p
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.leaky_relu(self.bn1(out), 0.1)
        out = F.avg_pool2d(out, out.size(-1)).squeeze()
        out = self.linear(out)

        return out
