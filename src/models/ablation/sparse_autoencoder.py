from torch import nn
import torch.nn.functional as F
from ..encoders import take_top_T
from ..decoders import take_middle_of_img


class sparse_autoencoder(nn.Module):
    def __init__(self, args):
        super(sparse_autoencoder, self).__init__()

        self.image_size = args.image_shape[0]
        self.T = args.top_T

        self.conv1 = nn.Conv2d(
            3, 100, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            100, 300, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.Conv2d(
            300, args.dict_nbatoms, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.deconv1 = nn.ConvTranspose2d(
            args.dict_nbatoms, 300, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.deconv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.deconv3 = nn.ConvTranspose2d(
            100, 3, kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_top_T(out, self.T)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = take_middle_of_img(out, self.image_size)
        return out.clamp(0.0, 1.0)

    def encoder_no_update(self):
        pass

    def set_BPDA_type(self, x):
        pass
