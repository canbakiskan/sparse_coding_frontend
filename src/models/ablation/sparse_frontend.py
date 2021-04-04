from torch import nn
import torch.nn.functional as F
from ..encoder import take_top_T
from ..decoders import take_middle_of_img


class sparse_frontend(nn.Module):
    def __init__(self, args):
        super(sparse_frontend, self).__init__()

        self.image_size = args.dataset.img_shape[0]
        self.T = args.defense.top_T

        self.conv1 = nn.Conv2d(
            3, 100, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            100, 300, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.Conv2d(
            300, args.dictionary.nb_atoms, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.deconv1 = nn.ConvTranspose2d(
            args.dictionary.nb_atoms, 300, kernel_size=3, stride=1, padding=0, bias=True
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

    def dictionary_update_off(self):
        pass

    def set_BPDA_type(self, x):
        pass
