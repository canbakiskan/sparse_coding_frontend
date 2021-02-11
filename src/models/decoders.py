import torch
from torch import nn
import torch.nn.functional as F


def take_middle_of_img(x, image_size):  # assumes square images
    width = x.shape[-1]
    start_index = (width - image_size) // 2
    return x[:, :, start_index: start_index + image_size, start_index: start_index + image_size]


class default_decoder(nn.Module):
    def __init__(self, args):

        super(default_decoder, self).__init__()
        self.image_size = args.image_shape[0]
        self.conv1 = nn.ConvTranspose2d(
            args.dict_nbatoms, 300, kernel_size=4, stride=2, padding=0, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(
            100, 3, kernel_size=3, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_middle_of_img(out, self.image_size)
        return out.clamp(0.0, 1.0)


class small_decoder(nn.Module):
    def __init__(self, args):

        super(small_decoder, self).__init__()
        self.image_size = args.image_shape[0]
        self.conv1 = nn.ConvTranspose2d(
            args.dict_nbatoms, 30, kernel_size=4, stride=2, padding=0, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            30, 10, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(
            10, 3, kernel_size=3, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_middle_of_img(out, self.image_size)
        return out.clamp(0.0, 1.0)


class deep_decoder(nn.Module):
    def __init__(self, args):

        super(deep_decoder, self).__init__()
        self.image_size = args.image_shape[0]
        self.conv1 = nn.ConvTranspose2d(
            args.dict_nbatoms, 300, kernel_size=4, stride=2, padding=0, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(
            100, 30, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv4 = nn.ConvTranspose2d(
            30, 10, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv5 = nn.ConvTranspose2d(
            10, 3, kernel_size=3, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.interpolate(out, size=(self.image_size+2), mode="bicubic")
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = take_middle_of_img(out, self.image_size)
        return out.clamp(0.0, 1.0)


class resize_decoder(nn.Module):
    def __init__(self, args):

        super(resize_decoder, self).__init__()
        self.image_size = args.image_shape[0]
        self.conv1 = nn.ConvTranspose2d(
            args.dict_nbatoms, 300, kernel_size=3, stride=1, padding=0, bias=True
        )

        self.conv2 = nn.ConvTranspose2d(
            300, 100, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(
            100, 3, kernel_size=3, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.interpolate(out, size=(self.image_size+2), mode="bicubic")
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = take_middle_of_img(out, self.image_size)
        return out.clamp(0.0, 1.0)


class identity_decoder(nn.Module):
    def forward(x):
        return x


decoder_dict = {"default_decoder": default_decoder,
                "resize_decoder": resize_decoder,
                "deep_decoder": deep_decoder,
                "small_decoder": small_decoder,
                "identity_decoder": identity_decoder}
