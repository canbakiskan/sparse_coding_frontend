import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import numpy as np
import logging

from .train_test_functions import (
    train_autoencoder_unsupervised,
    test_autoencoder_unsupervised,
)
from .parameters import get_arguments
from .utils.read_datasets import cifar10, tiny_imagenet, imagenette
from .models.autoencoders import *
from tqdm import tqdm
from .utils.namers import (
    autoencoder_ckpt_namer,
    autoencoder_log_namer,
)
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from .utils.get_modules import get_autoencoder


def main():
    args = get_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x_min = 0.0
    x_max = 1.0

    if args.dataset == "CIFAR10":
        train_loader, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        train_loader, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        train_loader, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    autoencoder = get_autoencoder(args)
    autoencoder.eval()

    random_indices = np.random.choice(len(test_loader), 10)

    for batch_idx, (data, targets) in enumerate(test_loader):
        if batch_idx in random_indices:
            reconstructions = autoencoder(data.to(device))

            plt.figure(figsize=(10, 5))

            img_index = np.random.choice(args.test_batch_size)
            plt.subplot(1, 2, 1)
            plt.imshow(data[img_index].detach().cpu().permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.imshow(
                reconstructions[img_index].detach().cpu().permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f"reconstructions_{batch_idx}.pdf")
            plt.close()


if __name__ == "__main__":
    main()
