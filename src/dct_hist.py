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
from .models.encoders import encoder_base_class
from tqdm import tqdm
from .utils.namers import (
    autoencoder_ckpt_namer,
    autoencoder_log_namer,
)
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def main():
    args = get_arguments()

    if args.dataset == "CIFAR10":
        train_loader, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        train_loader, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        train_loader, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    encoder = encoder_base_class(args)
    encoder.train()

    for batch_idx, (data, targets) in enumerate(train_loader):

        inner_products = encoder(data)
        plt.figure(figsize=(10, 5))

        plt.hist(inner_products[
            np.random.choice(args.train_batch_size),
            :,
            np.random.choice(int((args.image_shape[0] -
                                  args.defense_patchsize) / args.defense_stride+1)),
            np.random.choice(int((args.image_shape[0] - args.defense_patchsize) / args.defense_stride+1))], 50)
        plt.savefig(f"hist_{args.dict_type}_{batch_idx}.pdf")
        plt.close()
        if batch_idx == 9:
            break


if __name__ == "__main__":
    main()
