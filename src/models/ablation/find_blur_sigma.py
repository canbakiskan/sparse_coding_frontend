from tqdm import tqdm

from ...utils.get_modules import (
    get_classifier,
    get_autoencoder,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from .gaussian_blur import Gaussian_blur

from ...utils.read_datasets import cifar10, cifar10_from_file, tiny_imagenet, tiny_imagenet_from_file, imagenette, imagenette_from_file
import matplotlib.pyplot as plt
from ...utils import plot_settings


def main():

    from ...parameters import get_arguments

    args = get_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    autoencoder = get_autoencoder(args)
    autoencoder.eval()

    sigmas = torch.arange(0.4, 0.8, 0.025)
    blurs = []
    for i in range(len(sigmas)):
        blurs.append(Gaussian_blur(5, sigmas[i]).to(device))

    # this is just for the adversarial test below
    if args.dataset == "CIFAR10":
        train_loader, _ = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        train_loader, _ = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        train_loader, _ = imagenette(args)
    else:
        raise NotImplementedError

    after_autoencoder = torch.zeros(len(
        train_loader.dataset.targets), args.image_shape[2], args.image_shape[0], args.image_shape[1])

    after_blur = torch.zeros(len(sigmas), *after_autoencoder.shape)

    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, desc="Attack progress", leave=False)
    ):

        data = data.to(device)

        after_autoencoder[
            batch_idx
            * args.train_batch_size: (batch_idx + 1)
            * args.train_batch_size,
        ] = autoencoder(data).detach().cpu()

        for i in range(len(sigmas)):
            after_blur[i,
                       batch_idx
                       * args.train_batch_size: (batch_idx + 1)
                       * args.train_batch_size,
                       ] = blurs[i](data).detach().cpu()

    mse = MSELoss()
    loss = torch.zeros_like(sigmas)

    for i in range(len(sigmas)):
        loss[i] = mse(after_autoencoder, after_blur[i])

    plt.figure(figsize=(10, 5))
    plt.plot(sigmas, loss)
    plt.xlabel(r"$\sigma$")
    plt.ylabel("MSE Loss")
    plt.savefig(args.directory + f"figs/blur_sigmas_very_fine.pdf")
    plt.close()


if __name__ == "__main__":
    main()
