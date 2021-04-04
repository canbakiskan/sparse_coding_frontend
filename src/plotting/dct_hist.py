import numpy as np

from .parameters import get_arguments
from .utils.read_datasets import read_dataset
from .models.encoder import encoder_base_class

import matplotlib.pyplot as plt


def main():
    args = get_arguments()

    train_loader, test_loader = read_dataset(args)

    encoder = encoder_base_class(args)
    encoder.train()

    for batch_idx, (data, targets) in enumerate(train_loader):

        inner_products = encoder(data)
        plt.figure(figsize=(10, 5))

        plt.hist(
            inner_products[
                np.random.choice(args.neural_net.train_batch_size),
                :,
                np.random.choice(
                    int(
                        (args.dataset.img_shape[0] - args.defense.patch_size)
                        / args.defense.stride
                        + 1
                    )
                ),
                np.random.choice(
                    int(
                        (args.dataset.img_shape[0] - args.defense.patch_size)
                        / args.defense.stride
                        + 1
                    )
                ),
            ],
            50,
        )
        plt.savefig(f"hist_{args.dictionary.type}_{batch_idx}.pdf")
        plt.close()
        if batch_idx == 9:
            break


if __name__ == "__main__":
    main()
