import torch

import numpy as np
from ..parameters import get_arguments
from ..utils.read_datasets import read_dataset
import matplotlib.pyplot as plt
from ..utils.get_modules import get_frontend


def main():
    args = get_arguments()

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x_min = 0.0
    x_max = 1.0

    train_loader, test_loader = read_dataset(args)

    frontend = get_frontend(args)
    frontend.eval()

    random_indices = np.random.choice(len(test_loader), 10)

    for batch_idx, (data, targets) in enumerate(test_loader):
        if batch_idx in random_indices:
            reconstructions = frontend(data.to(device))

            plt.figure(figsize=(10, 5))

            img_index = np.random.choice(args.neural_net.test_batch_size)
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
