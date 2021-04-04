from os import path
from ..utils.namers import attack_file_namer
import numpy as np
from ..utils.read_datasets import(
    cifar10,
    tiny_imagenet,
    imagenette,
)
import matplotlib.pyplot as plt


def normalize(x):
    return (x-x.min())/(x.max()-x.min())


def main():

    from ..parameters import get_arguments

    args = get_arguments()

    if not path.exists(attack_file_namer(args)):
        print("Attack file was not found")
        exit()

    # this is just for the adversarial test below
    if args.dataset.name == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset.name == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset.name == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    clean_dataset = test_loader.dataset.data/255
    labels = np.array(test_loader.dataset.targets)

    attacked_dataset = np.load(attack_file_namer(args)).transpose(0, 2, 3, 1)
    perturbation = attacked_dataset-clean_dataset

    perturbation = perturbation[:200]

    print(
        f"Mean L^2 norm: {np.linalg.norm(perturbation.reshape(perturbation.shape[0], -1), ord=2, axis=-1).mean()}")
    exit()
    plt.figure(figsize=(3, 10))
    for i in range(10):
        plt.subplot(10, 3, 3*i+1)
        plt.imshow(clean_dataset[labels == i][0])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(10, 3, 3*i+2)
        plt.imshow(attacked_dataset[labels == i][0])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(10, 3, 3*i+3)
        plt.imshow(normalize(perturbation[labels == i][0]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('figs/attack_visualizations/' +
                path.splitext(path.basename(attack_file_namer(args)))[0]+'.pdf')


if __name__ == "__main__":
    main()
