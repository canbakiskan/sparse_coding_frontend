import numpy as np
from .parameters import get_arguments
from .utils.read_datasets import cifar10, tiny_imagenet, imagenette
from tqdm import trange
import os

args = get_arguments()

if not os.path.exists(args.directory + f'data/image_distances/{args.dataset}'):
    os.makedirs(args.directory +
                f'data/image_distances/{args.dataset}', exist_ok=True)

if not os.path.exists(args.directory + f'data/image_distances/{args.dataset}/distances.npy'):

    if args.dataset == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    nbimgs = test_loader.dataset.data.shape[0]
    data = test_loader.dataset.data.reshape(
        nbimgs, -1)/255
    labels = test_loader.dataset.targets

    l2dist = np.zeros((nbimgs, nbimgs))

    for i in trange(nbimgs):
        for j in range(i+1, nbimgs):
            dist = np.linalg.norm(data[i]-data[j], ord=2)
            l2dist[i, j] = dist

    l2dist = l2dist+l2dist.T

    np.save(
        f'./data/image_distances/{args.dataset}/distances.npy', l2dist)

else:
    print(
        f'./data/image_distances/{args.dataset}/distances.npy already exists.')
