import numpy as np
from .parameters import get_arguments
from .utils.read_datasets import cifar10, tiny_imagenet, imagenette
from tqdm import trange

args = get_arguments()

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

# closest_adv_idx = np.zeros(10000)

# for i in trange(10000):
#     min_index = -1
#     min_val = np.inf
#     for j in range(i+1, 10000):
#         dist = np.linalg.norm(data[i]-data[j], ord=2)
#         if dist < min_val and labels[i] != labels[j]:
#             min_index = j
#             min_val = dist
#     closest_adv_idx[i] = min_index

# np.save(
#     f'./data/image_distances/{args.dataset}/closest_adv_indices.npy', closest_adv_idx)

l2dist = np.zeros((nbimgs, nbimgs))

for i in trange(nbimgs):
    for j in range(i+1, nbimgs):
        dist = np.linalg.norm(data[i]-data[j], ord=2)
        l2dist[i, j] = dist

np.save(
    f'./data/image_distances/{args.dataset}/distances.npy', l2dist+l2dist.T)
