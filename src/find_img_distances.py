import numpy as np
from .parameters import get_arguments
from .utils.read_datasets import read_dataset
from tqdm import trange
import os

args = get_arguments()

if not os.path.exists(os.path.join(args.directory, 'data', 'image_distances', args.dataset.name, 'closest_img_indices.npy')):

    _, test_loader = read_dataset(args)(args)

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

    img_distances_idx = np.argsort(l2dist).astype(np.uint16)

    if not os.path.exists(os.path.join(args.directory, 'data', 'image_distances', args.dataset.name)):
        os.makedirs(os.path.join(args.directory, 'data',
                                 'image_distances', args.dataset.name), exist_ok=True)

    np.save(os.path.join(args.directory,
                         'data', 'image_distances', args.dataset.name, 'closest_img_indices.npy'), img_distances_idx[:, :1000])

else:
    print(os.path.join(args.directory,
                       'data', 'image_distances', args.dataset.name, 'closest_img_indices.npy') + ' already exists.')
