from .parameters import get_arguments
from .utils.namers import dict_file_namer
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import os


def main():

    args = get_arguments()

    (H, W, C) = args.defense.patch_shape

    dcts = np.zeros((H, W, C, H, W, C))

    for i in range(H):
        for j in range(W):
            for k in range(C):
                img = np.zeros((H, W, C))
                img[i, j, k] = 1.
                # each dct(dct(dct())) of 01000.. gives a column of DCT matrix
                # in that matrix each row is an atom (or DCT basis)
                dcts[:, :, :, i, j, k] = dct(
                    dct(dct(img, axis=0), axis=1), axis=2)

    dcts = dcts.reshape(H, W, C, -1)
    dcts = dcts.transpose(3, 0, 1, 2)
    dcts = dcts.reshape(-1, H*W*C)
    dcts = dcts.transpose()
    # resulting dcts is the DCT matrix with each row a DCT basis vector

    for i in range(H*W*C):
        dcts[i, :] /= np.linalg.norm(dcts[i, :])

    dict_filepath = dict_file_namer(args)

    if not os.path.exists(os.path.dirname(dict_filepath)):
        os.makedirs(os.path.dirname(dict_filepath))

    np.savez(dict_filepath, dict=dcts)

    dcts = (dcts-dcts.min())/(dcts.max()-dcts.min())\

    if args.dictionary.display:

        plt.figure(figsize=(15, 5))
        for i in range(H):
            for j in range(W):
                for k in range(C):
                    plt.subplot(H, W*C, W*C*i+W*k+j+1)
                    plt.imshow(
                        dcts[W*C*i+C*j+k, :].reshape(*args.defense.patch_shape))
                    plt.yticks([])
                    plt.xticks([])

        figures_dir = args.directory + "figs/"

        plt.savefig(figures_dir + "dct.pdf")

        plt.close()


if __name__ == "__main__":
    main()
