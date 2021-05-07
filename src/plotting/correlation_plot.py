from os.path import join
import matplotlib.pyplot as plt
import torch
from ..models.resnet import ResNetWide
from ..utils.read_datasets import cifar10
from ..utils.plot_settings import *
from ..utils.get_modules import get_frontend
import numpy as np
from ..parameters import get_arguments

args = get_arguments()

device = "cuda"

classifier = ResNetWide(num_outputs=10).to(device)

classifier.load_state_dict(
    torch.load(join(
        'checkpoints', 'classifiers', 'CIFAR10', 'resnetwide_sgd_cyc_0.0500_NT_ep_100.pt'),
        map_location=torch.device(device),
    )
)
classifier.to(device)

train_loader, test_loader = cifar10(args)
frontend = get_frontend(args)

plt.figure(figsize=(10, 10))
weights = classifier.conv1.weight.clone().reshape(160, -1)
weights /= torch.norm(weights, dim=1, keepdim=True)
weights = weights.detach().cpu().numpy()
asd = weights @ weights.transpose()
plt.imshow(
    asd,
    cmap=cm,
    vmin=-np.abs(asd).max(),
    vmax=np.abs(asd).max(),
    interpolation="nearest",
)
plt.xticks([])
plt.yticks([])
plt.savefig(join('figs', 'inner_cnn.pdf'))
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(asd.flatten(), 50)
plt.savefig(join('figs', 'hist_cnn.pdf'))
plt.close()

nb_cols = 2
nb_rows = 5
plt.figure(figsize=(10 * nb_cols, 4 * nb_rows))
for i in range(nb_cols * nb_rows):
    plt.subplot(nb_rows, nb_cols, i + 1)
    img_index = np.random.choice(50000)
    print(f"image: {img_index},", end=" ")
    img, _ = train_loader.dataset[img_index]
    img = img.to(device)

    classifier_out = classifier.norm(img.unsqueeze(0))
    classifier_out = classifier.conv1(classifier_out)
    # classifier_out = classifier.conv1(img.unsqueeze(0))

    classifier_out /= torch.norm(classifier.conv1.weight.view(160, -1), dim=1).view(
        1, 160, 1, 1
    )

    frontend_out = frontend.encoder.conv(img.unsqueeze(0))
    # print(f"===={out[0,0,0,0]}")

    # xlims = [-2.6, 2.6]
    patch_index = (np.random.choice(range(1, 30, 2)),
                   np.random.choice(range(1, 30, 2)))
    # patch_index = (22, 23)
    print(f"patch: {patch_index}")
    classifier_patch = classifier_out.squeeze().detach().cpu().numpy()[
        :, patch_index]
    frontend_patch = (
        frontend_out.squeeze()
        .detach()
        .cpu()
        .numpy()[:, patch_index[0] // 2, patch_index[1] // 2]
    )
    abs_max = max(np.abs(classifier_patch).max(), np.abs(frontend_patch).max())
    xlims = (-abs_max, abs_max)

    bin_edges = np.linspace(*xlims, 50)

    hist, _ = np.histogram(classifier_patch, bin_edges, density=True)
    # breakpoint()
    color, edgecolor = ("orange", "darkorange")

    plt.bar(
        bin_edges[:-1] + np.diff(bin_edges) / 2,
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.5,
        edgecolor="none",
        color=color,
    )
    plt.step(
        np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
        np.array([0, *hist, 0]),
        label=r"CNN $1^{st}$ layer",
        where="pre",
        color=edgecolor,
    )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_yaxis().set_visible(False)

    # print(f"===={out[0,0,0,0]}")

    hist, _ = np.histogram(frontend_patch, bin_edges, density=True)

    # bin_edges, hist = np.histogram(out.squeeze().detach().cpu().numpy()[
    #     :, np.random.choice(32), np.random.choice(32)], 50)

    color, edgecolor = ("steelblue", "steelblue")

    plt.bar(
        bin_edges[:-1] + np.diff(bin_edges) / 2,
        hist,
        width=(bin_edges[1] - bin_edges[0]),
        alpha=0.5,
        edgecolor="none",
        color=color,
    )
    plt.step(
        np.array([*bin_edges, bin_edges[-1] + (bin_edges[1] - bin_edges[0])]),
        np.array([0, *hist, 0]),
        label=r"Overcomplete dictionary",
        where="pre",
        color=edgecolor,
    )
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_yaxis().set_visible(False)


# ax = plt.gca()

# ax.xaxis.set_major_locator(ticker.MultipleLocator(18))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(4.5))
# ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=225, decimals=0))
# plt.gca().get_xaxis().set_major_formatter(
#     FuncFormatter(lambda x, p: format((x / 225), ".2"))
# )

# fontsize = 21
# plt.xlabel("Correlation value", fontsize=fontsize)
# plt.ylabel("Histogram density", fontsize=fontsize)
# plt.xlim(xlims)
# plt.legend(loc="upper center", fontsize=fontsize)

plt.tight_layout()

plt.savefig(join('figs', 'more_correlations_normalized.pdf'))
plt.close()
