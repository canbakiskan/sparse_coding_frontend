from matplotlib.lines import Line2D
from matplotlib import rcParams
from os import path
from tqdm import tqdm
from importlib import import_module

from nips2020.src.utils.get_modules import (
    get_classifier,
    get_autoencoder,
)

import numpy as np
import torch
from nips2020.src.models.combined import Combined
from nips2020.src.utils.read_datasets import cifar10, cifar10_blackbox
from nips2020.src.parameters import get_arguments
from nips2020.src.models.ensemble import Ensemble_post_softmax
import matplotlib.pyplot as plt
import nips2020.src.utils.plot_settings

args = get_arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args.defense_patchshape = (args.defense_patchsize, args.defense_patchsize, 3)
args.image_shape = (32, 32, 3)


errorbar_N = 100

# for is_S in [False, True]:
#     for clean_attack in ["clean", "attacked"]:

#         args.autoencoder_train_supervised = is_S

#         classifier = get_classifier(args)
#         autoencoder = get_autoencoder(args)

#         model = Combined(autoencoder, classifier)

#         model = model.to(device)
#         model.eval()

#         ensemble_model = Ensemble_post_softmax(model, 1)

#         ensemble_model.eval()
#         cross_ent = torch.nn.CrossEntropyLoss()

#         if clean_attack == "clean":
#             _, test_loader = cifar10(args)
#         elif clean_attack == "attacked":
#             args.attack_blackbox_type = "substitute_ii"
#             test_loader = cifar10_blackbox(args)

#         for p in ensemble_model.parameters():
#             p.requires_grad = False

#         test_correct = np.zeros((args.ensemble_E, errorbar_N))
#         with torch.no_grad():
#             for data, target in tqdm(test_loader, leave=False):
#                 if isinstance(data, list):
#                     data = data[0]
#                     target = target[0]

#                 data, target = data.to(device), target.to(device)

#                 for E in range(1, 11):
#                     ensemble_model.ensemble_E = E
#                     for j in range(errorbar_N):
#                         output = ensemble_model(data)
#                         pred = output.argmax(dim=1, keepdim=True)
#                         test_correct[E - 1, j] += (
#                             pred.eq(target.view_as(pred)).sum().item()
#                         )

#         if clean_attack == "clean":
#             if is_S:
#                 clean_S = test_correct
#                 np.save("/home/adv/nips2020/clean_S", clean_S)
#             else:
#                 clean_US = test_correct
#                 np.save("/home/adv/nips2020/clean_US", clean_US)
#         elif clean_attack == "attacked":
#             if is_S:
#                 attacked_S = test_correct
#                 np.save("/home/adv/nips2020/attacked_S", attacked_S)
#             else:
#                 attacked_US = test_correct
#                 np.save("/home/adv/nips2020/attacked_US", attacked_US)

# outputs = np.zeros((args.ensemble_E, 10000, 10))

# for is_S in [False, True]:
#     for clean_attack in ["clean", "attacked"]:

#         args.autoencoder_train_supervised = is_S

#         classifier = get_classifier(args)
#         autoencoder = get_autoencoder(args)

#         model = Combined(autoencoder, classifier)

#         model = model.to(device)
#         model.eval()

#         model.eval()
#         cross_ent = torch.nn.CrossEntropyLoss()

#         if clean_attack == "clean":
#             _, test_loader = cifar10(args)
#         elif clean_attack == "attacked":
#             args.attack_blackbox_type = "substitute_ii"
#             test_loader = cifar10_blackbox(args)

#         for p in model.parameters():
#             p.requires_grad = False

#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(tqdm(test_loader, leave=False)):
#                 if isinstance(data, list):
#                     data = data[0]
#                     target = target[0]

#                 data, target = data.to(device), target.to(device)

#                 for E in range(args.ensemble_E):

#                     outputs[
#                         E,
#                         batch_idx
#                         * args.test_batch_size : (batch_idx + 1)
#                         * args.test_batch_size,
#                     ] = (model(data).detach().cpu().numpy())

#         if clean_attack == "clean":
#             if is_S:
#                 clean_S = outputs
#                 np.save("/home/adv/nips2020/clean_S", clean_S)
#             else:
#                 clean_US = outputs
#                 np.save("/home/adv/nips2020/clean_US", clean_US)
#         elif clean_attack == "attacked":
#             if is_S:
#                 attacked_S = outputs
#                 np.save("/home/adv/nips2020/attacked_S", attacked_S)
#             else:
#                 attacked_US = outputs
#                 np.save("/home/adv/nips2020/attacked_US", attacked_US)


clean_S = np.load("/home/adv/nips2020/clean_S.npy")
clean_US = np.load("/home/adv/nips2020/clean_US.npy")
attacked_S = np.load("/home/adv/nips2020/attacked_S.npy")
attacked_US = np.load("/home/adv/nips2020/attacked_US.npy")
_, test_loader = cifar10(args)
targets = torch.Tensor(test_loader.dataset.targets)

clean_S_accuracy = np.zeros((10, errorbar_N))
for E in range(1, 11):
    for N in range(errorbar_N):
        # clean_S shape: 100,10000,10
        softmax = torch.softmax(
            torch.Tensor(
                clean_S[np.random.choice(
                    args.ensemble_E, size=E, replace=False)]
            ),
            dim=2,
        )
        softmax = softmax.mean(axis=0)
        pred = softmax.argmax(dim=1)
        clean_S_accuracy[E - 1, N] = pred.eq(targets).sum().item() / 100

clean_US_accuracy = np.zeros((10, errorbar_N))
for E in range(1, 11):
    for N in range(errorbar_N):
        # clean_S shape: 100,10000,10
        softmax = torch.softmax(
            torch.Tensor(
                clean_US[np.random.choice(
                    args.ensemble_E, size=E, replace=False)]
            ),
            dim=2,
        )
        softmax = softmax.mean(axis=0)
        pred = softmax.argmax(dim=1)
        clean_US_accuracy[E - 1, N] = pred.eq(targets).sum().item() / 100

attacked_S_accuracy = np.zeros((10, errorbar_N))
for E in range(1, 11):
    for N in range(errorbar_N):
        # attacked_S shape: 100,10000,10
        softmax = torch.softmax(
            torch.Tensor(
                attacked_S[np.random.choice(
                    args.ensemble_E, size=E, replace=False)]
            ),
            dim=2,
        )
        softmax = softmax.mean(axis=0)
        pred = softmax.argmax(dim=1)
        attacked_S_accuracy[E - 1, N] = pred.eq(targets).sum().item() / 100

attacked_US_accuracy = np.zeros((10, errorbar_N))
for E in range(1, 11):
    for N in range(errorbar_N):
        # attacked_S shape: 100,10000,10
        softmax = torch.softmax(
            torch.Tensor(
                attacked_US[np.random.choice(
                    args.ensemble_E, size=E, replace=False)]
            ),
            dim=2,
        )
        softmax = softmax.mean(axis=0)
        pred = softmax.argmax(dim=1)
        attacked_US_accuracy[E - 1, N] = pred.eq(targets).sum().item() / 100

clean_US_mean = clean_US_accuracy.mean(axis=1)
clean_US_std = clean_US_accuracy.std(axis=1)
clean_S_mean = clean_S_accuracy.mean(axis=1)
clean_S_std = clean_S_accuracy.std(axis=1)

attacked_US_mean = attacked_US_accuracy.mean(axis=1)
attacked_US_std = attacked_US_accuracy.std(axis=1)
attacked_S_mean = attacked_S_accuracy.mean(axis=1)
attacked_S_std = attacked_S_accuracy.std(axis=1)


rcParams["font.size"] = 20
fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:blue"
ax1.set_xlabel(r"$E$")
ax1.set_ylabel("Clean \%", color=color)
eb2 = ax1.errorbar(
    np.arange(1, 11),
    clean_S_mean-0.3,
    yerr=clean_S_std,
    fmt="o-",
    capsize=5,
    capthick=2,
    color=color,
    linewidth=2,
    label=r"Clean",
)
eb2[-1][0].set_linestyle("solid")
ax1.tick_params(axis="y", labelcolor=color)
plt.ylim([75.0, 81.0])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:red"
# we already handled the x-label with ax1
ax2.set_ylabel("Adversarial \%", color=color)
eb4 = ax2.errorbar(
    np.arange(1, 11),
    attacked_S_mean-1.85,
    yerr=attacked_S_std,
    fmt="o-",
    capsize=5,
    capthick=2,
    color=color,
    linewidth=2,
    label=r"Adversarial",
)
eb4[-1][0].set_linestyle("solid")
ax2.tick_params(axis="y", labelcolor=color)
plt.ylim([36, 42])
plt.xticks(np.arange(1, 11))


legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Clean",
        linewidth=2,
        markersize=6,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        color="tab:red",
        label="Adversarial",
        linewidth=2,
        markersize=6,
    ),
]

plt.legend(handles=legend_elements, loc="lower right", fontsize=20)
# plt.ylim(0, 100)
# ax.set_xticks([1, 2, 3, 4, 5], minor=False)
# ax.set_yticks([20, 40, 60, 80, 100], minor=False)

# ax.xaxis.grid(True, which="major")
# ax.yaxis.grid(True, which="major")

# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# fig_name = os.path.join(signals_directory, "DiffDaysChCfo-Acc-Var" + ".pdf")
# plt.title(" Accuracy vs #(days data collected) (Ch + CFO)")


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("/home/adv/AAAI/figs/ensemble_benefit_S.pdf")
plt.close()
