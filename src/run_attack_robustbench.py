from tqdm import tqdm
from deepillusion.torchdefenses import adversarial_test
from .utils.read_datasets import read_dataset

from .models.combined import Combined
import torch
from .utils.get_modules import (
    load_classifier,
    load_frontend,
)
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD
from torchattacks import PGDL2

from .utils.device import determine_device

from .parameters import get_arguments
from robustbench.data import load_cifar10

from robustbench.utils import load_model
from autoattack import AutoAttack
import os


def main():

    args = get_arguments()

    device = determine_device(args)

    classifier = load_classifier(args)

    if args.neural_net.no_frontend:
        model = classifier

    else:
        frontend = load_frontend(args)
        model = Combined(frontend, classifier)

    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # if not args.adv_testing.skip_clean:
    #     _, test_loader = read_dataset(args)
    #     test_loss, test_acc = adversarial_test(model, test_loader)
    #     print(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    fmodel = PyTorchModel(model, bounds=(0, 1))

    x_test, y_test = load_cifar10(n_examples=10000, data_dir=os.path.join(
        args.directory, 'data', 'original_datasets'))

    # model below:
    # Linf eps:8/255 acc:
    # L2 eps:0.6 acc:56.56

    # model = load_model(model_name='Carmon2019Unlabeled',
    #                    dataset='cifar10', threat_model='Linf').to('cuda')

    # ours
    # Linf eps:8/255 acc:35.16
    # L2 eps:0.6 acc:56.48
    # L2 eps:0.5 acc:

    adversary = AutoAttack(model, norm='L2', eps=0.5,
                           version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1

    x_adv = adversary.run_standard_evaluation(x_test, y_test)


if __name__ == "__main__":
    main()
