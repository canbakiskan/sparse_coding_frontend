#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
from tqdm import tqdm
import time
import sys
import logging
from deepillusion.torchdefenses import adversarial_test
from .utils.read_datasets import read_dataset

from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
    PGD_smooth,
)
from .models.combined import Combined
import torch
import numpy as np
from .utils.get_modules import (
    get_classifier,
    get_frontend,
)
from .utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from os import path
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD
import os

logger = logging.getLogger(__name__)


def main():

    from .parameters import get_arguments

    args = get_arguments()

    recompute = True
    if path.exists(attack_file_namer(args)):
        print(
            "Attack already exists. Do you want to recompute? [y/(n)]", end=" ")
        response = input()
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        if response != "y":
            recompute = False

    if not os.path.exists(os.path.dirname(attack_log_namer(args))):
        os.makedirs(os.path.dirname(attack_log_namer(args)))

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(attack_log_namer(args)),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(args)
    logger.info("\n")

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    read_from_file = args.adv_testing.method == "transfer" or not recompute

    if read_from_file:
        args.adv_testing.save = False

    classifier = get_classifier(args)

    if args.neural_net.no_frontend:
        model = classifier

    else:
        frontend = get_frontend(args)
        model = Combined(frontend, classifier)

    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # this is just for the adversarial test below
    _, test_loader = read_dataset(args)

    if not args.adv_testing.skip_clean:
        test_loss, test_acc = adversarial_test(model, test_loader)
        logger.info(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    fmodel = PyTorchModel(model, bounds=(0, 1))

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following

    clean_total = 0
    adv_total = 0

    for batch_idx, items in enumerate(
        tqdm(test_loader, desc="Attack progress", leave=False)
    ):
        if args.adv_testing.nb_imgs > 0 and args.adv_testing.nb_imgs < (batch_idx + 1) * args.neural_net.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        clean_acc = accuracy(fmodel, data, target)
        clean_total += int(clean_acc*args.neural_net.test_batch_size)

        # apply the attack
        # attack = LinfPGD(rel_stepsize=1/8., steps=40)
        # epsilons = [
        #     8./255.
        # ]

        # attack = L2PGD(rel_stepsize=1/40., steps=100)
        # epsilons = [
        #     255/255.
        # ]

        # raw_advs, clipped_advs, success = attack(
        #     fmodel, data, target, epsilons=epsilons)

        from torchattacks import PGDL2

        attack = PGDL2(model, eps=0.6, alpha=0.6/40,
                       steps=100, random_start=False, eps_for_division=1e-10)

        adv_images = attack(data, target)

        # breakpoint()
        attack_out = model(adv_images)
        pred_attack = attack_out.argmax(dim=1, keepdim=True)

        robust_accuracy = pred_attack.eq(
            target.view_as(pred_attack)).sum().item()

        # calculate and report the robust accuracy (the accuracy of the model when
        # it is attacked)
        # robust_accuracy = 1 - success.float().mean(axis=-1)
        adv_total += int(robust_accuracy)

        print(
            f"current clean accuracy:  {clean_total / ((batch_idx+1)*args.neural_net.test_batch_size) * 100:.2f} %")
        print(
            f"current adv accuracy:  {adv_total / ((batch_idx+1)*args.neural_net.test_batch_size) * 100:.2f} %")

        # print("robust accuracy for perturbations with")
        # for eps, acc in zip(epsilons, robust_accuracy):
        #     print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        # break

    print(f"clean accuracy:  {clean_total / 10000 * 100:.2f} %")
    print(f"adv accuracy:  {adv_total / 10000 * 100:.2f} %")

    exit()
    attacks = dict(
        PGD=PGD,
        PGD_EOT=PGD_EOT,
        PGD_smooth=PGD_smooth,
        PGD_EOT_normalized=PGD_EOT_normalized,
        PGD_EOT_sign=PGD_EOT_sign,
        FGSM=FGSM,
        RFGSM=RFGSM,
    )

    attack_params = {
        "norm": args.adv_testing.norm,
        "eps": args.adv_testing.budget,
        "alpha": args.adv_testing.rfgsm_alpha,
        "step_size": args.adv_testing.step_size,
        "num_steps": args.adv_testing.nb_steps,
        "random_start": (args.adv_testing.rand and args.adv_training.nb_restarts > 1),
        "num_restarts": args.adv_testing.nb_restarts,
        "EOT_size": args.adv_testing.EOT_size,
    }

    data_params = {"x_min": 0.0, "x_max": 1.0}

    if "CWlinf" in args.adv_testing.method:
        attack_method = args.adv_testing.method.replace("CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        attack_method = args.adv_testing.method
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=attacks[attack_method],
        attack_args=dict(
            net=model,
            data_params=data_params,
            attack_params=attack_params,
            progress_bar=args.adv_testing.progress_bar,
            verbose=True,
            loss_function=loss_function,
        ),
    )

    test_loss = 0
    correct = 0

    if args.adv_testing.save:
        attacked_images = torch.zeros(len(
            test_loader.dataset.targets), args.dataset.img_shape[2], args.dataset.img_shape[0], args.dataset.img_shape[1])

    attack_output = torch.zeros(
        len(test_loader.dataset.targets), args.dataset.nb_classes)

    start = time.time()
    for batch_idx, items in enumerate(
        tqdm(loaders, desc="Attack progress", leave=False)
    ):
        if args.adv_testing.nb_imgs > 0 and args.adv_testing.nb_imgs < (batch_idx + 1) * args.neural_net.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        if not read_from_file:
            attack_batch = generate_attack(
                args, model, data, target, adversarial_args)
            data += attack_batch
            data = data.clamp(0.0, 1.0)
            if args.adv_testing.save:
                attacked_images[
                    batch_idx
                    * args.neural_net.test_batch_size: (batch_idx + 1)
                    * args.neural_net.test_batch_size,
                ] = data.detach().cpu()

        with torch.no_grad():
            attack_output[
                batch_idx
                * args.neural_net.test_batch_size: (batch_idx + 1)
                * args.neural_net.test_batch_size,
            ] = (model(data).detach().cpu())

    end = time.time()
    logger.info(f"Attack computation time: {(end-start):.2f} seconds")

    _, test_loader = read_dataset(args)

    target = torch.tensor(test_loader.dataset.targets)[
        : args.adv_testing.nb_imgs]
    pred_attack = attack_output.argmax(dim=1, keepdim=True)[
        : args.adv_testing.nb_imgs]

    accuracy_attack = pred_attack.eq(
        target.view_as(pred_attack)).float().mean().item()

    logger.info(f"Attack accuracy: {(100*accuracy_attack):.2f}%")

    if args.adv_testing.save:
        attack_filepath = attack_file_namer(args)
        np.save(attack_filepath, attacked_images.detach().cpu().numpy())

        logger.info(f"Saved to {attack_filepath}")


if __name__ == "__main__":
    main()
