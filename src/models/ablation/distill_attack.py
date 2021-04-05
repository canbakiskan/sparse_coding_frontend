from tqdm import tqdm
import os
from os import path

from ...utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from ...utils.get_modules import (
    get_classifier,
    get_frontend,
)

import numpy as np
import torch
import torch.nn.functional as F
from ..combined import Combined,
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
    PGD_smooth,
)
from ...utils.read_datasets import(
    cifar10,
    cifar10_from_file,
    tiny_imagenet,
    tiny_imagenet_from_file,
    imagenette,
    imagenette_from_file
)
from deepillusion.torchdefenses import adversarial_test

import logging
import sys
import time
logger = logging.getLogger(__name__)


def generate_attack(args, model, data, target, adversarial_args):

    if args.adv_testing.box_type == "white":
        adversarial_args["attack_args"]["net"] = model

        adversarial_args["attack_args"]["x"] = data
        adversarial_args["attack_args"]["y_true"] = target
        perturbation = adversarial_args["attack"](
            **adversarial_args["attack_args"])

        return perturbation

    elif args.adv_testing.box_type == "other":
        if args.adv_testing.otherbox_type == "transfer":
            # it shouldn't enter this clause
            raise Exception(
                "Something went wrong, transfer attack shouldn't be using generate_attack")

        elif args.adv_testing.otherbox_type == "boundary":
            import foolbox as fb

            fmodel = fb.PyTorchModel(model, bounds=(0, 1))

            attack = fb.attacks.BoundaryAttack()
            l2_epsilons = [adversarial_args["attack_args"]
                           ["attack_params"]["eps"]]
            raw_advs, clipped_advs, success = attack(
                fmodel, data, target, epsilons=l2_epsilons,
                starting_points=adversarial_args["attack_args"]["starting_points"])
            return raw_advs[0] - data

        elif args.adv_testing.otherbox_type == "hopskip":
            import foolbox as fb

            fmodel = fb.PyTorchModel(model, bounds=(0, 1))

            attack = fb.attacks.HopSkipJump()
            l2_epsilons = [adversarial_args["attack_args"]
                           ["attack_params"]["eps"]]
            raw_advs, clipped_advs, success = attack(
                fmodel, data, target, epsilons=l2_epsilons,
                starting_points=adversarial_args["attack_args"]["starting_points"])
            return raw_advs[0] - data

        else:
            raise ValueError("Otherbox attack type not supported.")

    else:
        raise ValueError("Attack box type not supported.")


def main():

    from ...parameters import get_arguments

    args = get_arguments()
    args.ablation.distill = True

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

    read_from_file = (args.adv_testing.box_type ==
                      "other" and args.adv_testing.otherbox_type == "transfer") or not recompute

    if read_from_file:
        args.adv_testing.save = False

    classifier = get_classifier(args)

    model = classifier

    for p in model.parameters():
        p.requires_grad = False

    # this is just for the adversarial test below
    if args.dataset.name == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset.name == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset.name == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    if not args.adv_testing.skip_clean:
        test_loss, test_acc = adversarial_test(model, test_loader)
        logger.info(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

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

    if read_from_file:
        if args.dataset.name == "CIFAR10":
            test_loader = cifar10_from_file(args)
        elif args.dataset.name == "Tiny-ImageNet":
            test_loader = tiny_imagenet_from_file(args)
        elif args.dataset.name == "Imagenette":
            test_loader = imagenette_from_file(args)
        else:
            raise NotImplementedError
    else:
        if args.dataset.name == "CIFAR10":
            _, test_loader = cifar10(args)
        elif args.dataset.name == "Tiny-ImageNet":
            _, test_loader = tiny_imagenet(args)
        elif args.dataset.name == "Imagenette":
            _, test_loader = imagenette(args)
        else:
            raise NotImplementedError

    loaders = test_loader

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

    if args.dataset.name == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset.name == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset.name == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    target = torch.tensor(test_loader.dataset.targets)[
        : args.adv_testing.nb_imgs]
    pred_attack = attack_output.argmax(dim=1, keepdim=True)[
        : args.adv_testing.nb_imgs]

    accuracy_attack = pred_attack.eq(target.view_as(pred_attack)).mean().item()

    logger.info(f"Attack accuracy: {(100*accuracy_attack):.2f}%")

    if args.adv_testing.save:
        attack_filepath = attack_file_namer(args)

        if not os.path.exists(os.path.dirname(attack_file_namer(args))):
            os.makedirs(os.path.dirname(attack_file_namer(args)))

        np.save(attack_filepath, attacked_images.detach().cpu().numpy())

        logger.info(f"Saved to {attack_filepath}")


if __name__ == "__main__":
    main()
