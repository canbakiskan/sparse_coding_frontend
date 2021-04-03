from tqdm import tqdm
import os
from os import path

from .utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from .utils.get_modules import (
    get_classifier,
    get_autoencoder,
)

import numpy as np
import torch
import torch.nn.functional as F
from .models.combined import Combined, Combined_inner_BPDA_identity, Combined_inner_BPDA_gaussianblur
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
    PGD_smooth,
)
from .utils.read_datasets import(
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

    if args.attack_box_type == "white":

        if args.attack_whitebox_type == "SW":
            adversarial_args["attack_args"]["net"] = model.module_outer
            adversarial_args["attack_args"]["attack_params"]["EOT_size"] = 1

        else:
            adversarial_args["attack_args"]["net"] = model

    elif args.attack_box_type == "other":
        if args.attack_otherbox_type == "transfer":
            # it shouldn't enter this clause
            raise Exception(
                "Something went wrong, transfer attack shouldn't be using generate_attack")

        elif args.attack_otherbox_type == "boundary":
            import foolbox as fb

            fmodel = fb.PyTorchModel(model, bounds=(0, 1))

            attack = fb.attacks.BoundaryAttack()
            l2_epsilons = [adversarial_args["attack_args"]
                           ["attack_params"]["eps"]]
            raw_advs, clipped_advs, success = attack(
                fmodel, data, target, epsilons=l2_epsilons,
                starting_points=adversarial_args["attack_args"]["starting_points"])
            return raw_advs[0] - data

        elif args.attack_otherbox_type == "hopskip":
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
            raise Exception("Choose white or other for attack_box_type.")

    adversarial_args["attack_args"]["x"] = data
    adversarial_args["attack_args"]["y_true"] = target
    perturbation = adversarial_args["attack"](
        **adversarial_args["attack_args"])

    return perturbation


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

    if not os.path.exists(os.dirname(attack_log_namer(args))):
        os.makedirs(os.dirname(attack_log_namer(args)))

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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    read_from_file = (args.attack_box_type ==
                      "other" and args.attack_otherbox_type == "transfer") or not recompute

    if read_from_file:
        args.save_attack = False

    classifier = get_classifier(args)

    if args.no_autoencoder:
        model = classifier

    else:
        autoencoder = get_autoencoder(args)

        if args.attack_box_type == "white" and args.attack_whitebox_type == "W-AIGA":
            model = Combined_inner_BPDA_identity(autoencoder, classifier)
        elif args.attack_box_type == "white" and args.attack_whitebox_type == "W-AGGA":
            model = Combined_inner_BPDA_gaussianblur(
                autoencoder, classifier, args)
        else:
            if (
                args.attack_box_type == "white"
                and args.attack_whitebox_type == "top_T_dropout_identity"
            ):
                autoencoder.set_BPDA_type("identity")

            elif (
                args.attack_box_type == "white"
                and args.attack_whitebox_type == "top_T_top_U"
            ):
                autoencoder.set_BPDA_type("top_U")

            elif (
                args.attack_box_type == "white"
                and args.attack_whitebox_type == "W-NFGA"
            ):
                autoencoder.set_BPDA_type("maxpool_like")

            model = Combined(autoencoder, classifier)

        model = model.to(device)
        model.eval()

    if (
        "dropout" in args.autoencoder_arch
        and not args.no_autoencoder
        and args.ensemble_E > 1
    ):
        from .models.ensemble import Ensemble_post_softmax

        ensemble_model = Ensemble_post_softmax(model, args.ensemble_E)

    else:
        ensemble_model = model

    ensemble_model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # this is just for the adversarial test below
    if args.dataset == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    if not args.attack_skip_clean:
        test_loss, test_acc = adversarial_test(ensemble_model, test_loader)
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
        "norm": args.attack_norm,
        "eps": args.attack_epsilon,
        "alpha": args.attack_alpha,
        "step_size": args.attack_step_size,
        "num_steps": args.attack_num_steps,
        "random_start": (args.adv_training_rand and args.adv_training_num_restarts > 1),
        "num_restarts": args.attack_num_restarts,
        "EOT_size": args.attack_EOT_size,
    }

    data_params = {"x_min": 0.0, "x_max": 1.0}

    if "CWlinf" in args.attack_method:
        attack_method = args.attack_method.replace("CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        attack_method = args.attack_method
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=attacks[attack_method],
        attack_args=dict(
            net=model,
            data_params=data_params,
            attack_params=attack_params,
            progress_bar=args.attack_progress_bar,
            verbose=True,
            loss_function=loss_function,
        ),
    )

    test_loss = 0
    correct = 0

    if args.save_attack:
        attacked_images = torch.zeros(len(
            test_loader.dataset.targets), args.image_shape[2], args.image_shape[0], args.image_shape[1])

    attack_output = torch.zeros(
        len(test_loader.dataset.targets), args.num_classes)

    if read_from_file:
        if args.dataset == "CIFAR10":
            test_loader = cifar10_from_file(args)
        elif args.dataset == "Tiny-ImageNet":
            test_loader = tiny_imagenet_from_file(args)
        elif args.dataset == "Imagenette":
            test_loader = imagenette_from_file(args)
        else:
            raise NotImplementedError
    else:
        if args.dataset == "CIFAR10":
            _, test_loader = cifar10(args)
        elif args.dataset == "Tiny-ImageNet":
            _, test_loader = tiny_imagenet(args)
        elif args.dataset == "Imagenette":
            _, test_loader = imagenette(args)
        else:
            raise NotImplementedError

    loaders = test_loader

    start = time.time()

    if args.attack_box_type == "other" and (args.otherbox_type == "boundary" or args.otherbox_type == "hopskip"):
        img_distances_idx = np.load(
            f'./data/image_distances/{args.dataset}/closest_img_indices.npy')
        preds = torch.zeros(10000, dtype=torch.int)

        for batch_idx, items in enumerate(loaders):

            data, target = items
            data = data.to(device)
            target = target.to(device)
            preds[batch_idx
                  * args.test_batch_size: (batch_idx + 1)
                  * args.test_batch_size] = ensemble_model(data).argmax(dim=1).detach().cpu()

        closest_adv_idx = torch.zeros(10000, dtype=int)
        for i in range(10000):
            j = 1
            while preds[img_distances_idx[i, j]] == test_loader.dataset.targets[i]:
                j += 1
            closest_adv_idx[i] = img_distances_idx[i, j]

        closest_images = torch.tensor(
            test_loader.dataset.data[closest_adv_idx]/255, dtype=torch.float32).permute(0, 3, 1, 2)

    correct_sofar = 0
    for batch_idx, items in enumerate(
        pbar := tqdm(loaders, desc="Attack progress", leave=False)
    ):
        if args.defense_nbimgs < (batch_idx + 1) * args.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        if args.attack_box_type == "other" and (args.otherbox_type == "boundary" or args.otherbox_type == "hopskip"):
            adversarial_args['attack_args']['starting_points'] = closest_images[batch_idx
                                                                                * args.test_batch_size: (batch_idx + 1)
                                                                                * args.test_batch_size].to(device)

        if not read_from_file:
            attack_batch = generate_attack(
                args, model, data, target, adversarial_args)
            data += attack_batch
            data = data.clamp(0.0, 1.0)
            if args.save_attack:
                attacked_images[
                    batch_idx
                    * args.test_batch_size: (batch_idx + 1)
                    * args.test_batch_size,
                ] = data.detach().cpu()

        with torch.no_grad():
            out = ensemble_model(data).detach()
            attack_output[
                batch_idx
                * args.test_batch_size: (batch_idx + 1)
                * args.test_batch_size,
            ] = out.cpu()

            batch_pred = out.argmax(dim=1, keepdim=True)

            batch_correct = batch_pred.eq(
                target.view_as(batch_pred)).sum().item()
            correct_sofar += batch_correct
            accuracy_sofar = correct_sofar / \
                ((batch_idx+1) * args.test_batch_size)

        pbar.set_postfix(
            Adv_ac=f"{accuracy_sofar:.4f}", refresh=True,
        )

    end = time.time()
    logger.info(f"Attack computation time: {(end-start):.2f} seconds")

    if args.dataset == "CIFAR10":
        _, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        _, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        _, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    target = torch.tensor(test_loader.dataset.targets)[: args.defense_nbimgs]
    pred_attack = attack_output.argmax(dim=1, keepdim=True)[
        : args.defense_nbimgs]

    correct_attack = pred_attack.eq(target.view_as(pred_attack)).sum().item()
    accuracy_attack = correct_attack / args.defense_nbimgs

    logger.info(f"Attack accuracy: {(100*accuracy_attack):.2f}%")

    if args.save_attack:
        attack_filepath = attack_file_namer(args)

        if not os.path.exists(os.dirname(attack_file_namer(args))):
            os.makedirs(os.dirname(attack_file_namer(args)))

        np.save(attack_filepath, attacked_images.detach().cpu().numpy())

        logger.info(f"Saved to {attack_filepath}")


if __name__ == "__main__":
    main()
