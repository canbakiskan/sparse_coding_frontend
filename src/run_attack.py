from tqdm import tqdm
import os
from os import path

from .utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from .utils.get_modules import (
    get_classifier,
    get_frontend,
)

import numpy as np
import torch
from .models.combined import Combined
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
    PGD_smooth,
)
from .utils.read_datasets import read_dataset, read_test_dataset_from_file
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

    elif args.adv_testing.box_type == "black":
        if args.adv_testing.otherbox_method == "transfer":
            # it shouldn't enter this clause
            raise Exception(
                "Something went wrong, transfer attack shouldn't be using generate_attack")

        elif args.adv_testing.otherbox_method == "boundary":
            import foolbox as fb

            fmodel = fb.PyTorchModel(model, bounds=(0, 1))

            attack = fb.attacks.BoundaryAttack()
            l2_epsilons = [adversarial_args["attack_args"]
                           ["attack_params"]["eps"]]
            raw_advs, clipped_advs, success = attack(
                fmodel, data, target, epsilons=l2_epsilons,
                starting_points=adversarial_args["attack_args"]["starting_points"])
            return raw_advs[0] - data

        elif args.adv_testing.otherbox_method == "hopskip":
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
        test_loader = read_test_dataset_from_file(args)
    else:
        _, test_loader = read_dataset(args)

    loaders = test_loader

    start = time.time()

    if (args.adv_testing.method == "boundary" or args.adv_testing.method == "hopskip") and not read_from_file:
        img_distances_idx = np.load(
            f'./data/image_distances/{args.dataset.name}/closest_img_indices.npy')
        preds = torch.zeros(10000, dtype=torch.int)

        for batch_idx, items in enumerate(loaders):

            data, target = items
            data = data.to(device)
            target = target.to(device)
            preds[batch_idx
                  * args.neural_net.test_batch_size: (batch_idx + 1)
                  * args.neural_net.test_batch_size] = model(data).argmax(dim=1).detach().cpu()

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
        if args.adv_testing.nb_imgs > 0 and args.adv_testing.nb_imgs < (batch_idx + 1) * args.neural_net.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        if not read_from_file:
            if (args.adv_testing.method == "boundary" or args.adv_testing.method == "hopskip"):
                adversarial_args['attack_args']['starting_points'] = closest_images[batch_idx
                                                                                    * args.neural_net.test_batch_size: (batch_idx + 1)
                                                                                    * args.neural_net.test_batch_size].to(device)

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
            out = model(data).detach()
            attack_output[
                batch_idx
                * args.neural_net.test_batch_size: (batch_idx + 1)
                * args.neural_net.test_batch_size,
            ] = out.cpu()

            batch_pred = out.argmax(dim=1, keepdim=True)

            batch_correct = batch_pred.eq(
                target.view_as(batch_pred)).sum().item()
            correct_sofar += batch_correct
            accuracy_sofar = correct_sofar / \
                ((batch_idx+1) * args.neural_net.test_batch_size)

        pbar.set_postfix(
            Adv_ac=f"{accuracy_sofar:.4f}", refresh=True,
        )

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

        if not os.path.exists(os.path.dirname(attack_file_namer(args))):
            os.makedirs(os.path.dirname(attack_file_namer(args)))

        np.save(attack_filepath, attacked_images.detach().cpu().numpy())

        logger.info(f"Saved to {attack_filepath}")


if __name__ == "__main__":
    main()
