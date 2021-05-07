import time
import os
from tqdm import tqdm
import numpy as np

import logging

import torch
import torch.backends.cudnn as cudnn


from .train_test_functions import train, test

from .parameters import get_arguments
from .utils.read_datasets import read_dataset
from .utils.get_optimizer_scheduler import get_optimizer_scheduler
from .utils.device import determine_device
from .utils.namers import (
    frontend_ckpt_namer,
    classifier_ckpt_namer,
    classifier_log_namer,
)

from .models.combined import Combined
from .utils.get_modules import create_frontend, create_classifier
from .utils.logger import logger_setup
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
    PGD_smooth
)
from deepillusion.torchdefenses import adversarial_epoch, adversarial_test
import sys

logger = logging.getLogger(__name__)


def check_recompute():
    print(
        "Checkpoint already exists. Do you want to retrain? [y/(n)]", end=" ")
    response = input()
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    if response != "y":
        exit()


def apply_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def construct_adv_args(args, model):

    attacks = dict(
        PGD=PGD,
        PGD_EOT=PGD_EOT,
        PGD_EOT_normalized=PGD_EOT_normalized,
        PGD_EOT_sign=PGD_EOT_sign,
        PGD_smooth=PGD_smooth,
        FGSM=FGSM,
        RFGSM=RFGSM,
    )

    attack_params = {
        "norm": args.adv_training.norm,
        "eps": args.adv_training.budget,
        "alpha": args.adv_training.rfgsm_alpha,
        "step_size": args.adv_training.step_size,
        "num_steps": args.adv_training.nb_steps,
        "random_start": (
            args.adv_training.rand and args.adv_training.nb_restarts > 1
        ),
        "num_restarts": args.adv_training.nb_restarts,
        "EOT_size": args.adv_training.EOT_size,
    }

    data_params = {"x_min": 0.0, "x_max": 1.0}

    if "CWlinf" in args.adv_training.method:
        adv_training_attack = args.adv_training.method.replace(
            "CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        adv_training_attack = args.adv_training.method
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=attacks[adv_training_attack],
        attack_args=dict(
            net=model, data_params=data_params, attack_params=attack_params
        ),
        loss_function=loss_function,
    )

    return adversarial_args


def save_frontend(args, frontend):
    if not os.path.exists(os.path.dirname(frontend_ckpt_namer(args))):
        os.makedirs(os.path.dirname(frontend_ckpt_namer(args)))

    frontend_filepath = frontend_ckpt_namer(args)
    if args.frontend_train_supervised:
        torch.save(frontend.state_dict(), frontend_filepath)

    logger.info(f"Saved to {frontend_filepath}")


def save_classifier(args, classifier):
    if not os.path.exists(os.path.join(args.directory, 'checkpoints', 'classifiers')):
        os.makedirs(os.path.join(args.directory,
                                 'checkpoints', 'classifiers'))

    classifier_filepath = classifier_ckpt_namer(args)
    torch.save(classifier.state_dict(), classifier_filepath)

    logger.info(f"Saved to {classifier_filepath}")


def save_model(args, frontend, classifier):

    save_classifier(args, classifier)

    if not args.neural_net.no_frontend:
        save_frontend(args, frontend)


def main():

    args = get_arguments()

    if os.path.exists(classifier_ckpt_namer(args)):
        check_recompute()

    logger_setup(classifier_log_namer(args))
    logger.info(args)
    logger.info("\n")

    device = determine_device(args)
    apply_seed(args.seed)

    train_loader, test_loader = read_dataset(args)

    classifier = create_classifier(args)

    if not args.neural_net.no_frontend:

        frontend = create_frontend(args)

        for p in frontend.parameters():
            p.requires_grad = True

        if args.ablation.no_dictionary:
            frontend.dictionary_update_on()
        else:
            frontend.dictionary_update_off()

        model = Combined(frontend, classifier)

    else:
        model = classifier

    model.train()

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    optimizer, scheduler = get_optimizer_scheduler(
        args, model, len(train_loader))

    if args.adv_training.method:
        adversarial_args = construct_adv_args(args, model)

        train_args = dict(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            adversarial_args=adversarial_args,
        )

        test_args = dict(model=model, test_loader=test_loader)

        logger.info(args.adv_training.method + " training")
    else:
        logger.info("Standard training")

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

    for epoch in tqdm(range(1, args.neural_net.epochs + 1)):
        start_time = time.time()

        if args.adv_training.method:
            train_loss, train_acc = adversarial_epoch(**train_args)
            test_loss, test_acc = adversarial_test(**test_args)

        else:
            train_loss, train_acc = train(
                model, train_loader, optimizer, scheduler)
            test_loss, test_acc = test(model, test_loader)

        end_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info(
            f"{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}"
        )
        logger.info(
            f"Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    if args.neural_net.save_checkpoint:
        save_model(args, frontend, classifier)


if __name__ == "__main__":
    main()
