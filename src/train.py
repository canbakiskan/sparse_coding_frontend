import time
import os
from tqdm import tqdm
import numpy as np

import logging

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .models.resnet import ResNet, ResNetWide
from .models.efficientnet import EfficientNet
from .models.preact_resnet import PreActResNet101
from .models.ablation.dropout_resnet import dropout_ResNet
from .models.ablation.resnet_after_encoder import ResNet_after_encoder

from .train_test_functions import train, test

from .parameters import get_arguments
from .utils.read_datasets import read_dataset
from .utils.get_optimizer_scheduler import get_optimizer_scheduler

from .utils.namers import (
    frontend_ckpt_namer,
    frontend_log_namer,
    classifier_ckpt_namer,
    classifier_log_namer,
)

from .models.combined import Combined
from .utils.get_modules import get_frontend
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
)
from deepillusion.torchdefenses import adversarial_epoch, adversarial_test
from .models.frontend import frontend_class
import sys

logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

    args = get_arguments()
    if not os.path.exists(os.dirname(classifier_log_namer(args))):
        os.makedirs(os.dirname(classifier_log_namer(args)))

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(classifier_log_namer(args)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(args)
    logger.info("\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x_min = 0.0
    x_max = 1.0
    # L = round((32 - args.defense.patch_size) / args.defense.stride + 1)

    train_loader, test_loader = read_dataset(args)

    if args.neural_net.classifier_arch == "resnet":
        classifier = ResNet(num_outputs=args.dataset.nb_classes).to(device)
    elif args.neural_net.classifier_arch == "resnetwide":
        classifier = ResNetWide(num_outputs=args.dataset.nb_classes).to(device)
    elif args.neural_net.classifier_arch == "efficientnet":
        classifier = EfficientNet.from_name(
            "efficientnet-b0", num_classes=args.dataset.nb_classes, dropout_rate=0.2
        ).to(device)
    elif args.neural_net.classifier_arch == "preact_resnet":
        classifier = PreActResNet101(
            num_classes=args.dataset.nb_classes).to(device)
    elif args.neural_net.classifier_arch == "dropout_resnet":
        classifier = dropout_ResNet(
            dropout_p=args.defense.dropout_p,
            nb_filters=args.dictionary.nb_atoms,
            num_outputs=args.dataset.nb_classes,
        ).to(device)
    elif args.neural_net.classifier_arch == "resnet_after_encoder":
        classifier = ResNet_after_encoder(
            nb_filters=args.dictionary.nb_atoms, num_outputs=args.dataset.nb_classes).to(device)
    else:
        raise NotImplementedError

    if not args.neural_net.no_frontend:

        if args.frontend_train_supervised:

            frontend = frontend_class(
                args).to(device)

            for p in frontend.parameters():
                p.requires_grad = True

            if args.ablation.no_dictionary:
                frontend.dictionary_update_on()
            else:
                frontend.dictionary_update_off()

        else:
            frontend = get_frontend(args)

            for p in frontend.parameters():
                p.requires_grad = False

        model = Combined(frontend, classifier)

    else:
        model = classifier

    model.train()

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    # Which optimizer to be used for training

    optimizer, scheduler = get_optimizer_scheduler(
        args, model, len(train_loader))

    if args.adv_training.method:

        attacks = dict(
            PGD=PGD,
            PGD_EOT=PGD_EOT,
            PGD_EOT_normalized=PGD_EOT_normalized,
            PGD_EOT_sign=PGD_EOT_sign,
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

        logger.info(args.adv_training.method + " training")
        logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

        for epoch in tqdm(range(1, args.neural_net.epochs + 1)):
            start_time = time.time()

            train_args = dict(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                adversarial_args=adversarial_args,
            )
            train_loss, train_acc = adversarial_epoch(**train_args)

            test_args = dict(model=model, test_loader=test_loader)
            test_loss, test_acc = adversarial_test(**test_args)

            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(
                f"{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}"
            )
            logger.info(
                f"Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    else:

        logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

        logger.info("Standard training")
        for epoch in tqdm(range(1, args.neural_net.epochs + 1)):
            start_time = time.time()

            train_loss, train_acc = train(
                model, train_loader, optimizer, scheduler)
            test_loss, test_acc = test(model, test_loader)

            end_time = time.time()
            # lr = scheduler.get_lr()[0]
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}"
            )
            logger.info(
                f"Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    # Save model parameters
    if args.neural_net.save_checkpoint:
        if not os.path.exists(args.directory + "checkpoints/classifiers/"):
            os.makedirs(args.directory + "checkpoints/classifiers/")

        classifier_filepath = classifier_ckpt_namer(args)
        torch.save(classifier.state_dict(), classifier_filepath)

        logger.info(f"Saved to {classifier_filepath}")

        if not args.neural_net.no_frontend:
            if not os.path.exists(os.dirname(frontend_ckpt_namer(args))):
                os.makedirs(os.dirname(frontend_ckpt_namer(args)))

            frontend_filepath = frontend_ckpt_namer(args)
            if args.frontend_train_supervised:
                torch.save(frontend.state_dict(), frontend_filepath)

            logger.info(f"Saved to {frontend_filepath}")


if __name__ == "__main__":
    main()
