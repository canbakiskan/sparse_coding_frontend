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

from .train_test_functions import (
    train,
    test,
)

from .parameters import get_arguments
from .utils.read_datasets import cifar10, tiny_imagenet, imagenette

from .utils.namers import (
    autoencoder_ckpt_namer,
    autoencoder_log_namer,
    classifier_ckpt_namer,
    classifier_log_namer,
)

from .models.combined import Combined
from .utils.get_modules import get_autoencoder
from ..adversarial_framework.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
)
from ..adversarial_framework.torchdefenses import (
    adversarial_epoch,
    adversarial_test,
)
from .models.autoencoders import *
import sys

logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

    args = get_arguments()
    if not os.path.exists(args.directory + "logs"):
        os.mkdir(args.directory + "logs")

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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x_min = 0.0
    x_max = 1.0
    # L = round((32 - args.defense_patchsize) / args.defense_stride + 1)

    if args.dataset == "CIFAR10":
        train_loader, test_loader = cifar10(args)
    elif args.dataset == "Tiny-ImageNet":
        train_loader, test_loader = tiny_imagenet(args)
    elif args.dataset == "Imagenette":
        train_loader, test_loader = imagenette(args)
    else:
        raise NotImplementedError

    if args.classifier_arch == "resnet":
        classifier = ResNet(num_outputs=args.num_classes).to(device)
    elif args.classifier_arch == "resnetwide":
        classifier = ResNetWide(num_outputs=args.num_classes).to(device)
    elif args.classifier_arch == "efficientnet":
        classifier = EfficientNet.from_name(
            "efficientnet-b0", num_classes=args.num_classes, dropout_rate=0.2).to(device)
    elif args.classifier_arch == "preact_resnet":
        classifier = PreActResNet101(num_classes=args.num_classes).to(device)
    elif args.classifier_arch == "dropout_resnet":
        classifier = dropout_ResNet(
            dropout_p=args.dropout_p, nb_filters=args.dict_nbatoms, num_outputs=args.num_classes).to(device)
    else:
        raise NotImplementedError

    if not args.no_autoencoder:

        if args.autoencoder_train_supervised:

            autoencoder = autoencoder_dict[args.autoencoder_arch](
                args).to(device)

            for p in autoencoder.parameters():
                p.requires_grad = True

            autoencoder.encoder_no_update()

        else:
            autoencoder = get_autoencoder(args)

            for p in autoencoder.parameters():
                p.requires_grad = False

        model = Combined(autoencoder, classifier)

    else:
        model = classifier

    model.train()

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    # Which optimizer to be used for training

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    if args.lr_scheduler == "cyc":
        lr_steps = args.classifier_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 80],
            gamma=0.1)

    elif args.lr_scheduler == "mult":
        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = MultiplicativeLR(optimizer, lr_fun)
    else:
        raise NotImplementedError

    if args.adv_training_attack:

        attacks = dict(
            PGD=PGD,
            PGD_EOT=PGD_EOT,
            PGD_EOT_normalized=PGD_EOT_normalized,
            PGD_EOT_sign=PGD_EOT_sign,
            FGSM=FGSM,
            RFGSM=RFGSM,
        )

        attack_params = {
            "norm": args.adv_training_norm,
            "eps": args.adv_training_epsilon,
            "alpha": args.adv_training_alpha,
            "step_size": args.adv_training_step_size,
            "num_steps": args.adv_training_num_steps,
            "random_start": (
                args.adv_training_rand and args.adv_training_num_restarts > 1
            ),
            "num_restarts": args.adv_training_num_restarts,
            "EOT_size": args.adv_training_EOT_size,
        }

        data_params = {"x_min": 0.0, "x_max": 1.0}

        if "CWlinf" in args.adv_training_attack:
            adv_training_attack = args.adv_training_attack.replace(
                "CWlinf", "PGD")
            loss_function = "carlini_wagner"
        else:
            adv_training_attack = args.adv_training_attack
            loss_function = "cross_entropy"

        adversarial_args = dict(
            attack=attacks[adv_training_attack],
            attack_args=dict(
                net=model, data_params=data_params, attack_params=attack_params
            ),
            loss_function=loss_function,
        )

        logger.info(args.adv_training_attack + " training")
        logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

        for epoch in tqdm(range(1, args.classifier_epochs + 1)):
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
        for epoch in tqdm(range(1, args.classifier_epochs + 1)):
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
    if args.save_checkpoint:
        if not os.path.exists(args.directory + "checkpoints/classifiers/"):
            os.makedirs(args.directory + "checkpoints/classifiers/")

        classifier_filepath = classifier_ckpt_namer(args)
        torch.save(
            classifier.state_dict(), classifier_filepath,
        )

        logger.info(f"Saved to {classifier_filepath}")

        if not args.no_autoencoder:
            if not os.path.exists(args.directory + "checkpoints/autoencoders/"):
                os.makedirs(args.directory + "checkpoints/autoencoders/")

            autoencoder_filepath = autoencoder_ckpt_namer(args)
            if args.autoencoder_train_supervised:
                torch.save(
                    autoencoder.state_dict(), autoencoder_filepath,
                )

            logger.info(f"Saved to {autoencoder_filepath}")


if __name__ == "__main__":
    main()
