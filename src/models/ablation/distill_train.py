import time
import sys
from ..frontend import frontend_class
from deepillusion.torchdefenses import adversarial_epoch, adversarial_test
from deepillusion.torchattacks import (
    PGD,
    PGD_EOT,
    FGSM,
    RFGSM,
    PGD_EOT_normalized,
    PGD_EOT_sign,
)
from ...utils.get_modules import get_frontend
from ..combined import Combined
from ...utils.namers import (
    frontend_ckpt_namer,
    frontend_log_namer,
    classifier_ckpt_namer,
    classifier_log_namer,
    distillation_ckpt_namer
)
from ...utils.get_optimizer_scheduler import get_optimizer_scheduler
from ...utils.read_datasets import read_dataset
from ...parameters import get_arguments
from ...train_test_functions import train, test
from ..ablation.resnet_after_encoder import ResNet_after_encoder
from ..ablation.dropout_resnet import dropout_ResNet
from ..preact_resnet import PreActResNet101
from ..efficientnet import EfficientNet
from ..resnet import ResNet, ResNetWide
import torch.backends.cudnn as cudnn
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
import os
import torch
from ...utils.namers import (
    attack_log_namer,
    attack_file_namer,
)
from ...utils.get_modules import (
    get_classifier,
    get_frontend,
)
from ..combined import Combined, Combined_inner_BPDA_identity


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean    Examples::        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def teacher_epoch(model, teacher_model, train_loader, optimizer,
                  scheduler=None, adversarial_args=None):
    model.train()
    teacher_model.eval()
    device = model.parameters().__next__().device
    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # if adversarial_args and adversarial_args["attack"]:
        #     adversarial_args["attack_args"]["net"] = model
        #     adversarial_args["attack_args"]["x"] = data
        #     adversarial_args["attack_args"]["y_true"] = target
        #     perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
        #     data_perturbed = data + perturbs        # model.l1_normalize_weights()
        optimizer.zero_grad()
        output = model(data)
        teacher_labels = teacher_model(data)
        softmax_ = torch.nn.Softmax(1)
        teacher_labels = softmax_(teacher_labels)
        # breakpoint()
        loss = cross_entropy(output, teacher_labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


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

    # TEACHER MODEL

    teacher_classifier = get_classifier(args)

    if args.neural_net.no_frontend:
        teacher_model = teacher_classifier

    else:
        teacher_frontend = get_frontend(args)

        if args.adv_testing.box_type == "white":
            if args.adv_testing.backward == "top_T_dropout_identity":
                teacher_frontend.set_BPDA_type("identity")

            elif args.adv_testing.backward == "top_T_top_U":
                teacher_frontend.set_BPDA_type("top_U")

            teacher_model = Combined(teacher_frontend, teacher_classifier)

        teacher_model = teacher_model.to(device)
        teacher_model.eval()

    teacher_model.eval()

    for p in teacher_model.parameters():
        p.requires_grad = False
    # TEACHER MODEL

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

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

    logger.info("Standard training")
    for epoch in tqdm(range(1, args.neural_net.nb_epochs + 1)):
        start_time = time.time()

        train_loss, train_acc = teacher_epoch(model, teacher_model, train_loader, optimizer,
                                              scheduler=scheduler)
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
        if not os.path.exists(os.dirname(distillation_ckpt_namer(args))):
            os.makedirs(os.dirname(distillation_ckpt_namer(args)))

        distillation_filepath = distillation_ckpt_namer(args)
        torch.save(classifier.state_dict(), distillation_filepath)

        logger.info(f"Saved to {distillation_filepath}")


if __name__ == "__main__":
    main()
