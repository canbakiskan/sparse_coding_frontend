
from ..utils.get_modules import (
    get_classifier,
    get_frontend,
)

import torch
from ..models.combined import Combined

from ..utils.read_datasets import(
    cifar10,
    tiny_imagenet,
    imagenette,
)

import logging
from deepillusion.torchattacks.analysis.plot import loss_landscape
import time
logger = logging.getLogger(__name__)


def main():

    from ..parameters import get_arguments

    args = get_arguments()

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classifier = get_classifier(args)

    if args.neural_net.no_frontend:
        model = classifier

    else:
        frontend = get_frontend(args)

        if args.adv_testing.box_type == "white":
            if args.adv_testing.backward == "top_T_dropout_identity":
                frontend.set_BPDA_type("identity")

            elif args.adv_testing.backward == "top_T_top_U":
                frontend.set_BPDA_type("top_U")

        model = Combined(frontend, classifier)

        model = model.to(device)
        model.eval()

    model.eval()

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

    data_params = {"x_min": 0.0, "x_max": 1.0}

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

    loss_landscape(
        model,
        test_loader,
        data_params,
        attack_params,
        img_index=0,
        second_direction='random',
        fig_name='confidence_top10_step0.5_avg100_bigger.pdf',
        norm='inf',
        avg_points=100,
        z_axis_type='confidence',
        verbose=False,
    )


if __name__ == "__main__":
    main()
