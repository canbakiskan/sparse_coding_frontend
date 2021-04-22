"""
Hyper-parameters
"""

import argparse
from os import environ, path
import toml
import json
from types import SimpleNamespace
from .utils.powerset import powerset


def get_arguments():
    """ Hyperparameters and other configuration items"""

    if environ.get("PROJECT_PATH") is not None:
        directory = environ["PROJECT_PATH"]
    else:
        import pathlib

        directory = path.dirname(path.abspath(__file__))
        if "src" in directory:
            directory = directory.replace("src", "")

    if directory[-1] == "/" and directory[-2] == "/":
        directory = directory[:-1]
    elif directory[-1] != "/":
        directory += "/"

    parser = argparse.ArgumentParser(
        description="Sparse Coding Frontend source code configuration")

    parser.add_argument(
        "--directory",
        type=str,
        default=directory,
        metavar="",
        help="Directory of experiments",
    )

    # Adversarial testing parameters
    neural_net = parser.add_argument_group(
        "neural_net", "Neural-net related config"
    )
    neural_net.add_argument(
        "--neural_net_no_frontend",
        action="store_true",
        default=False,
        help="whether to use the frontend or not",
    )
    neural_net.add_argument(
        "--neural_net_epochs",
        type=int,
        default=70,
        metavar="",
        help="Number of epochs in training",
    )
    neural_net.add_argument(
        "--neural_net_lr_scheduler",
        type=str,
        default="cyc",
        choices=["cyc", "step", "mult"],
        help="LR scheduler for training",
    )

    neural_net.add_argument(
        "--neural_net_lr",
        type=float,
        default=0.01,
        help="learning rate for training",
    )

    neural_net.add_argument(
        "--neural_net_lr_max",
        type=float,
        default=0.02,
        help="maximum learning rate for cyclic LR scheduler",
    )

    # Adversarial training parameters
    adv_training = parser.add_argument_group(
        "adv_training", "Adversarial training related config"
    )

    adv_training.add_argument(
        "-tra",
        "--adv_training_method",
        type=str,
        default=None,
        choices=[
            "TRADES",
            "FGSM",
            "RFGSM",
            "PGD",
            "PGD_EOT",
            "PGD_EOT_normalized",
            "PGD_EOT_sign",
            "CWlinf_EOT",
            "CWlinf_EOT_normalized",
            "CWlinf",
        ],
        metavar="fgsm/pgd",
        help="Attack method",
    )

    # Adversarial testing parameters
    adv_testing = parser.add_argument_group(
        "adv_testing", "Adversarial testing related config"
    )

    adv_testing.add_argument(
        "--attack_box_type",
        type=str,
        default="white",
        choices=["other", "white"],
        metavar="other/white",
        help="Box type",
    )

    adv_testing.add_argument(
        "--attack_otherbox_method",
        type=str,
        default="boundary",
        choices=["transfer", "boundary", "hopskip", "genattack", "zoo"],
        metavar="",
        help="Other box (black, pseudowhite) attack method",
    )

    adv_testing.add_argument(
        "--attack_transfer_file",
        type=str,
        default=None,
        help="Source file for the transfer attack (only filename)",
    )

    adv_testing.add_argument(
        "-at_method",
        "--attack_method",
        type=str,
        default="PGD",
        choices=[
            "PGD_EOT",
            "PGD_smooth",
            "FGSM",
            "RFGSM",
            "PGD",
            "PGD_EOT",
            "PGD_EOT_normalized",
            "PGD_EOT_sign",
            "CWlinf_EOT",
            "CWlinf_EOT_normalized",
            "CWlinf",
        ],
        help="Attack method for white/semiwhite box attacks",
    )

    adv_testing.add_argument(
        "--attack_top_T_backward",
        type=str,
        default="top_U",
        choices=["default", "top_U"],
        metavar="",
        help="Top T operation backward pass type",
    )
    adv_testing.add_argument(
        "--attack_dropout_backward",
        type=str,
        default="default",
        choices=["default", "identity"],
        metavar="",
        help="Dropout operation backward pass type",
    )
    adv_testing.add_argument(
        "--attack_activation_backward_steepness",
        type=float,
        default=4.0,
        metavar="",
        help="Steepness of smooth backward pass approximation to activation function. 0.0 means identity backward pass. (default: 4.0)",
    )
    adv_testing.add_argument(
        "--attack_top_U",
        type=int,
        default=30,
        metavar="top_U",
        help="How many top coefficients to back propagate through",
    )

    adv_testing.add_argument(
        "-at_norm",
        "--attack_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Which attack norm to use",
    )
    adv_testing.add_argument(
        "-at_eps",
        "--attack_budget",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
    )
    adv_testing.add_argument(
        "-at_ss",
        "--attack_step_size",
        type=float,
        default=(1.0 / 255.0),
        metavar="",
        help="Step size for PGD",
    )
    adv_testing.add_argument(
        "-at_ni",
        "--attack_nb_steps",
        type=int,
        default=40,
        metavar="",
        help="Number of steps for PGD",
    )

    adv_testing.add_argument(
        "-at_rand",
        "--attack_rand",
        type=bool,
        default=True,
        help="randomly initialize PGD attack",
    )
    adv_testing.add_argument(
        "-at_nr",
        "--attack_nb_restarts",
        type=int,
        default=100,
        metavar="",
        help="number of restarts for PGD",
    )
    adv_testing.add_argument(
        "-at_eot",
        "--attack_EOT_size",
        type=int,
        default=40,
        metavar="",
        help="number of parallel models for eot PGD",
    )

    ablation = parser.add_argument_group(
        "ablation", "Ablation related config"
    )

    ablation.add_argument(
        "--ablation_no_dictionary",
        action="store_true",
        default=False,
        help="first layer of frontend is also trained, no fixed dictionary is used",
    )

    ablation.add_argument(
        "--ablation_distill",
        action="store_true",
        default=False,
        help="whether to use distilled model in run_attack",
    )

    # Defense
    defense = parser.add_argument_group("defense", "Defense arguments")

    defense.add_argument(
        "--defense_frontend_arch",
        type=str,
        default="top_T_activation_frontend",
        choices=[(encoder_name + "_" + decoder_name + "_frontend").replace("__", "_") for encoder_name in ["_".join(parts) for parts in powerset(
            ["top_T", "dropout", "activation", "noisy"])][1:] for decoder_name in ["", "small", "deep", "resize", "identity"]]+["sparse_frontend"],
        metavar="frontend_model",
        help="Frontend model name (default: top_T_activation_frontend)",
    )

    defense.add_argument(
        "--defense_dropout_p",
        type=float,
        default=0.95,
        metavar="frontend_dropout_p",
        help="Probability of dropping first neurons of frontend if frontend is dropout type (default: 0.0)",
    )

    defense.add_argument(
        "--defense_train_noise_gamma",
        type=float,
        default=100.0,
        metavar="train_noise_gamma",
        help="coefficient in front of l1 scaled noise times epsilon (default: 1.0)",
    )
    defense.add_argument(
        "--defense_test_noise_gamma",
        type=float,
        default=0.0,
        metavar="test_noise_gamma",
        help="coefficient in front of l1 scaled noise times epsilon (default: 1.0)",
    )

    defense.add_argument(
        "--defense_top_T", type=int, default=15, metavar="top_T", help="",
    )

    args = parser.parse_args()

    config = toml.load(path.join(args.directory, "src", "config.toml"))
    config = json.loads(json.dumps(config),
                        object_hook=lambda d: SimpleNamespace(**d))

    config.directory = args.directory

    for arg, val in args.__dict__.items():
        if "neural_net" in arg:
            arg = arg.replace("neural_net_", "")
            if arg == "lr_scheduler" or arg == "lr":
                field = config.neural_net.optimizer
            elif arg == "lr_max":
                field = config.neural_net.optimizer.cyclic
            else:
                field = config.neural_net
            setattr(field, arg, val)
        if "adv_training" in arg:
            setattr(config.adv_training, arg.replace("adv_training_", ""), val)
        if "attack_" in arg:
            setattr(config.adv_testing, arg.replace("attack_", ""), val)
        if "defense_" in arg:
            setattr(config.defense, arg.replace("defense_", ""), val)
        if "ablation_" in arg:
            setattr(config.ablation, arg.replace("ablation_", ""), val)

    if config.adv_testing.norm != 'inf':
        config.adv_testing.norm = int(config.adv_testing.norm)
    if config.adv_training.norm != 'inf':
        config.adv_training.norm = int(config.adv_training.norm)

    config.defense.patch_shape = (
        config.defense.patch_size, config.defense.patch_size, 3)

    if config.neural_net.optimizer.lr_scheduler == "cyc" and config.neural_net.optimizer.name != "sgd":
        raise AssertionError("Cyclic learning rate can only be used with SGD.")

    if config.dictionary.type == "dct":
        from numpy import product
        config.dictionary.nb_atoms = product(config.defense.patch_shape)

    return config
