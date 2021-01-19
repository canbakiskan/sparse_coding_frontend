"""
Hyper-parameters
"""

import argparse
from os import environ


def get_arguments():
    """ Hyper-parameters """

    parser = argparse.ArgumentParser(
        description="AAAI project source code parameters")

    # Directory

    if environ.get("PROJECT_PATH") is not None:
        directory = environ["PROJECT_PATH"]
    else:
        import pathlib

        directory = str(pathlib.Path().absolute())
        if "src" in directory:
            directory = directory.replace("src", "")

    if directory[-1] == "/" and directory[-2] == "/":
        directory = directory[:-1]
    elif directory[-1] != "/":
        directory += "/"

    parser.add_argument(
        "--directory",
        type=str,
        default=directory,
        metavar="",
        help="Directory of experiments",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "Tiny-ImageNet", "Imagenette"],
        help="Directory of experiments",
    )

    neural_net = parser.add_argument_group(
        "neural_net", "Neural Network arguments")

    # Neural Model
    neural_net.add_argument(
        "--classifier_arch",
        type=str,
        choices=["resnet", "resnetwide", "efficientnet",
                 "preact_resnet", "dropout_resnet"],
        default="resnet",
        metavar="classifier_name",
        help="Which classifier to use",
    )

    # Neural Model
    neural_net.add_argument(
        "--lr_scheduler",
        type=str,
        default="cyc",
        choices=["cyc", "mult", "step"],
        metavar="optimizer name",
        help="Which optimizer to use",
    )

    # Optimizer
    neural_net.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "rms"],
        metavar="optimizer name",
        help="Which optimizer to use",
    )

    neural_net.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="Learning rate",
    )
    neural_net.add_argument(
        "--lr_min", type=float, default=0.0, metavar="LR", help="Learning rate min",
    )
    neural_net.add_argument(
        "--lr_max", type=float, default=0.05, metavar="LR", help="Learning rate max",
    )
    neural_net.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="Optimizer momentum",
    )
    neural_net.add_argument(
        "--weight_decay", type=float, default=0.0005, metavar="WD", help="Weight decay",
    )

    neural_net.add_argument(
        "--save_attack",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to save the attack after training (default: False)",
    )
    # Batch Sizes & #Epochs
    neural_net.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for train",
    )
    neural_net.add_argument(
        "--test_batch_size",
        type=int,
        default=100,
        metavar="N",
        help="Batch size for test",
    )
    neural_net.add_argument(
        "--classifier_epochs",
        type=int,
        default=70,
        metavar="N",
        help="Number of epochs",
    )

    neural_net.add_argument(
        "-sm",
        "--save_checkpoint",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="For Saving the current Model, default = False ",
    )

    neural_net.add_argument(
        "--no_autoencoder", action="store_true", default=False, help="",
    )
    # Adversarial training parameters
    adv_training = parser.add_argument_group(
        "adv_training", "Adversarial training arguments"
    )

    adv_training.add_argument(
        "-tra",
        "--adv_training_attack",
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
    adv_training.add_argument(
        "--adv_training_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Attack norm",
    )
    adv_training.add_argument(
        "-tr_eps",
        "--adv_training_epsilon",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
    )
    adv_training.add_argument(
        "-tr_a",
        "--adv_training_alpha",
        type=float,
        default=(10.0 / 255.0),
        metavar="",
        help="random fgsm budget",
    )
    adv_training.add_argument(
        "-tr_ss",
        "--adv_training_step_size",
        type=float,
        default=(1.0 / 255.0),
        metavar="",
        help="Step size for PGD, adv training",
    )
    adv_training.add_argument(
        "-tr_ns",
        "--adv_training_num_steps",
        type=int,
        default=10,
        metavar="",
        help="Number of steps for PGD, adv training",
    )
    adv_training.add_argument(
        "-tr_rand",
        "--adv_training_rand",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="randomly initialize attack for training",
    )
    adv_training.add_argument(
        "-tr_nr",
        "--adv_training_num_restarts",
        type=int,
        default=1,
        metavar="",
        help="number of restarts for pgd for training",
    )
    adv_training.add_argument(
        "-tr_eot",
        "--adv_training_EOT_size",
        type=int,
        default=10,
        metavar="",
        help="number of parallel models for EOT PGD",
    )

    # Adversarial testing parameters
    adv_testing = parser.add_argument_group(
        "adv_testing", "Adversarial testing arguments"
    )

    adv_testing.add_argument(
        "--attack_skip_clean",
        action="store_true",
        default=False,
        help="skip clean testing",
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
        "--attack_whitebox_type",
        type=str,
        default="W-AIGA",
        choices=["SW", "W-NFGA", "W-AIGA", "W-AGGA"],
        metavar="",
        help="whitebox attack type",
    )

    adv_testing.add_argument(
        "--attack_otherbox_type",
        type=str,
        default="transfer",
        choices=["transfer", "decision"],
        metavar="",
        help="Other box (black, pseudowhite) attack type",
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
        default="PGD_EOT_normalized",
        choices=[
            "PGD_EOT",
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
        "-at_norm",
        "--attack_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Which attack norm to use",
    )
    adv_testing.add_argument(
        "-at_eps",
        "--attack_epsilon",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
    )
    adv_testing.add_argument(
        "-at_a",
        "--attack_alpha",
        type=float,
        default=(10.0 / 255.0),
        metavar="",
        help="RFGSM step size",
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
        "--attack_num_steps",
        type=int,
        default=20,
        metavar="",
        help="Number of steps for PGD",
    )

    adv_testing.add_argument(
        "--at_rand",
        action="store_true",
        default=False,
        help="randomly initialize PGD attack",
    )
    adv_testing.add_argument(
        "-at_nr",
        "--attack_num_restarts",
        type=int,
        default=1,
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
    adv_testing.add_argument(
        "--attack_progress_bar",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="show progress bar during PGD attack",
    )
    adv_testing.add_argument(
        "--attack_quantization_BPDA_steepness",
        type=float,
        default=0.0,
        metavar="",
        help="Steepness of backward pass approximation to activation&quantization function. 0.0 means identity. (default: 0.0)",
    )

    # Others
    others = parser.add_argument_group("others", "Other arguments")

    others.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    others.add_argument(
        "--seed", type=int, default=2020, metavar="S", help="random seed (default: 1)"
    )
    others.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    others.add_argument(
        "--ablation_blur_sigma",
        type=float,
        default=0.625,
        metavar="sigma",
        help="Sigma for the gaussian blur in ablation study",
    )

    # Defense
    defense = parser.add_argument_group("defense", "Defense arguments")

    defense.add_argument(
        "--defense_patchsize",
        type=int,
        default=4,
        metavar="patch_size",
        help="Width=height of each patch (default: 4)",
    )

    defense.add_argument(
        "--defense_stride",
        type=int,
        default=2,
        metavar="stride",
        help="Stride of patches while extracting (default: 2)",
    )

    defense.add_argument(
        "--defense_epsilon",
        type=float,
        default=(8.0 / 255.0),
        metavar="epsilon",
        help="Epsilon – assumed attack budget (/255) for reconstruction problem (default: 8.)",
    )

    defense.add_argument(
        "--autoencoder_arch",
        type=str,
        default="top_T_dropout_quant_autoencoder",
        choices=[
            "quant_autoencoder",
            "top_T_autoencoder",
            "top_T_noisy_autoencoder",
            "top_T_quant_autoencoder",
            "top_T_quant_noisy_autoencoder",
            "top_T_dropout_autoencoder",
            "top_T_dropout_quant_autoencoder",
            "top_T_dropout_quant_small_autoencoder",
            "top_T_dropout_quant_deep_autoencoder",
            "top_T_dropout_quant_resize_autoencoder",
            "top_T_quant_resize_autoencoder",
            "sparse_autoencoder",
            "gaussian_blur"
        ],
        metavar="autoencoder_model",
        help="Autoencoder_model model name (default: top_T_dropout_quant_autoencoder)",
    )

    defense.add_argument(
        "--autoencoder_train_supervised",
        action="store_true",
        help="Whether to retrain the autoencoder while training the classifier (default: False)",
    )

    defense.add_argument(
        "--dropout_p",
        type=float,
        default=0.95,
        metavar="autoencoder_dropout_p",
        help="Probability of dropping first neurons of autoencoder if autoencoder is dropout type (default: 0.0)",
    )

    defense.add_argument(
        "--activation_beta",
        type=float,
        default=3.0,
        metavar="activation_beta",
        help=" (default: 3.0)",
    )
    defense.add_argument(
        "--noise_gamma",
        type=float,
        default=1.0,
        metavar="noise_gamma",
        help="coefficient in front of l1 scaled noise times epsilon  (default: 1.0)",
    )

    defense.add_argument(
        "--autoencoder_epochs", type=int, default=50, metavar="epochs", help="",
    )

    defense.add_argument(
        "--top_T", type=int, default=50, metavar="top_T", help="",
    )

    defense.add_argument(
        "--ensemble_E",
        type=int,
        default=10,
        metavar="nb_parallel_models",
        help="Number of models running in parallel (default: 10)",
    )

    defense.add_argument(
        "--defense_nbimgs",
        type=int,
        default=0,
        metavar="nb_images",
        help="Number of images to process (default: 10000-all)",
    )

    dictionary = parser.add_argument_group(
        "dictionary", "Dictionary arguments")

    dictionary.add_argument(
        "--dict_type",
        type=str,
        default="overcomplete",
        choices=["overcomplete", "dct"],
        help="Type of the dictionary (default: overcomplete)",
    )
    dictionary.add_argument(
        "--dict_nbatoms",
        type=int,
        default=500,
        metavar="nb_atoms",
        help="Number of atoms in the dictionary (default: 100)",
    )
    dictionary.add_argument(
        "--dict_batchsize",
        type=int,
        default=5,
        metavar="dict_learn_batch_size",
        help="Minibatch size while learning dictionary (default: 5)",
    )

    dictionary.add_argument(
        "--dict_display",
        action="store_true",
        help="Whether to save a plot of atoms in the dictionary (only first 100 atoms)",
    )

    dictionary.add_argument(
        "--dict_iter",
        type=int,
        default=1000,
        metavar="N_iter",
        help="Number of iterations when learning the dictionary",
    )

    dictionary.add_argument(
        "--dict_lambda",
        type=float,
        default=1.0,
        metavar="lambda",
        help="Coefficient of L1 norm for dictionary learning (default: 1.0)",
    )

    dictionary.add_argument(
        "--dict_online",
        action="store_true",
        help="Whether to learn the dictionary online",
    )

    args = parser.parse_args()

    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.image_shape = (32, 32, 3)
        if args.defense_nbimgs == 0:
            args.defense_nbimgs = 10000
    elif args.dataset == "Tiny-ImageNet":
        args.num_classes = 200
        args.image_shape = (64, 64, 3)
        if args.defense_nbimgs == 0:
            args.defense_nbimgs = 10000
    elif args.dataset == "Imagenette":
        args.num_classes = 10
        args.image_shape = (160, 160, 3)
        if args.defense_nbimgs == 0:
            args.defense_nbimgs = 3925
    else:
        print(f"Dataset {args.dataset} is not currently available.")
        raise NotImplementedError

    args.defense_patchshape = (
        args.defense_patchsize, args.defense_patchsize, 3)

    if args.lr_scheduler == "cyclic" and args.optimizer != "sgd":
        print("Cyclic learning rate can only be used with SGD.")
        raise AssertionError

    if args.dict_type == "dct":
        from numpy import product
        args.dict_nbatoms = product(args.defense_patchshape)

    return args
