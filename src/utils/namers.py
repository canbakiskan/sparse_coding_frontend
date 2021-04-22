import os
import numpy as np


def dict_params_string(args):

    if args.ablation.no_dictionary:
        dictionary_parameters_string = "_nodict"

    else:
        if args.dictionary.type == "dct":
            dictionary_parameters_string = f"_dct_ps_{args.defense.patch_size}"

        elif args.dictionary.type == "overcomplete":

            dictionary_parameters = {
                "ps": args.defense.patch_size,
                "st": args.defense.stride,
                "l": args.dictionary.lamda,
                "n": args.dictionary.nb_atoms,
                "it": args.dictionary.iter,
            }

            dictionary_parameters_string = ""
            for key in dictionary_parameters:
                dictionary_parameters_string += (
                    "_" + str(key) + "_" + str(dictionary_parameters[key])
                )

    return dictionary_parameters_string[1:]


def frontend_params_string(args):

    frontend_params_string = dict_params_string(args)

    frontend_params_string += f"_{args.defense.frontend_arch}"

    frontend_params_string += f"_{args.neural_net.optimizer.name}"

    frontend_params_string += f"_{args.neural_net.optimizer.lr_scheduler}"

    if args.neural_net.optimizer.lr_scheduler == "cyc":
        frontend_params_string += f"_{args.neural_net.optimizer.cyclic.lr_max:.4f}"
    else:
        frontend_params_string += f"_{args.neural_net.optimizer.lr:.4f}"

    if "top_T" in args.defense.frontend_arch or "sparse" in args.defense.frontend_arch:
        frontend_params_string += f"_T_{args.defense.top_T}"

    if "activation" in args.defense.frontend_arch:
        frontend_params_string += (
            f"_j_{np.int(np.round(args.defense.activation_beta*args.defense.assumed_budget*255))}"
        )

    if "dropout" in args.defense.frontend_arch:
        frontend_params_string += f"_p_{args.defense.dropout_p:.2f}"

    if "noisy" in args.defense.frontend_arch:
        frontend_params_string += f"_g_{args.defense.train_noise_gamma:.2f}"

    frontend_params_string += f"_ep_{args.neural_net.epochs}"

    return frontend_params_string


def adv_training_params_string(args):
    adv_training_params_string = ""
    if args.adv_training.method:

        if args.adv_training.norm != 'inf':
            attack_params_string += f"_L{args.adv_training.norm}"

        adv_training_params_string += f"_{args.adv_training.method}"
        adv_training_params_string += (
            f"_eps_{np.int(np.round(args.adv_training.budget*255))}"
        )
        if "EOT" in args.adv_training.method:
            adv_training_params_string += f"_Ne_{args.adv_training.EOT_size}"
        if "PGD" in args.adv_training.method or "CW" in args.adv_training.method:
            adv_training_params_string += f"_Ns_{args.adv_training.nb_steps}"
            adv_training_params_string += (
                f"_ss_{np.int(np.round(args.adv_training.step_size*255))}"
            )
            adv_training_params_string += f"_Nr_{args.adv_training.nb_restarts}"
        if "FGSM" in args.adv_training.method:
            adv_training_params_string += (
                f"_a_{np.int(np.round(args.adv_training.rfgsm_alpha*255))}"
            )

    return adv_training_params_string


def classifier_params_string(args):
    classifier_params_string = args.neural_net.classifier_arch

    classifier_params_string += f"_{args.neural_net.optimizer.name}"

    classifier_params_string += f"_{args.neural_net.optimizer.lr_scheduler}"

    if args.neural_net.optimizer.lr_scheduler == "cyc":
        classifier_params_string += f"_{args.neural_net.optimizer.cyclic.lr_max:.4f}"
    else:
        classifier_params_string += f"_{args.neural_net.optimizer.lr:.4f}"

    if args.neural_net.classifier_arch == "dropout_resnet":
        classifier_params_string += f"_p_{args.defense.dropout_p:.2f}"
        classifier_params_string += f"_n_{args.dictionary.nb_atoms}"

    classifier_params_string += adv_training_params_string(args)

    if not args.adv_training.method and args.neural_net.no_frontend:
        classifier_params_string += "_NT"

    classifier_params_string += f"_ep_{args.neural_net.epochs}"

    if not args.neural_net.no_frontend:
        classifier_params_string += "_"
        classifier_params_string += frontend_params_string(args)

    return classifier_params_string


def attack_params_string(args):
    attack_params_string = f"{args.adv_testing.box_type}"
    if args.adv_testing.box_type == "other":
        attack_params_string += f"_{args.adv_testing.otherbox_method}"
        attack_params_string += f"_eps_{np.int(np.round(args.adv_testing.budget*255))}"

        if args.adv_testing.zoo_use_tanh:
            attack_params_string += f"_tanh"

    elif args.adv_testing.box_type == "white":

        if args.adv_testing.norm != 'inf':
            attack_params_string += f"_L{args.adv_testing.norm}"

        attack_params_string += f"_{args.adv_testing.method}"

        attack_params_string += f"_eps_{np.int(np.round(args.adv_testing.budget*255))}"

        if "EOT" in args.adv_testing.method:
            attack_params_string += f"_Ne_{args.adv_testing.EOT_size}"
        if "PGD" in args.adv_testing.method or "CW" in args.adv_testing.method:
            attack_params_string += f"_Ns_{args.adv_testing.nb_steps}"
            attack_params_string += f"_ss_{np.int(np.round(args.adv_testing.step_size*255))}"
            attack_params_string += f"_Nr_{args.adv_testing.nb_restarts}"
        if "RFGSM" in args.adv_testing.method:
            attack_params_string += f"_a_{np.int(np.round(args.adv_testing.rfgsm_alpha*255))}"

        if ("top_T" in args.defense.frontend_arch or "sparse" in args.defense.frontend_arch) and args.adv_testing.top_T_backward == "top_U":
            attack_params_string += f"_U_{args.adv_testing.top_U}"

        if "dropout" in args.defense.frontend_arch and args.adv_testing.dropout_backward == "identity":
            attack_params_string += f"_dropout_identity"

        if "activation" in args.defense.frontend_arch and args.adv_testing.activation_backward_steepness != 0.0:
            attack_params_string += f"_steep_{args.adv_testing.activation_backward_steepness:.1f}"

    return attack_params_string


def dict_file_namer(args):

    data_dir = args.directory + "data/"

    dict_filepath = os.path.join(
        data_dir, "dictionaries", args.dataset.name, dict_params_string(
            args) + ".npz"
    )
    return dict_filepath


def frontend_ckpt_namer(args):

    file_path = args.directory + \
        f"checkpoints/frontends/{args.dataset.name}/"

    file_path += frontend_params_string(args)

    file_path += ".pt"
    return file_path


def frontend_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset.name}/"

    file_path += frontend_params_string(args)

    file_path += ".log"

    return file_path


def classifier_ckpt_namer(args):

    file_path = args.directory + \
        f"checkpoints/classifiers/{args.dataset.name}/"

    file_path += classifier_params_string(args)

    file_path += ".pt"

    return file_path


def classifier_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset.name}/"

    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path


def attack_file_namer(args):

    file_path = args.directory + f"data/attacked_datasets/{args.dataset.name}/"

    file_path += attack_params_string(args)
    file_path += "_"
    if args.ablation.distill:
        file_path += "distill_"
    file_path += classifier_params_string(args)

    file_path += ".npy"

    return file_path


def attack_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset.name}/"

    file_path += attack_params_string(args)
    file_path += "_"
    if args.ablation.distill:
        file_path += "distill_"
    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path


def distillation_ckpt_namer(args):

    file_path = args.directory + \
        f"checkpoints/classifiers/{args.dataset.name}/"

    file_path += "distill_"
    file_path += classifier_params_string(args)

    file_path += ".npy"

    return file_path
