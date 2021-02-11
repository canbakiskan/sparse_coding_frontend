import torch
import os
import numpy as np


def dict_params_string(args):

    if args.ablation_no_dictionary:
        dictionary_parameters_string = "_nodict"

    else:
        if args.dict_type == "dct":
            dictionary_parameters_string = f"_dct_ps_{args.defense_patchsize}"

        elif args.dict_type == "overcomplete":

            dictionary_parameters = {
                "ps": args.defense_patchsize,
                "st": args.defense_stride,
                "l": args.dict_lambda,
                "n": args.dict_nbatoms,
                "it": args.dict_iter,
            }

            dictionary_parameters_string = ""
            for key in dictionary_parameters:
                dictionary_parameters_string += (
                    "_" + str(key) + "_" + str(dictionary_parameters[key])
                )

    return dictionary_parameters_string[1:]


def autoencoder_params_string(args):

    autoencoder_params_string = dict_params_string(args)

    autoencoder_params_string += f"_{args.autoencoder_arch}"

    autoencoder_params_string += f"_{args.optimizer}"

    autoencoder_params_string += f"_{args.lr_scheduler}"

    if args.lr_scheduler == "cyc":
        autoencoder_params_string += f"_{args.lr_max:.4f}"
    else:
        autoencoder_params_string += f"_{args.lr:.4f}"

    if "top_T" in args.autoencoder_arch or "sparse" in args.autoencoder_arch:
        autoencoder_params_string += f"_T_{args.top_T}"

    if "quant" in args.autoencoder_arch:
        autoencoder_params_string += (
            f"_j_{np.int(np.round(args.activation_beta*args.defense_epsilon*255))}"
        )

    if "dropout" in args.autoencoder_arch:
        autoencoder_params_string += f"_p_{args.dropout_p:.2f}"

    if "blur" in args.autoencoder_arch:
        autoencoder_params_string += f"_s_{args.ablation_blur_sigma:.2f}"

    if "noisy" in args.autoencoder_arch:
        autoencoder_params_string += f"_g_{args.noise_gamma:.2f}"

    if args.autoencoder_train_supervised:
        autoencoder_params_string += "_S"
        autoencoder_params_string += f"_ep_{args.classifier_epochs}"
    else:
        autoencoder_params_string += "_US"
        autoencoder_params_string += f"_ep_{args.autoencoder_epochs}"

    return autoencoder_params_string


def adv_training_params_string(args):
    adv_training_params_string = ""
    if args.adv_training_attack:
        adv_training_params_string += f"_{args.adv_training_attack}"
        adv_training_params_string += (
            f"_eps_{np.int(np.round(args.adv_training_epsilon*255))}"
        )
        if "EOT" in args.adv_training_attack:
            adv_training_params_string += f"_Ne_{args.adv_training_EOT_size}"
        if "PGD" in args.adv_training_attack or "CW" in args.adv_training_attack:
            adv_training_params_string += f"_Ns_{args.adv_training_num_steps}"
            adv_training_params_string += (
                f"_ss_{np.int(np.round(args.adv_training_step_size*255))}"
            )
            adv_training_params_string += f"_Nr_{args.adv_training_num_restarts}"
        if "FGSM" in args.adv_training_attack:
            adv_training_params_string += (
                f"_a_{np.int(np.round(args.adv_training_alpha*255))}"
            )

    return adv_training_params_string


def classifier_params_string(args):
    classifier_params_string = args.classifier_arch

    classifier_params_string += f"_{args.optimizer}"

    classifier_params_string += f"_{args.lr_scheduler}"

    if args.lr_scheduler == "cyc":
        classifier_params_string += f"_{args.lr_max:.4f}"
    else:
        classifier_params_string += f"_{args.lr:.4f}"

    if args.classifier_arch == "dropout_resnet":
        classifier_params_string += f"_p_{args.dropout_p:.2f}"
        classifier_params_string += f"_n_{args.dict_nbatoms}"

    classifier_params_string += adv_training_params_string(args)

    if not args.adv_training_attack and args.no_autoencoder:
        classifier_params_string += "_NT"

    classifier_params_string += f"_ep_{args.classifier_epochs}"

    if not args.no_autoencoder:
        classifier_params_string += "_"
        classifier_params_string += autoencoder_params_string(args)

    return classifier_params_string


def attack_params_string(args):
    attack_params_string = f"{args.attack_box_type}"
    if args.attack_box_type == "other":
        attack_params_string += f"_{args.attack_otherbox_type}"
        attack_params_string += f"_eps_{np.int(np.round(args.attack_epsilon*255))}"

    elif args.attack_box_type == "white":
        if not args.no_autoencoder:
            attack_params_string += f"_{args.attack_whitebox_type}"
        attack_params_string += f"_{args.attack_method}"
        attack_params_string += f"_eps_{np.int(np.round(args.attack_epsilon*255))}"
        if "EOT" in args.attack_method:
            attack_params_string += f"_Ne_{args.attack_EOT_size}"
        if "PGD" in args.attack_method or "CW" in args.attack_method:
            attack_params_string += f"_Ns_{args.attack_num_steps}"
            attack_params_string += f"_ss_{np.int(np.round(args.attack_step_size*255))}"
            attack_params_string += f"_Nr_{args.attack_num_restarts}"
        if "RFGSM" in args.attack_method:
            attack_params_string += f"_a_{np.int(np.round(args.attack_alpha*255))}"
        if args.attack_whitebox_type == "W-AGGA":
            attack_params_string += f"_sig_{args.ablation_blur_sigma:.2f}"
        if args.attack_whitebox_type == "W-NFGA" and args.attack_quantization_BPDA_steepness != 0.0:
            attack_params_string += f"_steep_{args.attack_quantization_BPDA_steepness:.1f}"

    return attack_params_string


def dict_file_namer(args):

    data_dir = args.directory + "data/"

    dict_filepath = os.path.join(
        data_dir, "dictionaries", args.dataset, dict_params_string(
            args) + ".npz"
    )
    return dict_filepath


def autoencoder_ckpt_namer(args):

    file_path = args.directory + f"checkpoints/autoencoders/{args.dataset}/"

    file_path += autoencoder_params_string(args)

    file_path += ".pt"
    return file_path


def autoencoder_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset}/"

    file_path += autoencoder_params_string(args)

    file_path += ".log"

    return file_path


def classifier_ckpt_namer(args):

    file_path = args.directory + f"checkpoints/classifiers/{args.dataset}/"

    file_path += classifier_params_string(args)

    file_path += ".pt"

    return file_path


def classifier_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset}/"

    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path


def attack_file_namer(args):

    file_path = args.directory + f"data/attacked_dataset/{args.dataset}/"

    file_path += attack_params_string(args)
    file_path += "_"
    file_path += classifier_params_string(args)

    file_path += ".npy"

    return file_path


def attack_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset}/"

    file_path += attack_params_string(args)
    file_path += "_"
    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path
