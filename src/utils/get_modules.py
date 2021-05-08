import numpy as np
import torch
from os import path
from .namers import (
    dict_file_namer,
    frontend_ckpt_namer,
    classifier_ckpt_namer
)
from ..models.frontend import frontend_class


def create_classifier(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.neural_net.classifier_arch == "resnetwide":
        from ..models.resnet import ResNetWide
        classifier = ResNetWide(num_outputs=args.dataset.nb_classes).to(device)

    elif args.neural_net.classifier_arch == "resnet":
        from ..models.resnet import ResNet
        classifier = ResNet(num_outputs=args.dataset.nb_classes).to(device)

    elif args.neural_net.classifier_arch == "efficientnet":
        from ..models.efficientnet import EfficientNet
        classifier = EfficientNet.from_name(
            "efficientnet-b0", num_classes=args.dataset.nb_classes, dropout_rate=0.2).to(device)

    elif args.neural_net.classifier_arch == "preact_resnet":
        from ..models.preact_resnet import PreActResNet101
        classifier = PreActResNet101(
            num_classes=args.dataset.nb_classes).to(device)

    elif args.neural_net.classifier_arch == "dropout_resnet":
        from ..models.ablation.dropout_resnet import dropout_ResNet
        classifier = dropout_ResNet(
            dropout_p=args.defense.dropout_p, nb_filters=args.dictionary.nb_atoms, num_outputs=args.dataset.nb_classes).to(device)
    elif args.neural_net.classifier_arch == "resnet_after_encoder":
        from ..models.ablation.resnet_after_encoder import ResNet_after_encoder
        classifier = ResNet_after_encoder(
            nb_filters=args.dictionary.nb_atoms, num_outputs=args.dataset.nb_classes).to(device)

    else:
        raise NotImplementedError

    return classifier


def load_classifier(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classifier = create_classifier(args)

    try:
        param_dict = torch.load(classifier_ckpt_namer(args),
                                map_location=torch.device(device),)
    except:
        raise FileNotFoundError(classifier_ckpt_namer(args))

    if "module" in list(param_dict.keys())[0]:
        for _ in range(len(param_dict)):
            key, val = param_dict.popitem(False)
            param_dict[key.replace("module.", "")] = val

    classifier.load_state_dict(param_dict)

    print(f"Classifier: {classifier_ckpt_namer(args)}")

    return classifier


def create_frontend(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    frontend = frontend_class(args).to(device)

    return frontend


def load_frontend(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    frontend = create_frontend(args)

    try:
        frontend_checkpoint = torch.load(
            frontend_ckpt_namer(args), map_location=device
        )

    except:
        raise FileNotFoundError(frontend_ckpt_namer(args))

    frontend.load_state_dict(frontend_checkpoint)

    print(f"Frontend: {frontend_ckpt_namer(args)}")

    return frontend


def get_dictionary(args):
    dict_filepath = dict_file_namer(args)
    if not path.exists(dict_filepath):
        print(
            "Dictionary with given patch size, lambda and number of atoms not found. Please run learn_patch_dict.py"
        )
        exit()
    else:
        f = np.load(dict_filepath, allow_pickle=True)
        dictionary = torch.Tensor(f["dict"]).t()

    print(f"Dictionary: {dict_filepath}")
    return dictionary
