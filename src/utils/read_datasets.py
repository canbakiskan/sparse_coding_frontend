import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from os import path
from .namers import attack_file_namer


def tiny_imagenet(args):

    data_dir = args.directory + "data/"
    train_dir = path.join(data_dir, "original_dataset",
                          "tiny-imagenet-200", "train")
    test_dir = path.join(data_dir, "original_dataset",
                         "tiny-imagenet-200", "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), ])

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def tiny_imagenet_from_file(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.attack_box_type == "other" and args.attack_otherbox_type == "transfer":
        filepath = args.directory + "data/attacked_dataset/" + \
            args.dataset + "/" + args.attack_transfer_file

    elif args.attack_box_type == "white":
        filepath = attack_file_namer(args)
    else:
        raise AssertionError

    test_images = np.load(filepath)

    data_dir = args.directory + "data/"
    test_dir = path.join(data_dir, "original_dataset",
                         "tiny-imagenet-200", "val")
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def tiny_imagenet_initialization_from_file(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    filepath = args.directory + "data/attacked_dataset/" + \
        args.dataset + "/" + args.attack_initialization_file

    test_images = np.load(filepath)

    data_dir = args.directory + "data/"
    test_dir = path.join(data_dir, "original_dataset",
                         "tiny-imagenet-200", "val")
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def imagenette(args):

    data_dir = args.directory + "data/"
    train_dir = path.join(data_dir, "original_dataset",
                          "imagenette2-160", "train")
    test_dir = path.join(data_dir, "original_dataset",
                         "imagenette2-160", "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop((160), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.CenterCrop(160),
            transforms.ToTensor(),
        ]
    )

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def imagenette_from_file(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.attack_box_type == "other" and args.attack_otherbox_type == "transfer":
        filepath = args.directory + "data/attacked_dataset/" + \
            args.dataset + "/" + args.attack_transfer_file

    elif args.attack_box_type == "white":
        filepath = attack_file_namer(args)
    else:
        raise AssertionError

    test_images = np.load(filepath)

    data_dir = args.directory + "data/"
    test_dir = path.join(data_dir, "original_dataset",
                         "imagenette2-160", "val")
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def imagenette_initialization_from_file(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    filepath = args.directory + "data/attacked_dataset/" + \
        args.dataset + "/" + args.attack_initialization_file

    test_images = np.load(filepath)

    data_dir = args.directory + "data/"
    test_dir = path.join(data_dir, "original_dataset",
                         "imagenette2-160", "val")
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def cifar10(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), ])

    trainset = datasets.CIFAR10(
        root=args.directory + "data/original_dataset",
        train=True,
        download=True,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root=args.directory + "data/original_dataset",
        train=False,
        download=True,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def cifar10_from_file(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.attack_box_type == "other" and args.attack_otherbox_type == "transfer":
        filepath = args.directory + "data/attacked_dataset/" + \
            args.dataset + "/" + args.attack_transfer_file

    elif args.attack_box_type == "white":
        filepath = attack_file_namer(args)
    else:
        raise AssertionError

    test_images = np.load(filepath)

    cifar10 = datasets.CIFAR10(
        path.join(args.directory, "data/original_dataset"),
        train=False,
        transform=None,
        target_transform=None,
        download=False,
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(cifar10.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def cifar10_initialization_from_file(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    filepath = args.directory + "data/attacked_dataset/" + \
        args.dataset + "/" + args.attack_initialization_file

    test_images = np.load(filepath)

    cifar10 = datasets.CIFAR10(
        path.join(args.directory, "data/original_dataset"),
        train=False,
        transform=None,
        target_transform=None,
        download=False,
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(cifar10.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader
