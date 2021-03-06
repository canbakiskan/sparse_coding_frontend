import torch
from torchvision import datasets, transforms

import numpy as np
from os.path import join
from .namers import attack_file_namer


def tiny_imagenet(args):

    data_dir = join(args.directory, 'data')
    train_dir = join(data_dir, "original_datasets",
                     "tiny-imagenet-200", "train")
    test_dir = join(data_dir, "original_datasets",
                    "tiny-imagenet-200", "val")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.neural_net.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def tiny_imagenet_from_file(args):
    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.adv_testing.method == "transfer":
        filepath = join(
            args.directory, 'data',
            'attacked_datasets', args.dataset.name, args.adv_testing.transfer_file
        )

    else:
        filepath = attack_file_namer(args)

    test_images = np.load(filepath)

    data_dir = join(args.directory, 'data')
    test_dir = join(data_dir, "original_datasets",
                    "tiny-imagenet-200", "val")
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.neural_net.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def imagenette(args):

    data_dir = join(args.directory, 'data')
    train_dir = join(data_dir, "original_datasets",
                     "imagenette2-160", "train")
    test_dir = join(data_dir, "original_datasets",
                    "imagenette2-160", "val")

    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop((160), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.CenterCrop(160), transforms.ToTensor()]
    )

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.neural_net.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def imagenette_from_file(args):
    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.adv_testing.method == "transfer":
        filepath = join(
            args.directory, 'data', 'attacked_datasets', args.dataset.name, args.adv_testing.transfer_file
        )

    else:
        filepath = attack_file_namer(args)

    test_images = np.load(filepath)

    data_dir = join(args.directory, 'data')
    test_dir = join(data_dir, "original_datasets",
                    "imagenette2-160", "val")
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.neural_net.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def cifar10(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=join(args.directory, 'data', 'original_datasets'),
        train=True,
        download=True,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.neural_net.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root=join(args.directory, 'data', 'original_datasets'),
        train=False,
        download=True,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.neural_net.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def cifar10_from_file(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.adv_testing.method == "transfer":
        filepath = join(
            args.directory, 'data', 'attacked_datasets', args.dataset.name, args.adv_testing.transfer_file
        )

    else:
        filepath = attack_file_namer(args)

    test_images = np.load(filepath)

    cifar10 = datasets.CIFAR10(
        join(args.directory, 'data', 'original_datasets'),
        train=False,
        transform=None,
        target_transform=None,
        download=False,
    )

    tensor_x = torch.Tensor(test_images / np.max(test_images))
    tensor_y = torch.Tensor(cifar10.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.neural_net.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader


def imagenet(args):

    data_dir = join(args.directory, 'data')
    train_dir = join(data_dir, "original_datasets", "imagenet", "train")
    test_dir = join(data_dir, "original_datasets", "imagenet", "val")

    use_cuda = args.use_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.neural_net.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.neural_net.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def read_dataset(args):

    if args.dataset.name == "CIFAR10":
        train_loader, test_loader = cifar10(args)
    elif args.dataset.name == "Tiny-ImageNet":
        train_loader, test_loader = tiny_imagenet(args)
    elif args.dataset.name == "Imagenette":
        train_loader, test_loader = imagenette(args)
    elif args.dataset.name == "Imagenet":
        train_loader, test_loader = imagenet(args)
    else:
        raise NotImplementedError

    return train_loader, test_loader


def read_test_dataset_from_file(args):

    if args.dataset.name == "CIFAR10":
        test_loader = cifar10_from_file(args)
    elif args.dataset.name == "Tiny-ImageNet":
        test_loader = tiny_imagenet_from_file(args)
    elif args.dataset.name == "Imagenette":
        test_loader = imagenette_from_file(args)
    else:
        raise NotImplementedError

    return test_loader
