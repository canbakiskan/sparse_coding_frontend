import torch


def determine_device(args):

    use_cuda = args.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device
