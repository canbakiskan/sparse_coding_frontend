import torch.optim as optim


def get_optimizer_scheduler(args, model, batches_per_epoch):

    if args.neural_net.optimizer.name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.neural_net.optimizer.lr,
            momentum=args.neural_net.optimizer.momentum,
            weight_decay=args.neural_net.optimizer.weight_decay,
        )
    elif args.neural_net.optimizer.name == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.neural_net.optimizer.lr,
            weight_decay=args.neural_net.optimizer.weight_decay,
            momentum=args.neural_net.optimizer.momentum,
        )

    elif args.neural_net.optimizer.name == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.neural_net.optimizer.lr, weight_decay=args.neural_net.optimizer.weight_decay
        )
    else:
        raise NotImplementedError

    if args.neural_net.optimizer.lr_scheduler == "cyc":
        lr_steps = args.neural_net.epochs * batches_per_epoch
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.neural_net.optimizer.cyclic.lr_min,
            max_lr=args.neural_net.optimizer.cyclic.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif args.neural_net.optimizer.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[35], gamma=0.1
        )

    elif args.neural_net.optimizer.lr_scheduler == "mult":

        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fun)
    else:
        raise NotImplementedError

    return optimizer, scheduler
