import torch.optim as optim


def get_optimizer_scheduler(args, model, batches_per_epoch):

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )

    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    if args.lr_scheduler == "cyc":
        lr_steps = args.classifier_epochs * batches_per_epoch
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[35], gamma=0.1
        )

    elif args.lr_scheduler == "mult":

        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fun)
    else:
        raise NotImplementedError

    return optimizer, scheduler
