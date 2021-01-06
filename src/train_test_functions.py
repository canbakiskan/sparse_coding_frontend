from tqdm import tqdm

import torch
import torch.nn as nn


def train(model, train_loader, optimizer, scheduler=None):
    """ Train given model with train_loader and optimizer """

    model.train()
    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target, in train_loader:
        if isinstance(data, list):
            data = data[0]
            target = target[0]

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
        scheduler.step()

    train_size = len(train_loader.dataset)

    return train_loss / train_size, train_correct / train_size


def train_autoencoder_unsupervised(model, train_loader, optimizer, scheduler=None):
    """ Train given autoencoder with train_loader and optimizer """

    model.train()

    train_loss = 0

    device = model.parameters().__next__().device
    with tqdm(
            total=len(train_loader),
            unit="Bt",
            unit_scale=True,
            unit_divisor=1000,
            leave=False,
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    ) as pbar:

        for batch_idx, (images, _) in enumerate(train_loader):

            if isinstance(images, list):
                images = images[0]

            images = images.to(device)

            optimizer.zero_grad()
            output = model(images)
            criterion = nn.MSELoss()

            loss = criterion(output, images)

            loss.backward()
            optimizer.step()
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                scheduler.step()

            train_loss += loss.item() * train_loader.batch_size
            nb_img_so_far = (batch_idx + 1) * train_loader.batch_size

            pbar.set_postfix(
                Train_Loss=train_loss / nb_img_so_far, refresh=True,
            )
            pbar.update(1)

    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
        scheduler.step()

    train_size = len(train_loader) * train_loader.batch_size

    return train_loss / train_size


def test(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                data = data[0]
                target = target[0]

            data, target = data.to(device), target.to(device)

            output = model(data)
            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    test_size = len(test_loader.dataset)

    return test_loss / test_size, test_correct / test_size


def test_autoencoder_unsupervised(model, test_loader):

    model.eval()

    test_loss = 0

    device = model.parameters().__next__().device

    for batch_idx, (images, _) in enumerate(test_loader):

        if isinstance(images, list):
            images = images[0]

        images = images.to(device)

        output = model(images)
        criterion = nn.MSELoss()
        loss = criterion(output, images)

        test_loss += loss.item() * test_loader.batch_size
        nb_img_so_far = (batch_idx + 1) * test_loader.batch_size

    test_size = len(test_loader) * test_loader.batch_size

    return test_loss / test_size
