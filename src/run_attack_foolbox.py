from tqdm import tqdm
from deepillusion.torchdefenses import adversarial_test
from .utils.read_datasets import read_dataset

from .models.combined import Combined
import torch
from .utils.get_modules import (
    load_classifier,
    load_frontend,
)
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD
from torchattacks import PGDL2

from .utils.device import determine_device

from .parameters import get_arguments


def main():

    args = get_arguments()

    device = determine_device(args)

    classifier = load_classifier(args)

    if args.neural_net.no_frontend:
        model = classifier

    else:
        frontend = load_frontend(args)
        model = Combined(frontend, classifier)

    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if not args.adv_testing.skip_clean:
        _, test_loader = read_dataset(args)
        test_loss, test_acc = adversarial_test(model, test_loader)
        print(f"Clean \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    fmodel = PyTorchModel(model, bounds=(0, 1))

    clean_total = 0
    adv_total = 0

    for batch_idx, items in enumerate(
        pbar := tqdm(test_loader, desc="Attack progress", leave=False)
    ):
        if args.adv_testing.nb_imgs > 0 and args.adv_testing.nb_imgs < (batch_idx + 1) * args.neural_net.test_batch_size:
            break

        data, target = items
        data = data.to(device)
        target = target.to(device)

        clean_acc = accuracy(fmodel, data, target)
        clean_total += int(clean_acc*args.neural_net.test_batch_size)

        epsilons = [args.adv_testing.budget]

        # apply the attack
        if args.adv_testing.method == "PGD":
            if args.adv_testing.norm == "inf":
                attack = LinfPGD(
                    abs_stepsize=args.adv_testing.step_size, steps=args.adv_testing.nb_steps)

            if args.adv_testing.norm == 2:
                attack = L2PGD(abs_stepsize=args.adv_testing.step_size,
                               steps=args.adv_testing.nb_steps)

                # TORCHATTACKS LIBRARY
                # attack = PGDL2(model, eps=args.adv_testing.budget, alpha=args.adv_testing.step_size,
                #                steps=args.adv_testing.nb_steps, random_start=False, eps_for_division=1e-10)

                # clipped_advs = attack(data, target)

        raw_advs, clipped_advs, success = attack(
            fmodel, data, target, epsilons=epsilons)

        with torch.no_grad():
            attack_out = model(clipped_advs)
            pred_attack = attack_out.argmax(dim=1, keepdim=True)

            attack_correct = pred_attack.eq(
                target.view_as(pred_attack)).sum().item()

            adv_total += int(attack_correct)
            adv_acc_sofar = adv_total / \
                ((batch_idx+1) * args.neural_net.test_batch_size)

        pbar.set_postfix(
            Adv_ac=f"{adv_acc_sofar:.4f}", refresh=True,
        )

    print(f"clean accuracy:  {clean_total / 10000 * 100:.2f} %")
    print(f"adv accuracy:  {adv_total / 10000 * 100:.2f} %")


if __name__ == "__main__":
    main()
