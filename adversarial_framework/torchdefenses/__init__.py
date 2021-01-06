"""Adversarial defense module implemented on Pytorch"""

from ._adversarial_train import adversarial_epoch, adversarial_test


__all__ = ["adversarial_epoch", "adversarial_test"]
