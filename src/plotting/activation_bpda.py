from ..utils.plot_settings import *
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

x = np.linspace(-1.1, 1.1, 2201)
jump = 8/255*3


def activation(x):
    return 0.5 * (np.sign(x - jump) + np.sign(x + jump))


def sech(x):
    return 1 / np.cosh(x)


def smooth_bpda(x, steepness):
    return 0.5 * (np.tanh((x-jump)*steepness) + np.tanh((x+jump)*steepness))


plt.figure(figsize=(10, 4))
plt.plot(x, activation(x), label='Activation', linewidth=2)
plt.plot(x, smooth_bpda(x, 2.0), label='steepness=2.0', linewidth=2)
plt.plot(x, smooth_bpda(x, 4.0), label='steepness=4.0', linewidth=2)
plt.plot(x, smooth_bpda(x, 8.0), label='steepness=8.0', linewidth=2)
plt.plot(x, smooth_bpda(x, 32.0), label='steepness=32.0', linewidth=2)
plt.xlim([-0.75, 0.75])
plt.ylim([-1.3, 1.3])
plt.xlabel(
    r"$\frac{\mathbf{\hat{x}}_{ij}(l)}{\left\|\mathbf{d}_{l}\right\|_{1}}$")
plt.ylabel(
    r"$\frac{\mathbf{\bar{x}}_{ij}(l)}{\left\|\mathbf{d}_{l}\right\|_{1}}$").set_rotation(0)
plt.legend()
plt.tight_layout()
plt.savefig(join('figs', 'bpda.pdf'))
