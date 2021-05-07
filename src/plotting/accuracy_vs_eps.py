from ..utils.plot_settings import *
import matplotlib.pyplot as plt
from os.path import join

plt.figure(figsize=(6, 3))
plt.plot([0, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128], [81.81,
                                                      61.63, 44.79, 31.21, 23.74, 18.2, 13.73, 10.73, 8.19, 6.25, 1.28], 'o-', markersize=5)
plt.ylim([0, 90])
plt.xlabel(r"$\epsilon \times 255$")
plt.ylabel("Adversarial accuracy")
plt.tight_layout()
plt.savefig(join('figs', 'acc_vs_eps.pdf'))
