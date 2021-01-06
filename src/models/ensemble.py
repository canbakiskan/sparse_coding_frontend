import torch
import torch.nn.functional as F
from torch import nn


class Ensemble_post_softmax(nn.Module):
    def __init__(self, model, ensemble_E):
        super(Ensemble_post_softmax, self).__init__()
        self.model = model
        self.ensemble_E = ensemble_E

    def forward(self, x):
        out = torch.zeros(x.shape[0], 10, device=x.device)
        for i in range(self.ensemble_E):
            out = out + F.softmax(self.model(x), dim=1)

        return out / self.ensemble_E

    def get_softmax(self, x):
        softmax = torch.zeros(x.shape[0], self.ensemble_E, 10, device=x.device)
        for i in range(self.ensemble_E):
            softmax[:, i, :] = F.softmax(self.model(x), dim=1)

        return softmax
