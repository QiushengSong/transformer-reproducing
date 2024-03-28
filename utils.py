import torch
import torch.nn as nn


class LabelSmoothingKLDivLoss(nn.Module):
    def __init__(self,
                 size,
                 padding_idx,
                 smoothing: float = 0.0,
                 ):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, outputs, target):
        assert outputs.size(1) == self.size
        true_dist = outputs.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(outputs, true_dist)


class VariationalOptimizer:
    def __init__(self,
                 model_size,
                 factor,
                 warmup: int,
                 optimizer,
                 ):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
            self._rate = rate
            self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return VariationalOptimizer(model.src_embed[0].d_model,
                                2,
                                4000,
                                torch.optim.Adam(model.parameters(),
                                                 lr=0,
                                                 betas=(0.9, 0.98),
                                                 eps=1e-9))