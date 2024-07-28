import torch.nn as nn
from dgl.nn.pytorch import AvgPooling, SumPooling

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.ave_pool = AvgPooling()

    def forward(self, g, predictions, targets):
        losses = self.ave_pool(g, (predictions - targets) ** 2)
        loss = losses.mean()
        return loss

class RelL2Loss(nn.Module):
    def __init__(self):
        super(RelL2Loss, self).__init__()
        self.sum_pool = SumPooling()

    def forward(self, g, predictions, targets):
        losses = self.sum_pool(g, (predictions - targets) ** 2)
        targets_norm = self.sum_pool(g, targets ** 2)
        loss = ((losses / targets_norm) ** 0.5).mean()
        return loss