import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, max_depth=256, lambd=0.5):
        super().__init__()
        self.lambd = lambd
        self.max_depth = max_depth

    def forward(self, pred, target):
        valid_mask = (target > 0.001).detach()
        valid_mask_upper = (target <= self.max_depth).detach()
        valid_mask = torch.logical_and(valid_mask, valid_mask_upper)
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss
