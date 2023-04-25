import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
test_var = True


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            '''
            if test_var:
                print("Input & Size", input.shape, input)
                print("Target & Size", target.shape, target)
                print(target.shape[-2:])        #second last dimension?
            '''
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            '''
            if test_var:
                print("New Masked Input & Size", input.shape, input)
                test_var = False
            '''

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(
        nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor(
            [len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(
            target_points,
            batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(
            x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
