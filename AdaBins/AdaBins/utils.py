import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
import torch.nn
from PIL import Image
# import scipy as sp     #the 'edges' function has also been commented out


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)  # UNCOMMENT
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_errors(gt, pred):  # also defined in evaluate.py

    mask_20 = gt < 20
    mask_80 = gt < 80
    mask_20_80 = np.logical_and(gt > 20, gt < 80)
    mask_all = gt > 0.001
    mask_80_256 = gt > 80
    masks = [mask_20, mask_80, mask_20_80, mask_80_256, mask_all]
    mask_names = [
        'mask_20',
        'mask_80',
        'mask_20_80',
        'mask_80_256',
        'mask_all']

    all_results = {}

    for i, mask in enumerate(masks):

        if mask.sum() < 1:
            continue

        thresh = np.maximum((gt[mask] / pred[mask]), (pred[mask] / gt[mask]))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt[mask] - pred[mask]) / gt[mask])
        sq_rel = np.mean(((gt[mask] - pred[mask]) ** 2) / gt[mask])

        rmse = (gt[mask] - pred[mask]) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt[mask]) - np.log(pred[mask])) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred[mask]) - np.log(gt[mask])
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log10 = (np.abs(np.log10(gt[mask]) - np.log10(pred[mask]))).mean()

        all_results[mask_names[i]] = dict(d1=d1,
                                          d2=d2,
                                          d3=d3,
                                          abs_rel=abs_rel,
                                          rmse=rmse,
                                          log10=log10,
                                          rmse_log=rmse_log,
                                          silog=silog,
                                          sq_rel=sq_rel,
                                          iter = 1)

    return all_results


##################################### Demo Utilities #####################
def b64_to_pil(b64string):
    image_data = re.sub('^data:image/.+;base64,', '', b64string)
    # image = Image.open(cStringIO.StringIO(image_data))
    return Image.open(BytesIO(base64.b64decode(image_data)))


# Compute edge magnitudes
# from scipy import ndimage #UNCOMMENT

'''
def edges(d):
    dx = sp.ndimage.sobel(d, 0)  # horizontal derivative #UNCOMMENT
    dy = sp.ndimage.sobel(d, 1)  # vertical derivative #UNCOMMENT
    return np.abs(dx) + np.abs(dy)
'''


class PointCloudHelper():
    def __init__(self, width=640, height=480):
        self.xx, self.yy = self.worldCoords(width, height)

    def worldCoords(self, width=640, height=480):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width / 2, height / 2
        fx = width / (2 * math.tan(hFov / 2))
        fy = height / (2 * math.tan(vFov / 2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def depth_to_points(self, depth):
        # depth[edges(depth) > 0.3] = np.nan  # Hide depth edges  - !!!NEED TO
        # UNCOMMENT
        length = depth.shape[0] * depth.shape[1]
        # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)

        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))

##########################################################################
