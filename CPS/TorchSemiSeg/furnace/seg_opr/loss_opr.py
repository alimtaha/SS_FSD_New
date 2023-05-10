import sys

import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

from furnace.engine.logger import get_logger

logger = get_logger()


class FocalLoss2d(nn.Module):
    def __init__(
            self,
            gamma=0,
            weight=None,
            reduction='mean',
            ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        if weight:
            self.loss = nn.NLLLoss(
                weight=torch.from_numpy(
                    np.array(weight)).float(),
                reduction=reduction,
                ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(
                reduction=reduction,
                ignore_index=ignore_index)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 *
                         F.log_softmax(input, 1), target)


class FocalMSE(nn.Module):
    def __init__(self, gamma=2):
        super(FocalMSE, self).__init__()
        self.gamma = gamma

        self.loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss_no_reduction = self.loss(pred, target)
        weight = (1 - pred)**self.gamma
        weighted_loss = torch.mean(loss_no_reduction * weight)
        return weighted_loss


class RCELoss(nn.Module):
    def __init__(
            self,
            ignore_index=255,
            reduction='mean',
            weight=None,
            class_num=37,
            beta=0.01):
        super(RCELoss, self).__init__()
        self.beta = beta
        self.class_num = class_num
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(
            reduction=reduction,
            ignore_index=ignore_index,
            weight=weight)
        self.criterion2 = nn.NLLLoss(
            reduction='none',
            ignore_index=ignore_index,
            weight=weight)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        max_pred, max_id = torch.max(pred, dim=1)		# pred (b, h, w)
        target_flat = target.view(b, 1, h, w)
        mask = (target_flat.ne(self.ignore_label)).float()
        target_flat = (mask * target_flat.float()).long()
        # convert to onehot
        label_pred = torch.zeros(
            b, self.class_num, h, w).cuda().scatter_(
            1, target_flat, 1)
        # print(label_pred.shape, max_id.shape)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this

        weighted_pred = F.log_softmax(pred, dim=1)
        loss1 = self.criterion(weighted_pred, target)

        label_pred = torch.clamp(label_pred, min=1e-9, max=1.0 - 1e-9)

        label_pred = torch.log(label_pred)
        loss2 = self.criterion2(label_pred, max_id)
        loss2 = torch.mean(loss2 * mask)
        # print(loss1, loss2)
        loss = loss1 + self.beta * loss2
        # print(loss1, loss2)
        # print(loss)
        return loss


class BalanceLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None):
        super(BalanceLoss, self).__init__()
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(
            reduction=reduction,
            ignore_index=ignore_index,
            weight=weight)

    def forward(self, pred, target):
        # prob = torch.exp(pred)
        # # prob = F.softmax(prob, dim=1)      # i add this
        # weighted_pred = pred * (1 - prob) ** 2
        # loss = self.criterion(weighted_pred, target)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this
        weighted_pred = F.log_softmax(pred, dim=1) * (1 - prob) ** 2
        loss = self.criterion(weighted_pred, target)
        return loss


class berHuLoss(nn.Module):
    def __init__(self, delta=0.2, ignore_index=0, reduction='mean'):
        super(berHuLoss, self).__init__()
        self.delta = delta
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        valid_mask = (1 - target.eq(self.ignore_index)).float()
        valid_delta = torch.abs(pred - target) * valid_mask
        max_delta = torch.max(valid_delta)
        delta = self.delta * max_delta

        f_mask = (1 - torch.gt(target, delta)).float() * valid_mask
        s_mask = (1 - f_mask) * valid_mask
        f_delta = valid_delta * f_mask
        s_delta = ((valid_delta ** 2) + delta ** 2) / (2 * delta) * s_mask

        loss = torch.mean(f_delta + s_delta)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
            pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
            (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class BYOLContrastiveRegulizer(nn.Module):
    def __init__(self, ignore_label, alpha, beta, theta):
        super(BYOLContrastiveRegulizer, self).__init__()
        self.ignore_label = ignore_label
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, im, lowres, depth=None, masked=None):
        b, c, h, w = im.size()
        
        im = im.view(b, -1)
        
        lowres = lowres.view(b, -1)

        #mask = (target.ne(self.ignore_label)).float()
        #target = mask * target
        #onehot = target.view(b, -1, 1)
        
        #if self.reduction == 'mean':
        #    loss = loss.mean()
        im_norm = F.normalize(im, dim=-1, p=2)
        lowres_norm = F.normalize(lowres, dim=-1, p=2)
        
        im_lowres_masked = 2 - 2 * (im_norm * lowres_norm).sum(dim = -1)

        if depth is not None:
            #depth = depth.view(b, -1)

            #double check dimension
            depth_norm = F.normalize(depth, dim=-1, p=2)
            im_dep = 2 - 2 * (im_norm * depth_norm).sum(dim = -1)


            if masked is not None:
                masked = masked.view(b, -1)
                masked_norm = F.normalize(masked, dim=-1, p=2)
                im_masked = 2 - 2 * (im_norm * masked_norm).sum(dim = -1)
                loss = self.alpha * im_dep + self.beta * im_lowres_masked + self.theta * im_masked
            else:
                loss = self.alpha * im_dep + self.beta * im_lowres_masked

        
        else:
            if masked is not None:
                masked = masked.view(b, -1)
                masked_norm = F.normalize(masked, dim=-1, p=2)
                im_masked = 2 - 2 * (im_norm * masked_norm).sum(dim = -1)
                loss = self.beta * im_lowres_masked + self.theta * im_masked
            else:
                loss = self.beta * im_lowres_masked
    
        return loss.mean()


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(
            self,
            ignore_label,
            reduction='mean',
            thresh=0.6,
            min_kept=256,
            down_ratio=1,
            use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(  # weight is used to weigh the classes especially for unbalanced datsets.
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,  # CHANGE to make sure weighted sum of free space and non free space have an equal distribution within the dataset - CityScapes paper contains number of pixels per class, could also write script
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        # flattens tensor to one dimension -
        # https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view
        target = target.view(-1)
        # .ne is not equals to, https://pytorch.org/docs/stable/generated/torch.ne.html
        valid_mask = target.ne(self.ignore_label)
        # masking out invalid targets to get number of valids
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        # print('num_valid', num_valid)                              # summing
        # all valid targets (non label 255)

        #prob_bins = torch.argmax(pred, dim=1).cpu()
        # print(prob_bins)
        #prob_bins = torch.bincount(prob_bins.view(-1).long())
        prob = F.softmax(pred, dim=1)

        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            # the masked fill function changes value of the element where the
            # mask is true to the second input argument (here it's 1 in the
            # False indices since the ~ operator is used).
            # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                # The probability is less than the threshold value to be dug
                # out - 概率小于阈值的挖出来
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        # ~ is the bitwise negation operator, so flips the mask - fills the False indices with the ignore label - 255
        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


def bce2d(input, target):
    b, c, h, w = input.size()

    log_p = input.permute(0, 2, 3, 1).contiguous(
    ).view(-1)      # (b, h, w, c) ==> (b, -1)
    target = target.view(-1)

    pos_index = (target == 1)
    neg_index = (target == 0)
    ignore_index = (target > 1)

    weight = torch.zeros(log_p.size()).cuda().float()
    pos_num = pos_index.sum().float()
    neg_num = neg_index.sum().float()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight[ignore_index] = 0
    # print(weight.max(), pos_num, neg_num)

    loss = F.binary_cross_entropy_with_logits(
        log_p, target.float(), weight, reduction='mean')
    return loss
