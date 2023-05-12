# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


# voc cityscapes metric
def hist_info(n_cl, pred, gt):
    print(pred.shape, gt.shape)
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # return boolean array where all conditions matched are true and others as false? same size as gt - done to not take into account the ignore ondex as seen below with the labelled sum
    # k is a list of length 1024, where every element is a list of length
    # 2048, it is effectively as 1024 x 2048 array (in the case of whole image
    # eval)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    # print('gt[k] shape', gt[k].shape)  #one dimensional scalar
    # print('pred[k] shape', pred[k].shape)  #one dimensional scalar
    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),  # return bins/histogram with number of occurunces per label (where the label corresponds to the index of the returned histogram array)
                       # this constructs the confusion matrix in a weird way,
                       # effectively it's giving every element in the matrix an
                       # index, so 0,0 is 0, 0,1 is 1, 0,2 is 2,
                       minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), labeled, correct

    # multiplying gt by number of classes means the index get scaled by 3. Now what happens is they are scaled by 3 then added to the non scaled classes. So for example in the case of two classes multiplying the labels [0,1,1,1,0,0] by two before adding them means you will get 0, 1, 2, 3. In this case, a 3 is derived only if the gt label was 1 and the pred was also 1, similarly for all combinations there is also only one way to deerive them. This means effectively every index of a confusion matrix will correspond to one of these indices. A 0 label GT and 1 label pred is an index 1. We then do a bin count to
    # get the numbers of each occurence of each index, then reshape it into a
    # matrix (square form). np.reshape looks at the shape then fills in the
    # firts row first, so 0 and 1, then moves on under in a new line to fill 2
    # and 3. 3 is label-1 GT and label-1 Pred so it is at the end of the
    # diagonal, which is correct for a confusion matrix in this case.


def compute_score(hist, correct, labeled):
    # np.diag extracts all the diagonal elemennts, for a confusion matrix, the
    # diagonal elements would be the correctly labelled classes
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # in this histogram, the images on the columns are the predictions,
    # whereas the images on the rows are the ground truth
    # hist.sum(1) and hist.sum(0) give you false positives and negatives, it's
    # a confusion matrix (however you then need to subtract all the diagnial
    # pixels since those are true positives)
    mean_IU = np.nanmean(iu)
    # this is just the mean iou exlcuding the first class (road in this case)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


def compute_score_recall_precision(hist, correct, labeled):
    # np.diag extracts all the diagonal elemennts, for a confusion matrix, the
    # diagonal elements would be the correctly labelled classes
    r = np.diag(hist) / (hist.sum(1))
    p = np.diag(hist) / (hist.sum(0))
    # in this histogram, the images on the columns are the predictions,
    # whereas the images on the rows are the ground truth
    # hist.sum(1) and hist.sum(0) give you false positives and negatives, it's
    # a confusion matrix (however you then need to subtract all the diagnial
    # pixels since the true positives will be added twice in the denominatior
    # once with the .sum(1) and then again with the .sum(0), so we need to cancel one of them)
    mean_r = np.nanmean(r)
    mean_p = np.nanmean(p)
    # this is just the mean iou exlcuding the first class (road in this case)
    mean_r_no_back = np.nanmean(r[1:])
    mean_p_no_back = np.nanmean(p[1:])

    return p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back

def recall_and_precision_all(pred, gt, n_cl):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # return boolean array where all conditions matched are true and others as false? same size as gt - done to not take into account the ignore ondex as seen below with the labelled sum
    # k is a list of length 1024, where every element is a list of length
    # 2048, it is effectively as 1024 x 2048 array (in the case of whole image
    # eval)
    precision = [0] * n_cl
    recall = [0] * n_cl
    mean_prec = 0
    mean_recall = 0
    for i in range(n_cl):
        tp = np.sum((pred[k] == i) & (gt[k] == i))
        fp = np.sum((pred[k] == i) & (gt[k] != i))
        #tn = np.sum((pred[k] != i) & (gt[k] != i))
        fn = np.sum((pred[k] != i) & (gt[k] == i))
        # to avoid NaNs if class doesn't exist in picture or not classified
        recall[i] = tp / ((tp + fn) + 1e-10)
        precision[i] = tp / ((tp + fp) + 1e-10)
    mean_prec = np.nanmean(precision)
    mean_recall = np.nanmean(recall)
    return precision, mean_prec, recall, mean_recall

def recall_and_precision(pred, gt, n_cl):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # return boolean array where all conditions matched are true and others as false? same size as gt - done to not take into account the ignore ondex as seen below with the labelled sum
    # k is a list of length 1024, where every element is a list of length
    # 2048, it is effectively as 1024 x 2048 array (in the case of whole image
    # eval)
    precision = [0] * n_cl
    recall = [0] * n_cl
    mean_prec = 0
    mean_recall = 0
    for i in range(n_cl):
        tp = np.sum((pred[k] == i) & (gt[k] == i))
        fp = np.sum((pred[k] == i) & (gt[k] != i))
        #tn = np.sum((pred[k] != i) & (gt[k] != i))
        fn = np.sum((pred[k] != i) & (gt[k] == i))
        # to avoid NaNs if class doesn't exist in picture or not classified
        recall[i] = tp / ((tp + fn) + 1e-10)
        precision[i] = tp / ((tp + fp) + 1e-10)
    mean_prec = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)
    return precision, mean_prec, recall, mean_recall

# ade metric


def meanIoU(area_intersection, area_union):
    iou = 1.0 * np.sum(area_intersection, axis=1) / np.sum(area_union, axis=1)
    meaniou = np.nanmean(iou)
    meaniou_no_back = np.nanmean(iou[1:])

    return iou, meaniou, meaniou_no_back


def intersectionAndUnion(imPred, imLab, numClass):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass,
                                          range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


def mean_pixel_accuracy(pixel_correct, pixel_labeled):
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (
        np.spacing(1) + np.sum(pixel_labeled))

    return mean_pixel_accuracy


def pixelAccuracy(imPred, imLab):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return pixel_accuracy, pixel_correct, pixel_labeled
