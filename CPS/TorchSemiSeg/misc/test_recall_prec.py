import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def recall_and_precision(pred, gt, n_cl):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # return boolean array where all conditions matched are true and others as false? same size as gt - done to not take into account the ignore ondex as seen below with the labelled sum
    # k is a list of length 1024, where every element is a list of length
    # 2048, it is effectively as 1024 x 2048 array (in the case of whole image
    # eval)
    precision = []
    recall = []
    print(len(recall))
    mean_prec = None
    mean_recall = None
    for i in range(n_cl):
        p_pred = (pred[k] == i)
        print('p_pred', p_pred)
        p_gt = (gt[k] == i)
        print('p_gt', p_gt)
        n_pred = (pred[k] != i)
        print('n_pred', n_pred)
        n_gt = (gt[k] != i)
        print('n_gt', n_gt)
        tp = np.sum(p_pred & p_gt)
        tp_test = np.sum(((pred[k] == i) & (gt[k] == i)))
        tn = np.sum(n_pred & n_gt)
        fn = np.sum(n_pred & p_gt)
        fp = np.sum(p_pred & n_gt)
        fp_test = np.sum(((pred[k] == i) & (gt[k] != i)))
        tn_test = np.sum(((pred[k] != i) & (gt[k] != i)))
        fn_test = np.sum(((pred[k] != i) & (gt[k] == i)))
        recall.append(tp / (tp + fn))
        precision.append(tp / (tp + fp))
        print('Class', i)

        print(tp / (tp + fp))
        print('precision', precision)
        print('recall', recall)

        print('\ntrue positive', tp)
        print('true positive test', tp_test)

        print('\nfalse positive', fp)
        print('false positive test', fp_test)

        print('\ntrue negative', tn)
        print('true negative test', tn_test)

        print('\nfalse negative', fn)
        print('false negative test', fn_test, '\n\n\n')

    mean_prec = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)

    print(mean_prec, mean_recall)
    return precision, mean_prec, recall, mean_recall


im = np.random.randint(0, 2, (1024, 2048))
pred = np.zeros_like(im)
pred[0, :] = 1

recall_and_precision(pred, im, 2)

plt.imshow(im)
plt.show()
