from pycocotools import mask as cocomask
from fastai.conv_learner import *

def get_segmentations(labeled):
    nr_true = int(labeled.max())
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations

def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)

def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious

def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)

def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)

def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)

def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / len(iou)
        ious.append(iou_mean)
    return np.mean(ious)

def my_eval(pred,targ):
    pred = to_np(torch.sigmoid(pred))
    targ = to_np(targ)
    losses = []
    for i in range(targ.shape[0]):
        losses.append(compute_eval_metric(targ[i],((pred[i]>0.5).astype(np.uint8))))
    return np.mean(losses)

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs



