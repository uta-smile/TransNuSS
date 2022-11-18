import numpy as np
import os
from PIL import Image

PRED_DIR = os.path.join('predictions')
GT_DIR = os.path.join('data', 'zenodo', 'test_masks')
# OUTPUT_SIZE = (400, 400)


def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def compute_dice_score(y_true, y_pred):
    smooth = 1.

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def evaluate():
    iou_scores = []
    dice_scores = []

    gt_filenames = sorted(os.listdir(GT_DIR))

    for gt_filename in gt_filenames:
        gt_image = Image.open(os.path.join(GT_DIR, gt_filename)).convert('1')
        # gt_image = gt_image.resize(OUTPUT_SIZE, Image.NEAREST)
        gt = np.array(gt_image).astype('uint8')

        pred_filename = gt_filename #.replace('.png', '.tiff')
        pred = np.array(Image.open(os.path.join(PRED_DIR, pred_filename)).convert('1')).astype('uint8')

        iou_score = compute_iou(gt, pred)
        iou_scores.append(iou_score)

        dice_score = compute_dice_score(gt, pred)
        dice_scores.append(dice_score)

        print(gt_filename + ': iou ' + str(iou_score) + ', dice score ' + str(dice_score))

    print('iou: ' + str(np.mean(iou_scores)))
    print('dice score: ' + str(np.mean(dice_scores)))


if __name__ == '__main__':
    evaluate()
