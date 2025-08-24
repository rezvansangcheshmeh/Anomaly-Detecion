import numpy as np

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    return intersection / union if union != 0 else 0
