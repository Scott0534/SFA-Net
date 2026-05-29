import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision, assd

def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def compute_smeasure(output_, target_, alpha=0.5):
    output = output_.astype(np.float32)
    target = target_.astype(np.float32)
    def compute_os(x, y):
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = ((x - x_mean) * (y - y_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum()
        var_y = ((y - y_mean) ** 2).sum()
        if var_x + var_y == 0:
            return 1.0
        return 2 * cov_xy / (var_x + var_y)
    def compute_rs(x, y):
        intersection = (x * y).sum()
        union = x.sum() + y.sum() - intersection
        if union == 0:
            return 1.0
        return intersection / union
    os = compute_os(output, target)
    rs = compute_rs(output, target)
    s = alpha * os + (1 - alpha)*rs
    return np.clip(s, 0, 1)
#
def compute_mae(output, target):
    pred = output.view(-1).cpu().numpy()
    gt = target.view(-1).cpu().numpy().astype(np.float32)
    return np.mean(np.abs(pred - gt))

import numpy as np

# def compute_smeasure(pred, gt, alpha=0.5):
#     """
#     Standard S-Measure (Structure Measure)
#     pred, gt: np.ndarray of shape (H, W), values in [0, 1]
#     """
#     pred = np.asarray(pred, dtype=np.float32)
#     gt = np.asarray(gt, dtype=np.float32)
#
#     # 1. Object-aware similarity
#     mu_pred = pred.mean()
#     mu_gt = gt.mean()
#     sigma_pred = ((pred - mu_pred) ** 2).sum()
#     sigma_gt = ((gt - mu_gt) ** 2).sum()
#     sigma_pred_gt = ((pred - mu_pred) * (gt - mu_gt)).sum()
#     So = 2 * sigma_pred_gt / (sigma_pred + sigma_gt + 1e-8)
#
#     # 2. Region-aware similarity
#     # Center mask (Gaussian)
#     h, w = gt.shape
#     y, x = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w), indexing='ij')
#     cx = (w - 1) / 2.0
#     cy = (h - 1) / 2.0
#     sigma_x = w / 4.0
#     sigma_y = h / 4.0
#     center_mask = np.exp(-((x - cx)**2 / (2 * sigma_x**2) + (y - cy)**2 / (2 * sigma_y**2)))
#
#     def compute_int(x, y):
#         return 2 * (x * y).sum() / (x.sum() + y.sum() + 1e-8)
#
#     gt_mean = gt.mean()
#     if gt_mean == 0 or gt_mean == 1:
#         Sr = 1.0
#     else:
#         pred_fg = pred * center_mask
#         gt_fg = gt * center_mask
#         pred_bg = (1 - pred) * (1 - center_mask)
#         gt_bg = (1 - gt) * (1 - center_mask)
#         int_fg = compute_int(pred_fg, gt_fg)
#         int_bg = compute_int(pred_bg, gt_bg)
#         Sr = 0.5 * int_fg + 0.5 * int_bg
#
#     S = alpha * So + (1 - alpha) * Sr
#     return np.clip(S, 0, 1)

# ==========================
# 完全修复好的 iou_score
# ==========================
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    try:
        asd_ = assd(output_, target_)
    except:
        asd_ = 0
    try:
        hd_ = hd(output_, target_)
    except:
        hd_ = 0

    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    tp = np.logical_and(output_, target_).sum()
    fp = np.logical_and(output_, np.logical_not(target_)).sum()
    fn = np.logical_and(np.logical_not(output_), target_).sum()
    tn = np.logical_and(np.logical_not(output_), np.logical_not(target_)).sum()

    acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    f1 = (2 * precision_ * recall_ + smooth) / (precision_ + recall_ + smooth)
    s_measure_ = compute_smeasure(output_.astype(np.float32), target_.astype(np.float32))
    mae_ = np.mean(np.abs(output - target))

    return iou_, dice_, hd95_, recall_, specificity_, precision_, acc, f1, asd_, hd_, s_measure_, mae_