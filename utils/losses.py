import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from typing import List, Dict, Tuple
import numpy as np
# # 修正2：设置edt别名，解决函数未定义问题
# from scipy.ndimage import distance_transform_edt as edt
#
__all__ = [ 'EdgeStructureCombinedLoss',"BCEDiceLoss"]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== SSIM 损失模块（已修复bug）=====================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 移除硬编码 cuda()，自动跟随设备
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 修复 padding 必须为整数
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12  = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        device = img1.device
        # 窗口自动切换设备
        if channel == self.channel and self.window.device == device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(device)
            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)
###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)





class EdgeStructureCombinedLoss(nn.Module):

    def __init__(self, loss_weights: List[float] = None):
        super().__init__()

        self.BCEDiceLoss=structure_loss()

    def forward(
            self,
            model_outputs: List[torch.Tensor],  # 所有输出均为logits
            mask: torch.Tensor,  # 原始分割掩码

    ) -> Tuple[torch.Tensor, Dict[str, float]]:


        # 解包输出：均为logits
        aux1_pred_logits, aux2_pred_logits, aux3_pred_logits, final_pred_logits = model_outputs
        aux1_loss = self.BCEDiceLoss(aux1_pred_logits, mask)
        aux2_loss = self.BCEDiceLoss(aux2_pred_logits, mask)
        aux3_loss = self.BCEDiceLoss(aux3_pred_logits, mask)


        final_loss = self.BCEDiceLoss(final_pred_logits, mask)

        # 3. 总损失（加权和）
        total_loss = (

                1* aux1_loss +
                1 * aux2_loss +
                1* aux3_loss +

                1* final_loss
        )

        # 4. 损失详情
        # loss_dict = {
        #     "total_loss": total_loss.item(),
        #
        #     "aux1_loss": aux1_loss.item(),
        #     "aux2_loss": aux2_loss.item(),
        #     "aux3_loss": aux3_loss.item(),
        #
        #     "final_loss": final_loss.item(),
        # }

        return total_loss




