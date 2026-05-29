import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class structure_loss(torch.nn.Module):
    def __init__(self, boundary_weight=0.3):
        super().__init__()
        self.boundary_weight = boundary_weight

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - inter / (union - inter + 1e-8)
        return (wbce + wiou).mean()

    # ====================== 这里彻底修复 ======================
    def _distance_boundary_loss(self, pred, mask):
        """
        正确的边界损失：使用边缘检测生成边界权重图，无维度错误
        """
        pred = torch.sigmoid(pred)

        # 1. 拉普拉斯算子提取边界（正确、稳定、通用）
        kernel = torch.tensor([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)

        # 2. 计算 mask 的边缘
        edge_mask = torch.abs(F.conv2d(mask, kernel, padding=1))
        edge_mask = (edge_mask > 0).float()  # 二值化边缘

        # 3. 只在边缘区域计算 BCE（弱边缘增强）
        boundary_bce = F.binary_cross_entropy(pred, mask, reduction='none')
        bound_loss = (boundary_bce * edge_mask).mean()

        return bound_loss

    # =========================================================

    def forward(self, pred, mask):
        loss_struct = self._structure_loss(pred, mask)
        loss_bound = self._distance_boundary_loss(pred, mask)
        total_loss = loss_struct + self.boundary_weight * loss_bound
        return total_loss


class EdgeStructureCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCEDiceLoss = structure_loss()

    def forward(
            self,
            model_outputs: List[torch.Tensor],
            mask: torch.Tensor,
    ) -> torch.Tensor:
        aux1_pred_logits, aux2_pred_logits, aux3_pred_logits, final_pred_logits = model_outputs

        aux1_loss = self.BCEDiceLoss(aux1_pred_logits, mask)
        aux2_loss = self.BCEDiceLoss(aux2_pred_logits, mask)
        aux3_loss = self.BCEDiceLoss(aux3_pred_logits, mask)
        final_loss = self.BCEDiceLoss(final_pred_logits, mask)

        total_loss = aux1_loss + aux2_loss + aux3_loss + final_loss
        return total_loss