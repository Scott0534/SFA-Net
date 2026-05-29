import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir: str,
            split: str = "train",
            transform=None,
            train_file_dir: str = "train_0.txt",
            val_file_dir: str = "val_0.txt",
            edge_kernel_size: int = 3,
            img_extension: str = '.png',
            mask_extension: str = '.png'
    ):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.img_extension = img_extension
        self.mask_extension = mask_extension

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (edge_kernel_size, edge_kernel_size)
        )

        # 加载对应 fold 的 txt
        list_file = train_file_dir if split == "train" else val_file_dir
        list_path = os.path.join(base_dir, list_file)

        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        print(f"[{split.upper()}] Loaded {len(self.sample_list)} samples from {list_path}")

    def __len__(self):
        return len(self.sample_list)

    def _generate_edge_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        dilated = cv2.dilate(mask_u8, self.kernel, iterations=1)
        eroded = cv2.erode(mask_u8, self.kernel, iterations=1)
        edge = dilated - eroded
        return (edge > 127).astype(np.float32)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        case_name = os.path.splitext(case_name)[0]

        img_path = os.path.join(self.base_dir, 'images', case_name + self.img_extension)
        mask_path = os.path.join(self.base_dir, 'masks', "0",case_name + self.mask_extension)

        # ===================== 修复读取图片 =====================
        # 读取图像（解决空格/括号无法读取）
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            image = None
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码（解决空格/括号无法读取）
        try:
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except:
            mask = None
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        # ======================================================

        # def __getitem__(self, idx):
        #     # 1. 从txt读取完整文件名（如 bus_0001-l.png）
        #     full_img_name = self.sample_list[idx]
        #     case_name = os.path.splitext(full_img_name)[0]  # 得到 bus_0001-l
        #
        #     # ===================== 核心修复：路径拼接 =====================
        #     # 原图路径：base_dir/images/bus_0001-l.png
        #     img_path = os.path.join(self.base_dir, 'images', full_img_name)
        #
        #     # 掩码路径：base_dir/masks/mask_0001-l.png
        #     # 从 bus_0001-l 提取 0001-l，拼接 mask_ 前缀
        #     suffix = case_name.split('_', 1)[1]  # 从 bus_0001-l 得到 0001-l
        #     mask_name = f"mask_{suffix}{self.mask_extension}"
        #     mask_path = os.path.join(self.base_dir, 'masks', mask_name)
        #     # ==============================================================
        #
        #     # 读取图像（兼容中文/特殊字符路径）
        #     try:
        #         image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        #     except Exception as e:
        #         raise FileNotFoundError(f"Image not found: {img_path}, Error: {e}")
        #     if image is None:
        #         raise FileNotFoundError(f"Image is None: {img_path}")
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        #     # 读取掩码（兼容中文/特殊字符路径）
        #     try:
        #         mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        #     except Exception as e:
        #         raise FileNotFoundError(f"Mask not found: {mask_path}, Error: {e}")
        #     if mask is None:
        #         raise FileNotFoundError(f"Mask is None: {mask_path}")

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 统一转为 numpy 处理
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # 归一化到 0~1
        image = image.astype(np.float32)
        mask = mask.astype(np.float32) / 255.0

        # 二值化
        mask_2d = (mask > 0.5).astype(np.float32)

        # 生成边缘（必须打开！你的loss要用！）
        edge_mask_2d = self._generate_edge_mask(mask_2d)

        # 转 tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0).float()
        edge_mask_tensor = torch.from_numpy(edge_mask_2d).unsqueeze(0).float()

        return {
            "image": image,
            "label": mask_tensor,
            "edge_mask": edge_mask_tensor,  # 必须打开
            "case": case_name  # 必须打开
        }