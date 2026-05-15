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
            edge_kernel_size: int = 3,
            img_extension: str = '.jpg',
            mask_extension: str = '.jpg'
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

        # ===================== 核心修改：按文件夹路径加载 =====================
        # 自动拼接 train/val 文件夹路径
        self.img_dir = os.path.join(base_dir, split, "images")  # 训练/验证图片文件夹
        self.mask_dir = os.path.join(base_dir, split, "masks")  # 掩码文件夹

        # 自动读取文件夹内所有指定后缀的图片文件名（不带后缀）
        self.sample_list = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith(self.img_extension):
                name = os.path.splitext(filename)[0]  # 去掉后缀
                self.sample_list.append(name)
        # =====================================================================

        print(f"[{split.upper()}] Loaded {len(self.sample_list)} samples from {self.img_dir}")

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

        # ===================== 修改：直接从对应文件夹读取 =====================
        img_path = os.path.join(self.img_dir, case_name + self.img_extension)
        mask_path = os.path.join(self.mask_dir, case_name + self.mask_extension)
        # =====================================================================

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

        # 生成边缘
        edge_mask_2d = self._generate_edge_mask(mask_2d)

        # 转 tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0).float()
        edge_mask_tensor = torch.from_numpy(edge_mask_2d).unsqueeze(0).float()

        return {
            "image": image,
            "label": mask_tensor,
            "edge_mask": edge_mask_tensor,
            "case": case_name
        }