




# # import os
# # import cv2
# # import torch
# # import numpy as np
# # from PIL import Image
# # from torchvision import transforms
# # from tqdm import tqdm
# # import torch.nn.functional as F
# #
# # # 导入你的网络
# # from lib.van3 import Network
# #
# # # ===================== 【配置 全部改这里】 =====================
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # WEIGHT_PATH = "/home/wc/AI/zyb/newus/4.19result/16加xmax和avg改融合van4pvtb2结合傅里叶lh + out, hl + out, hh + out/Network_best_20260424_1121.pth"
# #
# # IMAGE_FOLDER = "/home/wc/AI/zyb/newus/data/busi/images"
# # LABEL_FOLDER = "/home/wc/AI/zyb/newus/data/busi/masks/0"
# # SAVE_FOLDER = "./pred_results/01vis_results"
# #
# # INPUT_SIZE = (256, 256)
# # THRESHOLD = 0.5
# # PREDICT_LAYER = "o1"
# # # ==============================================================
# #
# # os.makedirs(SAVE_FOLDER, exist_ok=True)
# #
# # def preprocess(image_path):
# #     img = Image.open(image_path).convert("RGB")
# #     transform = transforms.Compose([
# #         transforms.Resize(INPUT_SIZE),
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.485,0.456,0.406],[0.226,0.224,0.225])
# #     ])
# #     return transform(img).unsqueeze(0)
# #
# # def build_model():
# #     model = Network(in_channels=3, num_classes=1, pretrained_path=None)
# #     ckpt = torch.load(WEIGHT_PATH, map_location=DEVICE)
# #     state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# #     model.load_state_dict(state_dict, strict=False)
# #     model.to(DEVICE).eval()
# #     return model
# #
# # def get_contour(mask):
# #     mask = (mask > 127).astype(np.uint8)
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     return contours
# #
# # @torch.no_grad()
# # def predict_and_draw_on_mask(model, img_path, label_path):
# #     tensor = preprocess(img_path).to(DEVICE)
# #     o1, o2, o3, o4 = model(tensor)
# #
# #     # 选择单一层预测
# #     if PREDICT_LAYER == "o1":
# #         pred = o1
# #     elif PREDICT_LAYER == "o2":
# #         pred = o2
# #     elif PREDICT_LAYER == "o3":
# #         pred = o3
# #     else:
# #         pred = o4
# #
# #     # 二值化预测图
# #     pred = torch.sigmoid(pred)
# #     pred = F.interpolate(pred, size=INPUT_SIZE)
# #     pred = pred.squeeze().cpu().numpy()
# #     pred_mask = (pred > THRESHOLD).astype(np.uint8) * 255
# #
# #     # 转为3通道彩色图（才能画红色轮廓）
# #     vis_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
# #
# #     # 读取标签
# #     label = np.array(Image.open(label_path).convert("L").resize(INPUT_SIZE))
# #     label_mask = (label > 127).astype(np.uint8) * 255
# #
# #     # 在【二值预测图】上画【红色标签轮廓】
# #     cv2.drawContours(vis_mask, get_contour(label_mask), -1, (0, 0, 255), 2)
# #
# #     return vis_mask
# #
# # if __name__ == "__main__":
# #     model = build_model()
# #     img_names = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png','jpg','jpeg'))]
# #
# #     for name in tqdm(img_names):
# #         img_path = os.path.join(IMAGE_FOLDER, name)
# #         label_path = os.path.join(LABEL_FOLDER, name)
# #
# #         if not os.path.exists(label_path):
# #             continue
# #
# #         # 在预测mask上画轮廓
# #         result = predict_and_draw_on_mask(model, img_path, label_path)
# #         save_path = os.path.join(SAVE_FOLDER, name)
# #         cv2.imwrite(save_path, result)
# #
# #     print("✅ 完成！输出：二值预测图 + 红色标签轮廓")
# #
# # # 批量运行
# # if __name__ == "__main__":
# #     model = build_model()
# #     img_names = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
# #
# #     for name in tqdm(img_names):
# #         img_path = os.path.join(IMAGE_FOLDER, name)
# #         label_path = os.path.join(LABEL_FOLDER, name)  # 标签和原图同名
# #
# #         if not os.path.exists(label_path):
# #             continue
# #
# #         vis = predict_and_draw(model, img_path, label_path)
# #         save_path = os.path.join(SAVE_FOLDER, name)
# #         cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
# #
# #     print("✅ 全部叠加完成！保存到：", SAVE_FOLDER)
#
#
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from lib.van3 import Network

# ===================== 配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1
WEIGHT_PATH = "/home/wc/AI/zyb/newus/10轮权重result/10轮权重/Network_best_20260513_1527.pth"
INPUT_FOLDER = "/home/wc/AI/zyb/newus/data/tn3k/train/images"
OUTPUT_FOLDER = "./10轮权重pred_results/o4busibest"
INPUT_SIZE = (256, 256)
SAVE_SIZE = (256, 256)
THRESHOLD = 0.5
# ==================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def build_model(weight_path):
    model = Network(in_channels=3, num_classes=1, pretrained_path=None)
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

# ===================== 【关键：只取某一层预测】 =====================
@torch.no_grad()
def predict_by_single_layer(model, img_tensor):
    x = img_tensor.to(DEVICE)
    o1, o2, o3, o4 = model(x)  # 网络输出4层

    # ===================== 在这里选择你要的层 =====================
    pred = o4   # 只用第4层
    # pred = o3   # 只用第3层
    # pred = o2   # 只用第2层
    # pred = o1   # 只用第1层

    # 后处理不变
    pred = torch.sigmoid(pred)
    pred = F.interpolate(pred, size=SAVE_SIZE)
    pred = pred.squeeze().cpu().numpy()
    bin_mask = (pred > THRESHOLD).astype(np.uint8) * 255
    return bin_mask
# ====================================================================

def batch_infer():
    model = build_model(WEIGHT_PATH)
    img_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('png','jpg','jpeg'))]

    for name in tqdm(img_files):
        path = os.path.join(INPUT_FOLDER, name)
        tensor = preprocess(path)
        mask = predict_by_single_layer(model, tensor)
        save_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(name)[0] + "_mask.png")
        Image.fromarray(mask).save(save_path)

    print("✅ 只使用单一层预测完成！")

if __name__ == "__main__":
    batch_infer()
# #
# #
# # # import os
# # # import torch
# # # import numpy as np
# # # from PIL import Image
# # # from torchvision import transforms
# # # from tqdm import tqdm
# # # import torch.nn.functional as F
# # #
# # # # 导入你的网络
# # # from lib.van3 import Network
# # #
# # # # ===================== 配置（只改这里） =====================
# # # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # # NUM_CLASSES = 1
# # # WEIGHT_PATH = "/home/wc/AI/zyb/newus/4.19result/16加xmax和avg改融合van4pvtb2结合傅里叶lh + out, hl + out, hh + out/Network_best_20260424_1121.pth"
# # # INPUT_FOLDER = "/home/wc/AI/zyb/newus/data/busi/images"
# # # OUTPUT_FOLDER = "./pred_results/busi"
# # # FEATURE_SAVE_FOLDER = "./features"
# # # INPUT_SIZE = (256, 256)
# # # SAVE_SIZE = (256, 256)
# # # THRESHOLD = 0.5
# # #
# # # # ========== 选择你要提取哪一层！直接改这里 ==========
# # # # 可选：f1 / f2 / f3 / f4
# # # SAVE_FEATURE = "f4"  # <--- 想保存哪层就写哪层！
# # # # ======================================================
# # #
# # # # 创建文件夹
# # # os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# # # os.makedirs(FEATURE_SAVE_FOLDER, exist_ok=True)
# # #
# # # # 预处理
# # # def preprocess(image_path, size=(256, 256)):
# # #     img = Image.open(image_path).convert("RGB")
# # #     transform = transforms.Compose([
# # #         transforms.Resize(size),
# # #         transforms.ToTensor(),
# # #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # #                              std=[0.229, 0.224, 0.225])
# # #     ])
# # #     img_tensor = transform(img).unsqueeze(0)
# # #     return img_tensor
# # #
# # # # 模型加载
# # # def build_model(weight_path):
# # #     model = Network(in_channels=3, num_classes=NUM_CLASSES, pretrained_path=None)
# # #     checkpoint = torch.load(weight_path, map_location=DEVICE)
# # #     state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
# # #     model.load_state_dict(state_dict, strict=False)
# # #     model.to(DEVICE).eval()
# # #     print("✅ 模型加载成功")
# # #     return model
# # #
# # # # 特征图处理为256
# # # def process_feature_map(feat):
# # #     feat = feat[0, 0:1]
# # #     feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
# # #     feat = F.interpolate(feat.unsqueeze(0), size=SAVE_SIZE, mode='bilinear', align_corners=False)
# # #     return (feat.squeeze().cpu().numpy() * 255).astype(np.uint8)
# # #
# # # # 预测 + 只提取单一层
# # # @torch.no_grad()
# # # def predict_single_feature(model, img_tensor):
# # #     x = img_tensor.to(DEVICE)
# # #     f1, f2, f3, f4 = model.backbone.forward_features(x)  # 四层都拿，但只保存一个
# # #
# # #     # 预测输出
# # #     o1, o2, o3, o4 = model(x)
# # #     pred = (o1 + o2 + o3 + o4) / 4
# # #     pred = torch.sigmoid(pred)
# # #     pred = F.interpolate(pred, size=SAVE_SIZE)
# # #     pred = pred.squeeze().cpu().numpy()
# # #     bin_mask = (pred > THRESHOLD).astype(np.uint8) * 255
# # #
# # #     # 只返回你选中的层
# # #     if SAVE_FEATURE == "f1":
# # #         feat = process_feature_map(f1)
# # #     elif SAVE_FEATURE == "f2":
# # #         feat = process_feature_map(f2)
# # #     elif SAVE_FEATURE == "f3":
# # #         feat = process_feature_map(f3)
# # #     elif SAVE_FEATURE == "f4":
# # #         feat = process_feature_map(f4)
# # #     else:
# # #         feat = None
# # #
# # #     return bin_mask, feat
# # #
# # # # 批量运行
# # # def batch_predict_folder():
# # #     img_formats = ('.jpg', '.jpeg', '.png', '.bmp')
# # #     img_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(img_formats)]
# # #     model = build_model(WEIGHT_PATH)
# # #
# # #     for img_name in tqdm(img_files, desc="批量预测中"):
# # #         img_path = os.path.join(INPUT_FOLDER, img_name)
# # #         base = os.path.splitext(img_name)[0]
# # #         tensor = preprocess(img_path)
# # #         bin_mask, feat = predict_single_feature(model, tensor)
# # #
# # #         # 保存二值图
# # #         Image.fromarray(bin_mask).save(os.path.join(OUTPUT_FOLDER, f"{base}_mask.png"))
# # #         # 保存指定层特征图
# # #         Image.fromarray(feat).save(os.path.join(FEATURE_SAVE_FOLDER, f"{base}_{SAVE_FEATURE}.png"))
# # #
# # #     print(f"\n🎉 全部完成！只保存了：{SAVE_FEATURE} 层")
# # #
# # # if __name__ == "__main__":
# # #     batch_predict_folder()