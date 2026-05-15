import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from lib.van32 import Network

# ===================== 配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1
WEIGHT_PATH = "/home/wc/AI/zyb/newus/消融result/消融实验1/van31相加融合/Network_best_20260425_1851.pth"
INPUT_FOLDER = "/home/wc/AI/zyb/newus/data/busi/images"
OUTPUT_FOLDER = "./pred_results/o4busi_heatmap_overlay相加"
INPUT_SIZE = (256, 256)
SAVE_SIZE = (256, 256)
ALPHA = 0.5    # 热力图透明度，0~1
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

def gen_overlay_heatmap(orig_img_np, prob_map, alpha=0.5):
    # 概率图转伪彩色热力图
    prob_norm = (prob_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
    # 原图与热力图叠加
    overlay = cv2.addWeighted(orig_img_np, 1-alpha, heatmap, alpha, 0)
    return overlay

@torch.no_grad()
def predict_prob_map(model, img_tensor):
    x = img_tensor.to(DEVICE)
    o1, o2, o3, o4 = model(x)

    # 选择输出层
    pred = o4
    # pred = o3
    # pred = o2
    # pred = o1

    pred = torch.sigmoid(pred)
    pred = F.interpolate(pred, size=SAVE_SIZE)
    prob_map = pred.squeeze().cpu().numpy()
    return prob_map

def batch_infer():
    model = build_model(WEIGHT_PATH)
    img_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('png','jpg','jpeg'))]

    for name in tqdm(img_files):
        img_path = os.path.join(INPUT_FOLDER, name)
        # 预处理送入网络
        tensor = preprocess(img_path)
        prob_map = predict_prob_map(model, tensor)

        # 读取原图并resize到和热力图一样大小
        orig_img = Image.open(img_path).convert("RGB").resize(SAVE_SIZE)
        orig_img_np = np.array(orig_img)
        orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)

        # 生成叠加图
        overlay_img = gen_overlay_heatmap(orig_img_np, prob_map, alpha=ALPHA)

        # 保存
        save_name = os.path.splitext(name)[0] + "_overlay.png"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        cv2.imwrite(save_path, overlay_img)

    print("✅ 原图+热力图叠加完成！保存至：", OUTPUT_FOLDER)

if __name__ == "__main__":
    batch_infer()