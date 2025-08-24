import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

from tools.visualization import overlay_anomaly
from tools.metrics import compute_iou

def test_model(extractor, flow_model, test_dir, gt_dir, device, threshold=0.05):
    # لیست تصاویر تست و ground truth
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    gt_masks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    # تبدیل‌ها
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    extractor.eval()
    flow_model.eval()

    for i in range(min(5, len(test_images))):
        # بارگذاری تصویر و ماسک
        img = Image.open(test_images[i]).convert("RGB")
        gt = Image.open(gt_masks[i]).convert("L")

        img_tensor = transform(img).unsqueeze(0).to(device)
        gt_tensor = transforms.ToTensor()(transforms.Resize((256, 256))(gt)).squeeze(0).cpu().numpy()
        # gt_tensor = transforms.ToTensor()(gt_tensor).squeeze(0).cpu().numpy()

        with torch.no_grad():
            features = extractor(img_tensor)  # [B, C, H, W]
            log_prob, _ = flow_model(features)  # [B, H, W]
            anomaly_map = -log_prob.squeeze(0).cpu().numpy()  # [H, W]

        # نرمال‌سازی anomaly map
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        # resize anomaly map به اندازه‌ی تصویر اصلی
        anomaly_tensor = torch.tensor(anomaly_map).unsqueeze(0).unsqueeze(0)
        resized_map = F.interpolate(anomaly_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        resized_map = resized_map.squeeze().numpy()

        # ساخت ماسک باینری
        binary_mask = (resized_map > threshold).astype(np.uint8)

        # ساخت تصویر overlay شده
        overlay_img = overlay_anomaly(img, binary_mask)

        # محاسبه IoU
        iou = compute_iou(resized_map, gt_tensor, threshold)

        # مصورسازی نهایی
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(gt_tensor, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        sns.heatmap(resized_map, ax=axs[2], cmap="jet", cbar=False)
        axs[2].set_title("Anomaly Map")
        axs[2].axis("off")

        axs[3].imshow(overlay_img)
        axs[3].set_title(f"Overlay\nIoU: {iou:.4f}")
        axs[3].axis("off")

        plt.tight_layout()
        plt.show()
