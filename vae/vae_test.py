import os
import glob
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

from tools.visualization import overlay_anomaly, show_reconstruction, show_anomaly_map
from tools.metrics import compute_iou

def test_model(model, test_dir, gt_dir, device, threshold=0.05):
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    gt_masks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    model.eval()
    for i in range(5):  # نمایش فقط ۵ تصویر اول
        img = Image.open(test_images[i]).convert("RGB")
        gt = Image.open(gt_masks[i]).convert("L")

        img_tensor = transform(img).unsqueeze(0).to(device)
        gt_tensor = transform(gt).squeeze(0).cpu().numpy()

        with torch.no_grad():
            reconstructed, mu, logvar = model(img_tensor)
            anomaly_map = torch.abs(img_tensor - reconstructed).mean(dim=1).squeeze(0).cpu().numpy()

        # نرمال‌سازی anomaly map
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        # ساخت ماسک باینری با آستانه
        binary_map = (anomaly_map > threshold).astype(np.uint8)

        # ساخت تصویر overlay شده
        overlay_img = overlay_anomaly(img, binary_map)

        # محاسبه IoU
        iou = compute_iou(anomaly_map, gt_tensor, threshold)

        # مصورسازی کامل
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(gt_tensor, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axs[2].set_title("Reconstructed Image")
        axs[2].axis("off")

        sns.heatmap(anomaly_map, ax=axs[3], cmap="jet", cbar=False)
        axs[3].set_title("Anomaly Map")
        axs[3].axis("off")

        axs[4].imshow(overlay_img)
        axs[4].set_title(f"Overlay\nIoU: {iou:.4f}")
        axs[4].axis("off")

        plt.tight_layout()
        plt.show()
