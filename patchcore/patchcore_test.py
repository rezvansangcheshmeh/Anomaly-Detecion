import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import faiss

from tools.visualization import overlay_anomaly
from tools.metrics import compute_iou

def test_model(memory_bank, backbone, test_dir, gt_dir, device, threshold=0.05):
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    gt_masks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    backbone.eval()
    index = faiss.IndexFlatL2(memory_bank.shape[1])
    index.add(memory_bank.numpy())

    for i in range(5):
        img = Image.open(test_images[i]).convert("RGB")
        gt = Image.open(gt_masks[i]).convert("L")
        img_tensor = transform(img).unsqueeze(0).to(device)
        gt_tensor = transform(gt).squeeze(0).cpu().numpy()

        with torch.no_grad():
            features = backbone(img_tensor)

            patch_list = []
            for fmap in features:
                fmap = F.interpolate(fmap, size=(16, 16), mode='bilinear', align_corners=False)
                B, C, H, W = fmap.shape
                fmap = fmap.view(B, C, H * W)  # [B, C, N]
                patch_list.append(fmap)

            concat = torch.cat(patch_list, dim=1)  # [B, C_total, N]
            concat = concat.permute(0, 2, 1).squeeze(0).cpu().numpy()  # [N, C_total]

        D, _ = index.search(concat, k=1)
        anomaly_map = D.reshape(16, 16)
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        anomaly_map = F.interpolate(torch.tensor(anomaly_map).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear').squeeze().numpy()

        binary_map = (anomaly_map > threshold).astype(np.uint8)
        overlay_img = overlay_anomaly(img, binary_map)
        iou = compute_iou(anomaly_map, gt_tensor, threshold)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[1].imshow(gt_tensor, cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(overlay_img)
        axs[2].set_title(f"Overlay\nIoU: {iou:.4f}")
        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.show()
