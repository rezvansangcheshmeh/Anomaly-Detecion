import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

def show_reconstruction(original, reconstructed):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original")
    axs[1].imshow(reconstructed.permute(1, 2, 0).detach().cpu().numpy())
    axs[1].set_title("Reconstructed")
    plt.show()

def show_anomaly_map(original, anomaly_map):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original")
    sns.heatmap(anomaly_map.squeeze(), ax=axs[1], cmap="jet")
    axs[1].set_title("Anomaly Map")
    plt.show()

def overlay_anomaly(original_img, binary_mask, color=(255, 0, 0)):
    img_np = np.array(original_img.resize((256, 256)))
    mask_rgb = np.zeros_like(img_np)
    mask_rgb[binary_mask == 1] = color
    return cv2.addWeighted(img_np, 0.8, mask_rgb, 0.5, 0)
