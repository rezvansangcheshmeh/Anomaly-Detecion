import torch
import numpy as np
from patchcore.patchcore_model import PatchCoreBackbone

import torch.nn.functional as F

def extract_features(train_loader, device):
    model = PatchCoreBackbone().to(device)
    model.eval()
    all_features = []

    with torch.no_grad():
        for images in train_loader:
            images = images.to(device)
            features = model(images)  # لیستی از feature maps

            patch_list = []
            for fmap in features:
                # resize همه‌ی feature mapها به یک اندازه‌ی فضایی
                fmap = F.interpolate(fmap, size=(16, 16), mode='bilinear', align_corners=False)
                B, C, H, W = fmap.shape
                fmap = fmap.view(B, C, H * W)  # [B, C, N]
                patch_list.append(fmap)

            # حالا کانال‌ها رو کنار هم بذاریم
            concat = torch.cat(patch_list, dim=1)  # [B, C_total, N]
            concat = concat.permute(0, 2, 1)       # [B, N, C_total]
            all_features.append(concat.cpu())

    all_features = torch.cat(all_features, dim=0)  # [Total_Patches, C_total]
    return all_features.view(-1, all_features.shape[-1])  # [N_total, C_total]


