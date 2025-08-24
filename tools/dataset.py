from torch.utils.data import Dataset
from PIL import Image
import os

class BottleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
