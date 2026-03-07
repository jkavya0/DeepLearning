from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

# Provided normalization values
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str = "train"):
        """
        Args:
            data (pandas.DataFrame): rows contain image path + two labels.
              - Robust behavior:
                * image path: uses "filename" if present, else the FIRST column
                * labels: uses ["crack","inactive"] if present, else the LAST TWO columns
            mode (str): "train" or "val"
        """
        self.data = data.reset_index(drop=True)
        self.mode = mode.lower()
        assert self.mode in {"train", "val"}, "mode must be 'train' or 'val'"

        # Define transformations (simple and to the point)
        if self.mode == "train":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # --- Image path: prefer "filename", else first column ---
        if "filename" in self.data.columns:
            img_path = row["filename"]
        else:
            img_path = row.iloc[0]

        img = imread(Path(img_path))

        # Convert grayscale (H,W) or (H,W,1) to RGB
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            img = gray2rgb(img)

        img = self.transform(img)

        # --- Labels: prefer ["crack","inactive"], else last two columns ---
        if all(col in self.data.columns for col in ["crack", "inactive"]):
            crack_label = float(row["crack"])
            inactive_label = float(row["inactive"])
            label = torch.tensor([crack_label, inactive_label], dtype=torch.float32)
        else:
            last_two = row.iloc[-2:].to_numpy(dtype=np.float32)
            label = torch.from_numpy(last_two).float()

        return img, label
