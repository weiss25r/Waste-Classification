import torch
import pandas as pd

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GarbageClsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=',', header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx][0]
        label = self.img_labels.iloc[idx][1]
        
        img_path = os.path.join(self.img_dir, str(self.classes[label]), filename)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label