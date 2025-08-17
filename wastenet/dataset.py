import torch
import pandas as pd

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

class WasteDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
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
    
class WasteDatasetModule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size=32, num_workers=0, train_transform=None, val_test_transform=None):
        super().__init__()
        self.img_dir = dataset_dir
        self.batch_size = batch_size
        
        self.num_workers = num_workers
        
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = WasteDataset(
                os.path.join(self.img_dir,'train.csv'),
                os.path.join(self.img_dir, 'imgs'),
                self.train_transform
            )
            
            self.val_dataset = WasteDataset(
                os.path.join(self.img_dir,'val.csv'),
                os.path.join(self.img_dir, 'imgs'),
                self.val_test_transform
            )
            
        if stage == 'test':
            self.test_dataset = WasteDataset(
                os.path.join(self.img_dir,'test.csv'),
                os.path.join(self.img_dir, 'imgs'),
                self.val_test_transform
            )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)