import os

import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


class ImageCSVDataset(Dataset):
    def __init__(self, root: str, csv_file: str, transform=lambda x: x):
        super().__init__()
        self.root = root
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> torch.Tensor:
        fname = os.path.join(self.root, self.df.iloc[index, 0])
        image = Image.open(fname).convert('RGB')
        image = self.transform(image)
        return image


def get_dataloaders(root: str = 'data/COCO/',
                    train_csv: str = 'data/COCO/train_0.csv',
                    val_csv: str = 'data/COCO/val.csv',
                    batch_size: int = 16,
                    val_batch_size: int = 16,
                    num_workers: int = 0):

    train_tfm = transforms.Compose([transforms.RandomCrop([256, 256], pad_if_needed=True),
                                    transforms.ToTensor(),])
    val_tfm = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop([256, 256]),
                                  transforms.ToTensor(),])

    trainset = ImageCSVDataset(root, train_csv, train_tfm)
    valset = ImageCSVDataset(root, val_csv, val_tfm)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(valset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader}


def cal_psnr(source_img, target_img, mean=0, std=1):
    delta = source_img - target_img
    delta = 255 * (delta * std)
    delta = delta.reshape(-1, source_img.shape[-3],
                          source_img.shape[-2],
                          source_img.shape[-1])
    psnr = 20 * np.log10(255) \
        - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))
    return psnr


def cal_acc(source_msg: torch.Tensor, target_msg: torch.Tensor):
    acc = (torch.where(source_msg > 0.5, 1, 0) ==
           target_msg.to(torch.uint8)).to(torch.float).mean(dim=-1)
    return acc
