import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        self.list_files = os.listdir(self.root)

        # self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root, img_file)

        dicom_sample = np.load(img_path, allow_pickle=True)
        dicom_sample_CT, dicom_sample_PD = np.array(list(dicom_sample[0]), dtype= float), np.array(list(dicom_sample[1]), dtype=float)

        dicom_sample_CT = self.transform(dicom_sample_CT).permute(1,2,0).unsqueeze(0)
        dicom_sample_PD = self.transform(dicom_sample_PD).permute(1,2,0).unsqueeze(0)

        return {"CT": dicom_sample_CT, "PD": dicom_sample_PD}

    def __len__(self):
        return len(self.list_files)