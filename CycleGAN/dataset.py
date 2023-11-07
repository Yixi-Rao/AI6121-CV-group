import os
from PIL import Image

from torch.utils.data import Dataset
import numpy as np

class CycleGAN_Dataset(Dataset):
    def __init__(self, root_s, root_t, trans=None, start=0, end=2000):
        self.root_s = root_s
        self.root_t = root_t
        self.trans  = trans
        
        self.sourse_dir = os.listdir(root_s)[start: end + 1]
        self.target_dir = os.listdir(root_t)[start: end + 1]
        
        self.dataset_len = max(len(self.sourse_dir), len(self.target_dir))
        self.source_len  = end - start
        self.target_len  = end - start

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        source_img = self.sourse_dir[index % self.source_len]
        target_img = self.target_dir[index % self.target_len]

        img_s_path = os.path.join(self.root_s, source_img)
        img_t_path = os.path.join(self.root_t, target_img)

        source_img = np.array(Image.open(img_s_path).convert("RGB"))
        target_img = np.array(Image.open(img_t_path).convert("RGB"))

        if self.trans:
            aug        = self.trans(image=source_img, image0=target_img)
            source_img = aug["image"]
            target_img = aug["image0"]

        return source_img, target_img
