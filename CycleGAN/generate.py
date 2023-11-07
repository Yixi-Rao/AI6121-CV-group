import torch
import config

import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.utils import save_image
from generator_model import Generator

def load_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
def translate_image(image_path, G):
    transforms = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                            ToTensorV2()])
    
    image            = transforms(image=np.array(Image.open(image_path).convert("RGB")))['image']
    translated_image = G(image.cuda())
    save_image(image * 0.5 + 0.5, f"result/source_{'00001'}.png")
    save_image(translated_image * 0.5 + 0.5, f"result/target_{'00001'}.png")

if __name__ == "__main__":
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_T = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    load_checkpoint(config.CHECKPOINT_GEN_S, gen_S)
    load_checkpoint(config.CHECKPOINT_GEN_T, gen_T)
    
    translate_image('data/train/sources/00002.png', gen_S)