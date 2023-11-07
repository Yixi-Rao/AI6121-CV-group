import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR           = "data/train"
VAL_DIR             = "data/val"

BATCH_SIZE          = 2
LEARNING_RATE       = 1e-5
LAMBDA_CYCLE        = 10

NUM_WORKERS         = 4
NUM_EPOCHS          = 2

RESUME              = False
LOAD_MODEL_STAGE    = 1
SAVE_MODEL          = True

CHECKPOINT_GEN_T    = "saved_models/genT.pth.tar"
CHECKPOINT_GEN_S    = "saved_models/genS.pth.tar"

CHECKPOINT_CRITIC_T = "saved_models/criticT.pth.tar"
CHECKPOINT_CRITIC_S = "saved_models/criticS.pth.tar"

CHECKPOINT_schLR_D = "saved_models/schLRD.pth.tar"
CHECKPOINT_schLR_G = "saved_models/schLRG.pth.tar"

transforms = A.Compose([A.Resize(width=360, height=180),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                        ToTensorV2()], additional_targets={"image0": "image"})
