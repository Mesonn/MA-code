
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "dataset/train/images")
TRAIN_MASK_DIR =os.path.join(BASE_DIR, "dataset/train/masks")
VAL_IMG_DIR = os.path.join(BASE_DIR, "dataset/test/images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "dataset/test/masks")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
NUM_EPOCHS = 100
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = os.path.join(BASE_DIR,"saves/checkpoint.pth.tar")

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )