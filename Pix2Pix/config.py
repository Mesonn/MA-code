import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,VisualInformationFidelity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryJaccardIndex,Dice
import os
import cv2
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset/train")
VAL_DIR = os.path.join(BASE_DIR, "dataset/test")
OUTPUT_DIR  = os.path.join(BASE_DIR, f"output/out_{timestamp}")
LOG_DIR = os.path.join(BASE_DIR, f"logs/log_{timestamp}") 
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"saves/save_{timestamp}")
CHECKPOINT_DISC = os.path.join(CHECKPOINT_DIR, "disc.pth.tar")
CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, "gen.pth.tar")

both_transform = A.Compose(
    [A.Resize(width=256, height=256,interpolation=cv2.INTER_NEAREST),
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p =0.5),
     #A.Equalize(p = 0.5),
     #A.ColorJitter(p=0.2),
     ], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [   
        #A.HorizontalFlip(p=0.5),
        #A.ColorJitter(p=0.2),
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)

GEN_METRICS ={
    "PSNR": PeakSignalNoiseRatio(),
    "SSIM":StructuralSimilarityIndexMeasure(),
    "LPIPS":LearnedPerceptualImagePatchSimilarity(),
    "VIF":VisualInformationFidelity()
}

DISC_METRICS = {
    "Accuracy": BinaryAccuracy(),
    "F1 Score":BinaryF1Score(),
}

if __name__ == "__main__":
    print(TRAIN_DIR)