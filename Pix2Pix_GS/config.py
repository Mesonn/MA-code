import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,VisualInformationFidelity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryJaccardIndex,Dice
import os
import cv2
from datetime import datetime
timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "dataset/train/bface")
TRAIN_TRANS_DIR = os.path.join(BASE_DIR, "dataset/train/transgrey")
VAL_IMG_DIR = os.path.join(BASE_DIR, "dataset/test/bface")
VAL_TRANS_DIR = os.path.join(BASE_DIR, "dataset/test/transgrey")
OUTPUT_DIR  = os.path.join(BASE_DIR, f"output/out_{timestamp}")
LOG_DIR = os.path.join(BASE_DIR, f"logs/log_{timestamp}") 
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 501
LOAD_MODEL = False
SAVE_MODEL = True
TRANSFORM = True
PIN_MEMORY = True
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"saves/save_{timestamp}")
CHECKPOINT_DISC = os.path.join(CHECKPOINT_DIR, "disc.pth.tar")
CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, "gen.pth.tar")



TRAIN_TRANSFORM = A.Compose(
    [

        A.Resize(width=256, height=256,interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p =0.5),
        A.RandomRotate90(p =0.5),
        #A.RandomScale(p=0.5),
        #A.RandomBrightnessContrast(p = 0.5),
        #A.RandomGamma(p =0.5),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
        
    ]
)



TEST_TRANSFORM = A.Compose(
    [   
        A.Resize(width=256, height=256,interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=0.5,std=0.5),
        ToTensorV2(),
    ]
)





GEN_METRICS ={
    "MSE" : MeanSquaredError(),
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
    print(TRAIN_IMG_DIR)