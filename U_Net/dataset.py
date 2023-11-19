import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import config
import pandas as pd



class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.data = pd.DataFrame({
            'image': self.images,
            'mask': self.masks
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #img_path = os.path.join(self.image_dir,self.images[index])
        #mask_path = os.path.join(self.mask_dir,self.images[index])
        img_path = os.path.join(self.image_dir,self.data.iloc[index]['image'])
        mask_path = os.path.join(self.mask_dir,self.data.iloc[index]['mask'])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        mask[mask == 255.0] = 1.0


        if self.transform is not None:
            augmentations = self.transform(image = image , mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask    
         

if __name__ == "__main__":
    image_dir = config.TRAIN_IMG_DIR
    mask_dir = config.TRAIN_MASK_DIR
    dataset = SegDataset(image_dir, mask_dir)
    images = dataset.images
    print(images)
