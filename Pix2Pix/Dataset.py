import numpy as np
from PIL import Image
import config
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image


class GehirnDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        image = np.array(Image.open(img_path))
        width = image.shape[1]
        width = width //2 
        input_image = image[:, width:, :]
        target_image = image[:,:width,:]

        augmentations = config.both_transform(image= input_image,image0 = target_image)
        input_image,target_image = augmentations["image"],augmentations["image0"]

        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image = target_image)["image"]

        return input_image,target_image
    
if __name__ == "__main__":
    dataset = GehirnDataset("/home/gizmoo/dscience/Essentials/Pix2Pix/dataset/train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        #print(x.shape)
        save_image(x * 0.5 +0.5, "x.png")
        save_image(y * 0.5 +0.5, "y.png")
        import sys

        sys.exit()




           
