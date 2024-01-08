import numpy as np
import pandas as pd
from PIL import Image
import config
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image


class GehirnDataset(Dataset):
    def __init__(self,image_dir,trans_dir,transform = None):
        self.image_dir = image_dir
        self.trans_dir = trans_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.trans = sorted(os.listdir(trans_dir))
        self.data = pd.DataFrame(
           {
               'image':self.images,
               'trans':self.trans
           } 
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.data.iloc[index]['image'])
        trans_path = os.path.join(self.trans_dir,self.data.iloc[index]['trans'])

        image = cv2.imread(img_path)
        trans = cv2.imread(trans_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trans = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)

        # image = np.array(Image.open(img_path).convert("RGB"))
        # trans = np.array(Image.open(trans_path).convert("RGB"))


        if self.transform:

            transformed = self.transform(image = image, mask = trans)
            # augmentations = config.both_transform(image= image,trans = trans)
            # image,trans = augmentations["image"],augmentations["trans"]
            image = transformed["image"]
            trans = transformed["mask"]
            
        return image,trans


# class GehirnDataset(Dataset):
#     def __init__(self,root_dir):
#         self.root_dir = root_dir
#         self.list_files = os.listdir(self.root_dir)
#         print(self.list_files)

#     def __len__(self):
#         return len(self.list_files)

#     def __getitem__(self, index):
#         img_file = self.list_files[index]
#         img_path = os.path.join(self.root_dir,img_file)
#         image = np.array(Image.open(img_path))
#         width = image.shape[1]
#         width = width //2 
#         input_image = image[:, :width, :]
#         target_image = image[:,width:,:]

#         augmentations = config.both_transform(image= input_image,image0 = target_image)
#         input_image,target_image = augmentations["image"],augmentations["image0"]

#         input_image = config.transform_only_input(image = input_image)["image"]
#         target_image = config.transform_only_mask(image = target_image)["image"]

#         return input_image,target_image
    
if __name__ == "__main__":
    image_dir = config.TRAIN_IMG_DIR
    trans_dir = config.TRAIN_TRANS_DIR
    dataset = GehirnDataset(image_dir,trans_dir,transform=config.TRAIN_TRANSFORM)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        #print(x.shape)
        #plt.show()
        x = x.numpy().squeeze().transpose(1,2,0)
        y = y.numpy().squeeze()
        print(x.shape)
        print(x.shape, y.shape)
        fig, axs = plt.subplots(1, 2)


        # Display x in the first subplot
        axs[0].imshow(x)
        axs[0].set_title('x')
        axs[0].axis('off')  # Hide axes

        # Display y in the second subplot
        axs[1].imshow(y)
        axs[1].set_title('y')
        axs[1].axis('off')  # Hide axes

        # Show the plot
        plt.show()


        #save_image(x * 0.5 +0.5, "x.png")
        #save_image(y * 0.5 +0.5, "y.png")
        import sys

        sys.exit()




           
