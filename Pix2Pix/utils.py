import torch
import config
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import tifffile as tiff
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import cv2
from patchify import patchify,unpatchify
from torchvision.utils import save_image

def save_imgs(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    
    gen.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        prediction = gen(x)
    
    plt.figure(figsize=(15,15))
    for i in range(3):
        plt.subplot(3, 3, i*3+1)
        plt.title("Input Image")
        plt.imshow(x[i].cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(3, 3, i*3+2)
        plt.title("Ground Truth")
        plt.imshow(y[i].cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.axis("off")

        plt.subplot(3, 3, i*3+3)
        plt.title("Prediction Image")
        plt.imshow(prediction[i].cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.axis("off")
    
    plt.savefig(os.path.join(folder, f"epoch_{epoch}.png"))
    plt.close()

def create_directory(directory, overwrite=False):
    if os.path.exists(directory) and overwrite:
        shutil.rmtree(directory)  # Removes an existing directory and its contents
    if not os.path.exists(directory):
        os.makedirs(directory)  # Creates a new directory

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def natural_sort_key(s):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def check_accuracy(loader, generator, discriminator,disc_metrics,gen_metrics ,device = config.DEVICE, writer=None, epoch=None, is_train=False):
    generator.eval()
    discriminator.eval()


    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            fake = generator(x)
            real_preds = discriminator(x, y).view(-1)
            fake_preds = discriminator(x, fake.detach()).view(-1)

            real_targets = torch.ones_like(real_preds, device=device)
            fake_targets = torch.zeros_like(fake_preds, device=device)

            # Update disc metrics
            disc_metrics.update(torch.cat([real_preds, fake_preds]), torch.cat([real_targets, fake_targets]))
            #Update gen metrics 
            gen_metrics.update(fake, y)
        # Compute metrics
        computed_disc_metrics = disc_metrics.compute()
        computed_gen_metrics = gen_metrics.compute()
        # Add metrics to TensorBoard
        for name, value in computed_disc_metrics.items():
            writer.add_scalar(f'{"Train" if is_train else "Validation"}/{name}', value, epoch)
        for name, value in computed_gen_metrics.items():
            writer.add_scalar(f'{"Train" if is_train else "Validation"}/{name}', value, epoch)
        disc_metrics.reset()
        gen_metrics.reset()


    generator.train()
    discriminator.train()





def generate_image_patches(images_path,image_patches_path,patch_size):
    patch_size_x , patch_size_y = patch_size
    create_directory(image_patches_path,overwrite=True)
    for path,subdirs,files in os.walk(images_path):
        print(path)
        #dirname = path.split(os.path.sep)
        images = sorted(os.listdir(path),key=natural_sort_key)
        for image_index, image_name in enumerate(images):
            image = cv2.imread(os.path.join(path,image_name),1)
            SIZE_X = (image.shape[1]//patch_size_x)*patch_size_x
            SIZE_Y = (image.shape[0]//patch_size_y)*patch_size_y
            image = Image.fromarray(image)
            image = image.crop((0,0,SIZE_X,SIZE_Y))
            image = np.array(image)

            print ("Now patchifiying image :",os.path.join(path,image_name))
            patches_img = patchify(image,(patch_size_x,patch_size_y,3),step = patch_size_x)
            #print (patches_img.shape)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = single_patch_img[0]

                    cv2.imwrite(image_patches_path+f"image_{image_index}_patch_{i},{j}.png",single_patch_img)


def generate_mask_patches(masks_path,mask_patches_path,patch_size):
    patch_size_x , patch_size_y = patch_size
    create_directory(mask_patches_path,overwrite=True)
    
    for path,subdirs,files in os.walk(masks_path):
        print(path)
        # dirname = path.split(os.path.sep)
        images = sorted(os.listdir(path))
        for image_index, image_name in enumerate(images):
            image = cv2.imread(os.path.join(path,image_name),0)
            SIZE_X = (image.shape[1]//patch_size_x)*patch_size_x
            SIZE_Y = (image.shape[0]//patch_size_y)*patch_size_y
            image = Image.fromarray(image)
            image = image.crop((0,0,SIZE_X,SIZE_Y))
            image = np.array(image)

            print ("Now patchifiying image :",os.path.join(path,image_name))
            patches_img = patchify(image,(patch_size_x,patch_size_y),step = patch_size_x)
            #print (patches_img.shape)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    #single_patch_img = single_patch_img[0]

                    cv2.imwrite(mask_patches_path+f"image_{image_index}_patch_{i},{j}.png",single_patch_img)

def display(display_list, titles=None, cmap=None):
    if titles is None:
        titles = [''] * len(display_list)
        
    num_rows = len(display_list)
    num_cols = len(display_list[0])
    
    plt.figure(figsize=(15, 15 * num_rows / num_cols))
    
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
            plt.title(titles[i])
            plt.imshow(display_list[i][j], cmap=cmap)
            plt.axis('off')
    plt.show()


def unstack_tiffimage(input_file,output_PNG_folder):
    stack = tiff.imread(input_file)
    #os.makedirs(output_folder, exist_ok = True )
    os.makedirs(output_PNG_folder, exist_ok= True )
    for i, frame in enumerate(stack):
        #output_file = os.path.join(output_folder,f"i_{i}.tif")
        #tiff.imwrite(output_file,frame)
        output_png_file = os.path.join(output_PNG_folder,f"{i}.png")
        frame_pil = Image.fromarray(frame)
        #frame_pil = frame_pil.convert("RGB")
        frame_pil.save(output_png_file,'PNG')



def concatenate_images(dir1, dir2, output_dir):
    # Load images from directory1
    images_dir1 = []
    for filename in os.listdir(dir1):
        if filename.lower().endswith(".png"):
            image = cv2.imread(os.path.join(dir1, filename))
            images_dir1.append(image)

    # Load images from directory2
    images_dir2 = []
    for filename in os.listdir(dir2):
        if filename.lower().endswith(".png"):
            image = cv2.imread(os.path.join(dir2, filename))
            images_dir2.append(image)

    # Concatenate images horizontally
    concatenated_images = []
    for img1, img2 in zip(images_dir1, images_dir2):
        concatenated_image = np.concatenate((img1, img2), axis=1)
        concatenated_images.append(concatenated_image)

    # Save concatenated images in output directory
    create_directory(output_dir,overwrite=True)
    for i, image in enumerate(concatenated_images):
        output_path = os.path.join(output_dir, f"{i+1}.png")
        cv2.imwrite(output_path, image)
       


