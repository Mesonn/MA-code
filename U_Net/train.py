import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import config
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_prediction_imgs
)


def train_fn(loader, model, optimizer,loss_fn,scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device = config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device = config.DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds,targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        loop.set_postfix(loss = loss.item())   

def main():
    model = UNet(in_channels=3 , out_channels=1).to(device = config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr =config.LEARNING_RATE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE,
        )

    train_loader,val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        config.train_transform,
        config.val_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS): 
        train_fn(train_loader,model,optimizer,loss_fn,scaler)

        # save Model 
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(model, optimizer , filename=config.CHECKPOINT_FILE)

        check_accuracy(val_loader,model,device=config.DEVICE)            
        if epoch % 1 == 0:
            save_prediction_imgs(val_loader, model, folder= config.OUTPUT_DIR)




if __name__ == "__main__":
    main()