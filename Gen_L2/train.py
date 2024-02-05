import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import config
from torchmetrics import MetricCollection
from Generator import Generator
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    #check_accuracy,
    create_directory,
    save_imgs,
    #plot_one_example,
    #get_all_metrics,
    #plot_and_save_metrics,plot_index_example
)
from torch.utils.tensorboard import SummaryWriter


def train_fn(loader, model, optimizer,loss_fn,scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device = config.DEVICE)
        targets = targets.float().to(device = config.DEVICE)

        
        preds = model(data)
        #print(preds.shape,targets.shape)
        loss = loss_fn(preds,targets)

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()  


        loop.set_postfix(loss = loss.item())   

def main():
    create_directory(config.OUTPUT_DIR)
    create_directory(config.CHECKPOINT_DIR)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    model = Generator(in_channels=3).to(device = config.DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(),lr =config.LEARNING_RATE)
    metrics = MetricCollection(config.METRICS).to(device = config.DEVICE)


    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE,
        )

    train_loader,val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_TRANS_DIR,
        config.VAL_IMG_DIR,
        config.VAL_TRANS_DIR,
        config.BATCH_SIZE,
        config.TRAIN_TRANSFORM,
        config.TEST_TRANSFORM,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS): 
        train_fn(train_loader,model,optimizer,loss_fn,scaler)

        #save Model 
        #if config.SAVE_MODEL and epoch % 50 == 0:
            #save_checkpoint(model, optimizer , filename=config.CHECKPOINT_FILE)
        #Check accuracy on training data
        #check_accuracy(train_loader, model, device=config.DEVICE, metrics=metrics, writer=writer, epoch=epoch, is_train=True)

        # Check accuracy on validation data
        #check_accuracy(val_loader, model, device=config.DEVICE, metrics=metrics, writer=writer, epoch=epoch, is_train=False)

        if epoch % 10 == 0:
            #save_some_examples(gen, val_loader, epoch, folder= config.OUTPUT_DIR)
            save_imgs(model, val_loader, epoch, folder= config.OUTPUT_DIR)


if __name__ == "__main__":
    main()