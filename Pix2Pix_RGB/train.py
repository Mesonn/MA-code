import torch
from utils import save_checkpoint, load_checkpoint, check_accuracy,save_imgs,create_directory,get_loaders
import torch.nn as nn
import torch.optim as optim
import config
from Dataset import GehirnDataset
from Generator import Generator
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchmetrics import MetricCollection

#torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=False)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        #with torch.cuda.amp.autocast():
        y_fake = gen(x)
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2 

        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        #d_scaler.scale(D_loss).backward()
        #d_scaler.step(opt_disc)
        #d_scaler.update()

        # Train generator
        #with torch.cuda.amp.autocast():
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
        #g_scaler.scale(G_loss).backward()
        #g_scaler.step(opt_gen)
        #g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    create_directory(config.OUTPUT_DIR)
    create_directory(config.CHECKPOINT_DIR)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    gen_metrics = MetricCollection(config.GEN_METRICS).to(device = config.DEVICE)
    disc_metrics = MetricCollection(config.DISC_METRICS).to(device = config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
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

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )
        check_accuracy(train_loader, gen,disc,disc_metrics,gen_metrics ,device=config.DEVICE, writer=writer, epoch=epoch, is_train=True)
        check_accuracy(val_loader, gen, disc,disc_metrics,gen_metrics ,device=config.DEVICE, writer=writer, epoch=epoch, is_train=False)
        if config.SAVE_MODEL and epoch % 100 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        if epoch % 20 == 0:
            #save_some_examples(gen, val_loader, epoch, folder= config.OUTPUT_DIR)
            save_imgs(gen, val_loader, epoch, folder= config.OUTPUT_DIR)


if __name__ == "__main__":
    main()