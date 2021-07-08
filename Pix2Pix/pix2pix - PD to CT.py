import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models_PD_to_CT import *
from datasets import *

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--img_height", type=int, default=20, help="size of image height")
parser.add_argument("--img_width", type=int, default=17, help="size of image width")
parser.add_argument("--img_depth", type=int, default=17, help="size of image depth")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500,
                    help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

#validation = True
lambda_pixel = 3 # Loss weight of L1 pixel-wise loss between translated image and real image
Loss_D_rate = 1 # slows down Discriminator loss to balance Disc and Gen
# Calculate output of image discriminator (PatchGAN)
patch = (1, 3, 3, 3)

os.makedirs("PDtoCT/images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("PDtoCT/saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

#def criterion_pixelwise(output, target):
#    loss = torch.mean(torch.abs((output - target)**3))
#    return torch.tensor(loss, device=torch.device('cuda'))

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()


# Initialize generator and discriminator
generator = GeneratorWideUNet()
discriminator = Discriminator()

# Tensor type - here the type of Tensor is set. It needs to be done as well in the weight init method
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.2, patience=200, cooldown=0, verbose=True, min_lr=10**-8)
# scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor = 0.4, patience =  500,
# cooldown=0, verbose=True, min_lr=10**-8)

transforms_ = [
    transforms.ToTensor()
    #transforms.Normalize(mean=-1, std=2)
]
# Configure dataloaders
dataloader = DataLoader(
    ImageDataset("../Data/Dicom_Data_edited/train", transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    ImageDataset("../Data/Dicom_Data_edited/val", transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
)


def sample_images(imgs, batches_done):
    """Saves a generated sample from the validation set"""
    real_CT = imgs["CT"].type(Tensor)
    real_PD = imgs["PD"].type(Tensor)
    fake_PD = generator(real_CT)
    sub_data_CT_PD = np.array([real_CT, real_PD, fake_PD], dtype=object)
    np.save("PDtoCT/images/%s/%s" % (opt.dataset_name, batches_done), sub_data_CT_PD)


# ----------
#  Training documentation
# ----------

loss_batch_G, loss_batch_D, loss_batch_GAN, loss_batch_pixel = [], [], [], [],
loss_steps_G, loss_steps_D, loss_steps_pixel, loss_steps_GAN = [], [], [], []
loss_batch_G_val, loss_batch_D_val, loss_batch_GAN_val, loss_batch_pixel_val = [], [], [], []
loss_steps_G_val, loss_steps_D_val, loss_steps_pixel_val, loss_steps_GAN_val = [], [], [], []
D_accuracy_real, D_accuracy_fake, D_accuracy_fake_img, D_accuracy_real_img = [], [], [], []

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_CT = Variable(batch["CT"].type(Tensor))
        real_PD = Variable(batch["PD"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_CT.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_CT.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss

        fake_CT = generator(real_PD)
        pred_fake = discriminator(real_PD, fake_CT)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_batch_GAN.append(loss_GAN.cpu().item())

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_CT, real_CT)
        loss_batch_pixel.append(lambda_pixel * loss_pixel.cpu().item())

        # Total loss
        loss_G = lambda_pixel * loss_pixel + loss_GAN
        loss_batch_G.append(loss_G.cpu().item())

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss

        pred_real = discriminator(real_PD.detach(), real_CT.detach())
        loss_real = criterion_GAN(pred_real, valid)
        D_accuracy_real.append(np.round(np.mean(pred_real.cpu().detach().numpy())))

        # Fake loss
        pred_fake = discriminator(real_PD.detach(), fake_CT.detach())
        loss_fake = criterion_GAN(pred_fake, fake)
        D_accuracy_fake.append(np.round(np.mean((valid-pred_fake).cpu().detach().numpy())))

        # Total loss
        loss_D = Loss_D_rate * (loss_real + loss_fake)
        loss_batch_D.append(loss_D.cpu().item())

        #if i % rate_D_G_train == 0:
        loss_D.backward()
        optimizer_D.step()

        # --------------
        # Cross validation
        # --------------

        val_imgs = next(iter(val_dataloader))
        # Model inputs
        real_CT_val = val_imgs["CT"].type(Tensor)
        real_PD_val = val_imgs["PD"].type(Tensor)
        # Adversarial ground truths
        valid_val = Variable(Tensor(np.ones((real_CT_val.size(0), *patch))), requires_grad=False)
        fake_val = Variable(Tensor(np.zeros((real_CT_val.size(0), *patch))), requires_grad=False)
        # ------------------
        #  Generators validation
        # ------------------
        # GAN loss
        fake_CT_val = generator(real_PD_val)
        pred_fake_val = discriminator(real_PD_val, fake_CT_val)
        loss_GAN_val = criterion_GAN(pred_fake_val, valid_val)
        loss_batch_GAN.append(loss_GAN_val.item())
        # Pixel-wise loss
        loss_pixel_val = criterion_pixelwise(fake_CT_val, real_CT_val)
        loss_batch_pixel_val.append(lambda_pixel * loss_pixel_val.item())
        # Total loss
        loss_G_val = loss_GAN_val + lambda_pixel * loss_pixel_val
        loss_batch_G_val.append(loss_G_val.item())

        scheduler_G.step(loss_G_val)
        # ---------------------
        #  Discriminator validation
        # ---------------------
        # Real loss
        pred_real_val = discriminator(real_PD_val, real_CT_val)
        loss_real_val = criterion_GAN(pred_real_val, valid_val)
        # Fake loss
        pred_fake_val = discriminator(real_PD_val.detach(), fake_CT_val)
        loss_fake_val = criterion_GAN(pred_fake_val, fake_val)
        # Total loss
        loss_D_val = Loss_D_rate * (loss_real_val + loss_fake_val)
        loss_batch_D_val.append(loss_D_val.item())

        # scheduler_D.step(loss_D_val)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        time_left_accumulate = []
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left_accumulate.append(datetime.timedelta(seconds=batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        if i % 20 == 0:
            time_left = np.mean(time_left_accumulate)
            time_left_accumulate = []

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                time_left,
            )
        )

        if i % 5 == 0:
            # plot the loss curves
            loss_steps_D.append(np.mean(loss_batch_D)/Loss_D_rate)
            loss_steps_G.append(np.mean(loss_batch_G))
            loss_steps_GAN.append(np.mean(loss_batch_GAN))
            loss_steps_pixel.append(np.mean(loss_batch_pixel))
            loss_batch_G, loss_batch_D, loss_batch_GAN, loss_batch_pixel = [], [], [], []

            loss_steps_D_val.append(np.mean(loss_batch_D_val)/Loss_D_rate)
            loss_steps_G_val.append(np.mean(loss_batch_G_val))
            loss_steps_GAN_val.append(np.mean(loss_batch_GAN_val))
            loss_steps_pixel_val.append(np.mean(loss_batch_pixel_val))
            loss_batch_G_val, loss_batch_D_val, loss_batch_GAN_val, loss_batch_pixel_val = [], [], [], []

            D_accuracy_real_img.append(np.mean(D_accuracy_real))
            D_accuracy_fake_img.append(np.mean(D_accuracy_fake))
            D_accuracy_real, D_accuracy_fake = [], []

        if i % 20 == 0 and len(loss_steps_D) > 2:
            plt.figure(figsize=(14, 8))
            # plot Loss curves
            plt.subplot(2, 6, (1, 3))
            plt.ylabel("Loss curves")
            plt.plot(range(len(loss_steps_D)), loss_steps_D, 'y-', label="Disciminator")
            plt.plot(range(len(loss_steps_G)), loss_steps_G, 'b-', label="Generator")
            plt.plot(range(len(loss_steps_GAN)), loss_steps_GAN, 'r-', label="GAN")
            plt.plot(range(len(loss_steps_pixel)), loss_steps_pixel, 'c-', label="pixelwise loss")

            plt.plot(range(len(loss_steps_D_val)), loss_steps_D_val, 'y--')
            plt.plot(range(len(loss_steps_G_val)), loss_steps_G_val, 'b--')
            plt.plot(range(len(loss_steps_GAN_val)), loss_steps_GAN_val, 'r--')
            plt.plot(range(len(loss_steps_pixel_val)), loss_steps_pixel_val, 'c--')
            plt.xticks(np.arange(1, len(loss_steps_D)+1, 10))
            plt.grid()
            plt.title('loss_curve lr:{},'
                      ' lambda_pixel:{}, lr_D_adjust{},'
                      .format(opt.lr, Loss_D_rate, np.round(lambda_pixel, 7)))
            plt.ylim(0, 2)
            plt.legend()

            # plot Discriminator Accuracy
            plt.subplot(2, 6, (7, 9))
            plt.xlabel("Training Epochs")
            plt.ylabel("Discriminator Accuracy")
            plt.plot(range(len(D_accuracy_real_img)), D_accuracy_real_img, label="Disc_accuracy_real")
            plt.plot(range(len(D_accuracy_fake_img)), D_accuracy_fake_img, label="Disc_accuracy_fake")
            plt.grid()
            plt.legend()

            # plot real_PD
            real_PD_val = real_CT_val[0].cpu().squeeze().detach().numpy()
            fake_PD_val = fake_CT_val[0].cpu().squeeze().detach().numpy()

            plt.subplot(2, 6, 4)
            plt.ylabel('real_PD')
            plt.title('x-axis')
            plt.imshow(real_PD_val[int(np.size(real_PD_val, 0) / 2), :, :])
            plt.subplot(2, 6, 5)
            plt.title('y-axis')
            plt.imshow(real_PD_val[:, int(np.size(real_PD_val, 1) / 2), :])
            plt.subplot(2, 6, 6)
            plt.title('z-axis')
            plt.imshow(real_PD_val[:, :, int(np.size(real_PD_val, 2) / 2)])
            # plot fake_PD
            plt.subplot(2, 6, 10)
            plt.ylabel('fake_PD')
            plt.imshow(fake_PD_val[int(np.size(fake_PD_val, 0) / 2), :, :])
            plt.subplot(2, 6, 11)
            plt.imshow(fake_PD_val[:, int(np.size(fake_PD_val, 1) / 2), :])
            plt.subplot(2, 6, 12)
            plt.imshow(fake_PD_val[:, :, int(np.size(fake_PD_val, 2) / 2)])
            plt.show()

            # If at sample interval save image
    if epoch % opt.checkpoint_interval == 0:
        imgs = next(iter(val_dataloader))
        sample_images(imgs, epoch)

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "PDtoCT/saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "PDtoCT/saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
