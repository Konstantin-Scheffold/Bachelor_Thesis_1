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

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200
                    , help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=36, help="size of image height")
parser.add_argument("--img_width", type=int, default=30, help="size of image width")
parser.add_argument("--img_depth", type=int, default=30, help="size of image depth")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(,
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=20
                    , help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 2, opt.img_width // 2 ** 2, opt.img_depth // 2 ** 2)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

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

# Configure dataloaders
## mean is taken from: CT_mean = np.mean(Data_CT.data[30:-30, 30:-30, 30:-30])
transforms_ = [
 # braucht man das überhaupt?   transforms.Resize((opt.img_height, opt.img_width, opt.img_depth), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5) #sind die Werte in ordnung oder nicht etwas ranom 0.5
]

dataloader = DataLoader(
    ImageDataset("../Data/Dicom_Data_edited/train", transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
)

val_dataloader = DataLoader(
    ImageDataset("../Data/Dicom_Data_edited/val", transforms_=transforms_, mode="val"),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["CT"].type(Tensor))
    real_B = Variable(imgs["PD"].type(Tensor))
    fake_B = generator(real_A)
    sub_data_CT_PD = np.array([real_A, real_B,fake_B], dtype = object)
    np.save("images/%s/%s" % (opt.dataset_name, batches_done), sub_data_CT_PD)


# ----------
#  Training documentation
# ----------
loss_steps_G = []
loss_steps_D = []
loss_steps_pixel = []
loss_steps_GAN = []

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    loss_batch_G = []
    loss_batch_D = []
    loss_batch_GAN = []
    loss_batch_pixel = []

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
        fake_PD = generator(real_CT)
        pred_fake = discriminator(fake_PD, real_CT)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_batch_GAN.append(loss_GAN.item())

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_PD, real_PD)
        loss_batch_pixel.append(loss_pixel.item())

        # Total loss
        loss_G = loss_GAN + loss_pixel #  lambda_pixel * loss_pixel
        loss_batch_G.append(loss_G.item())

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss

        pred_real = discriminator(real_PD, real_CT)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_PD.detach(), real_CT)
        #        a, b, c = fake.size()[2]-pred_fake.size()[2], fake.size()[3]-pred_fake.size()[3], fake.size()[4]-pred_fake.size()[4]
        #        pred_fake = nn.ConstantPad3d((c, 0, b, 0, a, 0), 0)(pred_fake)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_batch_D.append(loss_D.item())

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

    loss_steps_D.append(np.mean(loss_batch_D))
    loss_steps_G.append(np.mean(loss_batch_G))
    loss_steps_GAN.append(np.mean(loss_batch_GAN))
    loss_steps_pixel.append(np.mean(loss_batch_pixel))


        # If at sample interval save image
    if epoch % opt.checkpoint_interval == 0:#batches_done % opt.sample_interval == 0:
        sample_images(batches_done)

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))


plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(opt.epoch, opt.n_epochs), loss_steps_D,label="Disciminator")
plt.plot(range(opt.epoch, opt.n_epochs), loss_steps_G,label="Generator")
plt.plot(range(opt.epoch, opt.n_epochs), loss_steps_GAN,label="GAN")
plt.plot(range(opt.epoch, opt.n_epochs), loss_steps_pixel,label="pixelwise loss")
plt.xticks(np.arange(1, opt.n_epochs+1, 1.0))
plt.legend()
plt.show()