import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, stride = 2, padding = 2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size = 3, stride = 2, padding = 2):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size = kernel_size, stride = stride, padding  = padding, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        a, b, c = skip_input.size()[2]-x.size()[2], skip_input.size()[3]-x.size()[3], skip_input.size()[4]-x.size()[4]
        x = nn.ConstantPad3d((c, 0, b, 0, a, 0), 0)(x)

        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
       # self.pad3 = nn.ConstantPad3d((1,0,1,0,1,0),0)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # was bringt das ? nn.functional.pad( , (1, 0, 1, 0, 1)),
            nn.Conv3d(128, 1, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, stride = 2):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 3, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(2, 64, normalization=False),
            *discriminator_block(64, 128, stride = 1),
            *discriminator_block(128, 256, stride = 1),
            *discriminator_block(256, 512),
            # warum braucht man das ? nn.ZeroPad3d((1, 0, 1, 0, 1)),


            nn.Conv3d(512, 1, 4, padding=1, bias=False)
        )
        self.premodel = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride = 3, padding = 1),
            nn.Conv3d(16, 1, 3, stride = 1, padding = 1)
        )

    def forward(self, img_PD, img_CT):
        # # Concatenate image and condition image by channels to produce input
        #
        # img_CT = transforms.functional.resize(self.premodel(img_CT), size = img_PD.shape())

        a, b, c = img_PD.size()[2]-img_CT.size()[2], img_PD.size()[3]-img_CT.size()[3], img_PD.size()[4]-img_CT.size()[4]
        img_CT = nn.ConstantPad3d((c, 0, b, 0, a, 0), 0)(img_CT)
        img_input = torch.cat((img_PD, img_CT), 1)
        return self.model(img_input)
