import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms
import torchvision.transforms as transforms

interpolation_mode = 'trilinear'

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
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=3, stride=1, padding=1):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=3, stride=1, padding=1):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        self.down0_5 = UNetDown(1, 24, normalize=False)
        self.down1 = UNetDown(24, 48, stride=2, kernel_size=(7, 5, 5))
        self.down2 = UNetDown(48, 96, kernel_size=4,  stride=2)
        self.down3 = UNetDown(96, 192,  kernel_size=4, stride=2)
        self.down4 = UNetDown(192, 192, dropout=0.5, kernel_size=2, stride=2)
        self.down5 = UNetDown(192, 192, dropout=0.5)
        self.down6 = UNetDown(192, 192, dropout=0.5, normalize=False)

        self.up1 = UNetUp(192, 192, dropout=0.5)
        self.up2 = UNetUp(384, 192, dropout=0.5)
        self.up3 = UNetUp(384, 192, dropout=0.5, kernel_size=2, stride=2)
        self.up4 = UNetUp(384, 96, kernel_size=4, stride=2, padding=1)
        self.up5 = UNetUp(192, 48, kernel_size=4, stride=2, padding=1)
        self.up6 = UNetUp(96, 24, stride=2, kernel_size=(8, 5, 5))

        self.final = nn.Sequential(
            nn.ConvTranspose3d(48, 1, kernel_size=4, stride=2, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d0_5 = self.down0_5(x)
        d1 = self.down1(d0_5)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        u6 = self.up6(u5, d0_5)
        u7 = self.final(u6)
        u8 = nn.functional.interpolate(u7, size=(52, 49, 49), mode=interpolation_mode)

        return u8


##############################
#       Wide U-Net
##############################


class WideUNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=5, stride=2, padding=1):
        super(WideUNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, in_size, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.InstanceNorm3d(in_size),
                  nn.LeakyReLU(0.2, inplace=True)]
        #if normalize:
        #    layers.append(nn.InstanceNorm3d(in_size))
        #layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WideUNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=4, stride=2, padding=1, output_padding=0):
        super(WideUNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, in_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(in_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, output_padding=output_padding),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorWideUNet(nn.Module):
    def __init__(self):
        super(GeneratorWideUNet, self).__init__()

        self.down1 = WideUNetDown(1, 24, normalize=False, stride=2, kernel_size=(7, 5, 5))
        self.down2 = WideUNetDown(24, 48, kernel_size=4,  stride=2)
        self.down3 = WideUNetDown(48, 96,  kernel_size=4, stride=2)
        self.down4 = WideUNetDown(96, 96, dropout=0.5, kernel_size=2, stride=2)
        self.down5 = WideUNetDown(96, 96, dropout=0.5, kernel_size=3, stride=1, normalize=False)
        #self.down6 = WideUNetDown(96, 192, dropout=0.5, normalize=False, kernel_size=3, stride=1)

        #self.up1 = WideUNetUp(192, 96, dropout=0.5, kernel_size=3, stride=1)
        self.up2 = WideUNetUp(96, 96, dropout=0.5, kernel_size=3, stride=1)
        self.up3 = WideUNetUp(192, 96, dropout=0.5, kernel_size=2, stride=2)
        self.up4 = WideUNetUp(192, 48, kernel_size=4, stride=2, padding=1)
        self.up5 = WideUNetUp(96, 24, kernel_size=4, stride=2)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(48, 1, kernel_size=4, stride=2, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        #d6 = self.down6(d5)

        #u1 = self.up1(d6, d5)
        u2 = self.up2(d5, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        u6 = self.final(u5)
        u7 = nn.functional.interpolate(u6, size=(52, 49, 49), mode=interpolation_mode)

        return u7


class FinalUNet_long(nn.Module):
    def __init__(self):
        super(FinalUNet_long, self).__init__()

        self.down0 = WideUNetDown(1, 16, normalize=False, stride=1, kernel_size=(7, 4, 4))
        self.down1 = WideUNetDown(16, 32, stride=1, kernel_size=3)
        self.down2 = WideUNetDown(32, 48, kernel_size=4, stride=2)
        self.down3 = WideUNetDown(48, 96, kernel_size=4, stride=2)
        self.down4 = WideUNetDown(96, 96, dropout=0.5, kernel_size=2, stride=2)
        self.down5 = WideUNetDown(96, 96, dropout=0.5, kernel_size=3, stride=1, normalize=False)

        self.up1 = WideUNetUp(96, 96, dropout=0.5, kernel_size=3, stride=1)
        self.up2 = WideUNetUp(192, 96, dropout=0.5, kernel_size=2, stride=2)
        self.up3 = WideUNetUp(192, 48, kernel_size=4, stride=2, padding=1)
        self.up4 = WideUNetUp(96, 32, kernel_size=4, stride=2, padding=1)
        self.up5 = WideUNetUp(64, 16, stride=1, kernel_size=3, padding=1)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4, d0)
        u6 = self.final(u5)
        u7 = nn.functional.interpolate(u6, size=(52, 49, 49), mode=interpolation_mode)

        return u7

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, stride=1, kernel_size=3, padding=1):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(2, 24, kernel_size=3, stride=2,  normalization=False, padding=0),
            *discriminator_block(24, 48, kernel_size=3, stride=1, padding=0),
            *discriminator_block(48, 96, kernel_size=3, stride=1, padding=0),
            #*discriminator_block(96, 96),
            nn.Conv3d(96, 1, kernel_size=3, stride=1, bias=False, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img_PD, img_CT):
        # Concatenate image and condition image by channels to produce input
        if img_PD.size() != img_CT.size():
            img_PD = nn.functional.interpolate(img_PD, size=img_CT.size()[2:], mode=interpolation_mode)

        img_input = torch.cat((img_PD, img_CT), 1)
        #noise = torch.normal(0, 0.1, list(img_input.size()), device=torch.device('cuda'), requires_grad=True)
        output = self.model(img_input)#+noise)
        return output

