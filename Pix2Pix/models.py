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
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=3, stride=2, padding=1):
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

        self.down0_5 = UNetDown(1, 1, kernel_size=(3, 4, 4),  padding=(0, 2, 2), normalize=False)
        self.down1 = UNetDown(1, 32, kernel_size=(6, 5, 5), stride=2, padding=1)
        self.down1_5 = UNetDown(32, 64, kernel_size=(3, 4, 4), stride=2)
        self.down2 = UNetDown(64, 128, kernel_size=3, stride=2)
        self.down3 = UNetDown(128, 128, kernel_size=3, dropout=0.5, stride=1)
        self.down4 = UNetDown(128, 128, dropout=0.5, stride=1)
        self.down5 = UNetDown(128, 128, dropout=0.5, stride=1, normalize=False)
        #self.down6 = UNetDown(512, 512, dropout=0.5, stride=1)
        #self.down7 = UNetDown(512, 512, dropout=0.5, stride=1)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5, stride=1)

        #self.up1 = UNetUp(512, 512, dropout=0.5, stride=1)
        #self.up2 = UNetUp(1024, 512, dropout=0.5, stride=1)
        #self.up3 = UNetUp(1024, 512, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.up4 = UNetUp(128, 128, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.up5 = UNetUp(256, 128, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.up6 = UNetUp(256, 128, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.up7 = UNetUp(256, 64, kernel_size=4, stride=2, padding=1)
        self.up8 = UNetUp(128, 32, kernel_size=4, stride=2, padding=1)
        self.up9 = UNetUp(64, 1, kernel_size=4, stride=2, padding=0)

        self.final = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d0_5 = self.down0_5(x)
        d1 = self.down1(d0_5)
        d1_5 = self.down1_5(d1)
        d2 = self.down2(d1_5)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        #d6 = self.down6(d5)
        #d7 = self.down7(d6)
        #d8 = self.down8(d7)

        #u1 = self.up1(d8, d7)
        #u2 = self.up2(u1, d6)
        #u3 = self.up3(u2, d5)
        u4 = self.up4(d5, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1_5)
        u8 = self.up8(u7, d1)
        u9 = self.up9(u8, d0_5)

        u10 = nn.functional.interpolate(u9, size=(20, 17, 17), mode=interpolation_mode)

        return self.final(u10)


##############################
#       Wide U-Net
##############################


class WideUNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=5, stride=2, padding=1):
        super(WideUNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, in_size, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                  ]
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

        self.down1 = WideUNetDown(1, 32, normalize=False, stride=2, kernel_size=(7, 5, 5))
        self.down2 = WideUNetDown(32, 64, kernel_size=4,  stride=2)
        self.down3 = WideUNetDown(64, 128,  kernel_size=3, stride=1)
        self.down4 = WideUNetDown(128, 128, dropout=0.5, kernel_size=4, stride=2)
        self.down5 = WideUNetDown(128, 128, dropout=0.5, kernel_size=3, stride=1)
        #self.down6 = WideUNetDown(128, 128, dropout=0.5, normalize=False, kernel_size=3, stride=1)

        #self.up1 = WideUNetUp(128, 128, dropout=0.5, kernel_size=3, stride=1)
        self.up2 = WideUNetUp(128, 128, dropout=0.5, kernel_size=3, stride=1)
        self.up3 = WideUNetUp(256, 128, dropout=0.5, kernel_size=4, stride=2)
        self.up4 = WideUNetUp(256, 64, kernel_size=3, stride=1, padding=1)
        self.up5 = WideUNetUp(128, 32, kernel_size=4, stride=2, padding=1)

        self.final = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
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
        u6 = nn.functional.interpolate(u5, size=(20, 17, 17), mode=interpolation_mode)

        return self.final(u6)

##############################
#        Fully Connected
##############################

class FullCon_layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, normalize=True):
        super(FullCon_layer, self).__init__()
        layers = [nn.Linear(in_size, out_size, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, stride=1, kernel_size=3):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(2, 64, stride=2,  normalization=False),
            *discriminator_block(64, 128, stride=2),
            *discriminator_block(128, 256, stride=1),
            *discriminator_block(256, 512),
            nn.Conv3d(512, 1, kernel_size=(5, 3, 3), padding=(3, 1, 2), stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_PD, img_CT):
        # Concatenate image and condition image by channels to produce input
        if img_PD.size() != img_CT.size():
            img_PD = nn.functional.interpolate(img_PD, size=img_CT.size()[2:], mode=interpolation_mode)
        img_input = torch.cat((img_PD, img_CT), 1)
        #noise = torch.normal(0, 0.25, list(img_input.size()) , device=torch.device('cuda'), requires_grad=True)
        output = self.model(img_input)#+noise)
        return output





class FullCon_Network(nn.Module):
    def __init__(self):
        super(FullCon_Network, self).__init__()

        self.step1 = FullCon_layer(49, 49, normalize=False)
        self.step2 = FullCon_layer(49, 98)
        self.step3 = FullCon_layer(98, 196)
        self.step4 = FullCon_layer(196, 392)
        self.step5 = FullCon_layer(392, 784, dropout=0.5)
        self.step6 = FullCon_layer(784, 392, dropout=0.5)
        self.step7 = FullCon_layer(392, 196, dropout=0.5)
        self.step8 = FullCon_layer(196, 98)
        self.step9 = FullCon_layer(98, 49)
        self.step10 = FullCon_layer(49, 49)

        self.final = nn.Sequential(
            nn.Linear(in_features=49, out_features=49),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        s1 = self.step1(x)
        s2 = self.step2(s1)
        s3 = self.step3(s2)
        s4 = self.step4(s3)
        s5 = self.step5(s4)
        s6 = self.step6(s5)
        s7 = self.step7(s6)
        s8 = self.step8(s7)
        s9 = self.step9(s8)
        s10 = self.step10(s9)

        s11 = self.final(s10)
        s12 = torch.reshape(s11, (-1, 1, 52, 49, 49))
        s13 = nn.functional.interpolate(s12, size=(20, 17, 17), mode=interpolation_mode)
        return s13
