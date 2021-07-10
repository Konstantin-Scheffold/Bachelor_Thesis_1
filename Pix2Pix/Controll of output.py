import matplotlib.pyplot as plt
import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
from models_PD_to_CT import *
from datasets import *
from torch.utils.data import DataLoader

load_model = True
print_out = False

if load_model:
    Tensor = torch.cuda.FloatTensor

    val_dataloader = DataLoader(
        ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
        batch_size=8,
        shuffle=True,
    )
    generator = GeneratorWideUNet()

    generator.load_state_dict(torch.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\PDtoCT\saved_models\facades\generator_5.pth'))
    generator.cuda()
    imgs = next(iter(val_dataloader))
    real_CT = imgs["PD"].type(Tensor)
    real_PD = imgs["CT"].type(Tensor)
    fake_PD = generator(real_CT)[0].cpu().squeeze().detach().numpy()
    real_PD = real_PD[0].cpu().squeeze().detach().numpy()
    real_CT = real_CT[0].cpu().squeeze().detach().numpy()

else:
    a = np.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\CTtoPD\images\facades\4.npy', allow_pickle=True)
    real_CT = a[0][0].cpu().squeeze().detach().numpy()
    real_PD = a[1][0].cpu().squeeze().detach().numpy()
    fake_PD = a[2][0].cpu().squeeze().detach().numpy()


if print_out:
    Data_PD.data = np.array(list(real_PD), dtype=float)
    make_a_mesh(Data_PD, 'PD_real.ply', -0.1)

    Data_PD.data = np.array(list(fake_PD), dtype=float)
    make_a_mesh(Data_PD, 'PD_fake.ply', -0.1)
else:
    # plot real_CT


    plt.subplot(3, 3, 1)
    plt.title('x-axis')
    plt.ylabel('input')
    plt.imshow(real_CT[int(np.size(real_CT, 0)/2), :, :])
    plt.subplot(3, 3, 2)
    plt.title('y-axis')
    plt.imshow(real_CT[:, int(np.size(real_CT, 1)/2), :])
    plt.subplot(3, 3, 3)
    plt.title('z-axis')
    plt.imshow(real_CT[:, :, int(np.size(real_CT, 2)/2)])
    # plot real_PD
    plt.subplot(3, 3, 4)
    plt.ylabel('target')
    plt.imshow(real_PD[int(np.size(real_PD, 0)/2), :, :])
    plt.subplot(3, 3, 5)
    plt.imshow(real_PD[:, int(np.size(real_PD, 1)/2), :])
    plt.subplot(3, 3, 6)
    plt.imshow(real_PD[:, :, int(np.size(real_PD, 2)/2)])
    # plot fake_PD
    plt.subplot(3, 3, 7)
    plt.ylabel('generated')
    plt.imshow(fake_PD[int(np.size(fake_PD, 0)/2), :, :])
    plt.subplot(3, 3, 8)
    plt.imshow(fake_PD[:, int(np.size(fake_PD, 1)/2), :])
    plt.subplot(3, 3, 9)
    plt.imshow(fake_PD[:, :, int(np.size(fake_PD, 2)/2)])
    plt.show()

