import matplotlib.pyplot as plt
import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
from models import *
from datasets import *
from torch.utils.data import DataLoader

load_model = True
print_out = False

if load_model:
    Tensor = torch.cuda.FloatTensor

    val_dataloader = DataLoader(
        ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
        batch_size=16,
        shuffle=True,
    )
    generator = GeneratorWideUNet()

    generator.load_state_dict(torch.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\CTtoPD\saved_models\facades\generator_11.pth'))
    generator.cuda()
    imgs = next(iter(val_dataloader))
    real_PD = imgs["PD"].type(Tensor)
    real_CT = imgs["CT"].type(Tensor)
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
    '''pixel_offset_linear = np.mean(np.abs(fake_PD - real_PD))
    pixel_offset_cubic = np.mean((fake_PD - real_PD) ** 2)
    a = np.array([[pixel_offset_linear], [''], [pixel_offset_cubic]], dtype=object)
    print(pixel_offset_linear, pixel_offset_cubic)
    plt.subplot(3, 4, 4)
    plt.axes(frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.table(a, loc='center right', edges='horizontal', colWidths=[0.5, 0.5, 0.5], cellLoc='center',
              rowLabels=['pixel offset linear', '', 'pixel offset cubic'])
'''
    plt.subplot(3, 4, 1)
    plt.title('z-axis')
    plt.ylabel('input')
    plt.imshow(real_CT[int(np.size(real_CT, 0)/2), :, :])

    plt.subplot(3, 4, 2)
    plt.title('x-axis')
    plt.imshow(real_CT[:, int(np.size(real_CT, 1)/2), :])

    plt.subplot(3, 4, 3)
    plt.title('y-axis')
    plt.imshow(real_CT[:, :, int(np.size(real_CT, 2)/2)])

    # plot real_PD
    plt.subplot(3, 4, 5)
    plt.ylabel('target')
    plt.imshow(real_PD[int(np.size(real_PD, 0)/2), :, :])

    plt.subplot(3, 4, 6)
    plt.imshow(real_PD[:, int(np.size(real_PD, 1)/2), :])

    plt.subplot(3, 4, 7)
    plt.imshow(real_PD[:, :, int(np.size(real_PD, 2)/2)])

    # plot fake_PD
    plt.subplot(3, 4, 9)
    plt.ylabel('generated')
    plt.imshow(fake_PD[int(np.size(fake_PD, 0)/2), :, :])

    plt.subplot(3, 4, 10)
    plt.imshow(fake_PD[:, int(np.size(fake_PD, 1)/2), :])

    plt.subplot(3, 4, 11)
    plt.imshow(fake_PD[:, :, int(np.size(fake_PD, 2)/2)])
    plt.show()

