import matplotlib.pyplot as plt
import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
from models import *
from datasets import *
from torch.utils.data import DataLoader


print_3d_file = True
print_cross_sections = True
number_check = False
name = 'ideal_MSE_L1'


Tensor = torch.FloatTensor

generator = FinalUNet()
generator.load_state_dict(torch.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\CTtoPD\saved_models\facades\generator_8.pth', map_location=torch.device('cpu')))

if print_cross_sections or print_3d_file:
    val_dataloader = DataLoader(
        ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
        batch_size=16,
        shuffle=True,
    )

    imgs = next(iter(val_dataloader))
    real_PD = imgs["PD"].type(Tensor)
    real_CT = imgs["CT"].type(Tensor)
    fake_PD_batch = generator(real_CT).squeeze().detach().numpy()
    real_PD_batch = real_PD.squeeze().detach().numpy()
    real_CT_batch = real_CT.squeeze().detach().numpy()

for i in range(15):

    if print_3d_file:
        fake_PD = fake_PD_batch[i]
        real_PD = real_PD_batch[i]
        real_CT = real_CT_batch[i]
        Data_PD.data = np.array(list(real_PD), dtype=float)
        make_a_mesh(Data_PD, r'C:\Users\Konra\OneDrive\Desktop\photos\ideal_MSE_L1\{}_real_{}.ply'.format(name, i), np.mean(Data_PD.data)-0.1)#-0.1)

        Data_PD.data = np.array(list(fake_PD), dtype=float)
        make_a_mesh(Data_PD, r'C:\Users\Konra\OneDrive\Desktop\photos\ideal_MSE_L1\{}_fake_{}.ply'.format(name, i), np.mean(Data_PD.data)-0.1)#-0.1)


    if print_cross_sections:
        fake_PD = fake_PD_batch[i]
        real_PD = real_PD_batch[i]
        real_CT = real_CT_batch[i]
        plt.subplot(3, 3, 1)
        plt.title('z-axis')
        plt.ylabel('input')
        plt.imshow(real_CT[int(np.size(real_CT, 0)/2), :, :])

        plt.subplot(3, 3, 2)
        plt.title('x-axis')
        plt.ylabel('input')
        plt.imshow(real_CT[:, int(np.size(real_CT, 1)/2), :])

        plt.subplot(3, 3, 3)
        plt.title('y-axis')
        plt.ylabel('input')
        #plt.ylabel('{}_{}'.format(name, i))
        plt.imshow(real_CT[:, :, int(np.size(real_CT, 2)/2)])

        # plot real_PD
        plt.subplot(3, 3, 4)
        plt.ylabel('target')
        plt.imshow(real_PD[int(np.size(real_PD, 0)/2), :, :])

        plt.subplot(3, 3, 5)
        plt.ylabel('target')
        plt.imshow(real_PD[:, int(np.size(real_PD, 1)/2), :])

        plt.subplot(3, 3, 6)
        plt.ylabel('target')
        plt.imshow(real_PD[:, :, int(np.size(real_PD, 2)/2)])

        # plot fake_PD
        plt.subplot(3, 3, 7)
        plt.ylabel('generated')
        plt.imshow(fake_PD[int(np.size(fake_PD, 0)/2), :, :])

        plt.subplot(3, 3, 8)
        plt.ylabel('generated')
        plt.imshow(fake_PD[:, int(np.size(fake_PD, 1)/2), :])

        plt.subplot(3, 3, 9)
        plt.ylabel('generated')
        plt.imshow(fake_PD[:, :, int(np.size(fake_PD, 2)/2)])
        plt.savefig(r'C:\Users\Konra\OneDrive\Desktop\photos\ideal_MSE_L1\{}_{}'.format(name, i))
        #plt.show()

if number_check:

    val_dataloader = DataLoader(
        ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
        batch_size=50,
        shuffle=True,
    )

    imgs = next(iter(val_dataloader))
    real_PD = imgs["PD"].type(Tensor)
    real_CT = imgs["CT"].type(Tensor)
    fake_PD_batch = generator(real_CT).squeeze().detach().numpy()
    real_PD_batch = real_PD.squeeze().detach().numpy()
    real_CT_batch = real_CT.squeeze().detach().numpy()
    mean_linear = list([])
    mean_cube = list([])

    for i in range(50):
        fake_PD = fake_PD_batch[i]
        real_PD = real_PD_batch[i]
        real_CT = real_CT_batch[i]

        mean_linear.append(np.mean(np.abs(fake_PD - real_PD)))
        mean_cube.append(np.mean(np.abs(fake_PD - real_PD)**3))

    print('#' * 10)
    print('{}'.format(name))
    print('pixel_offset_linear', np.mean(mean_linear), np.std(mean_linear))
    print('pixel_offset_cubic', np.mean(mean_cube), np.std(mean_cube))
    print('#' * 10)
