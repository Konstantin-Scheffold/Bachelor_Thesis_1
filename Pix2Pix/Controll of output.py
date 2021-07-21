import matplotlib.pyplot as plt
import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
from models import *
from datasets import *
from torch.utils.data import DataLoader

load_model = True
print_3d_file = False
print_cross_sections = True
number_check = False
name = 'WideUNet_MSE_MSE'

for i in range(5):
    if load_model:
        Tensor = torch.cuda.FloatTensor

        val_dataloader = DataLoader(
            ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
            batch_size=16,
            shuffle=True,
        )
        generator = FinalUNet()

        generator.load_state_dict(torch.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\CTtoPD\saved_models_mixed_abwechselnd\facades\generator_18.pth'))
        generator.cuda()

        imgs = next(iter(val_dataloader))
        real_PD = imgs["PD"].type(Tensor)
        real_CT = imgs["CT"].type(Tensor)
        fake_PD = generator(real_CT)[0].cpu().squeeze().detach().numpy()
        real_PD = real_PD[0].cpu().squeeze().detach().numpy()
        real_CT = real_CT[0].cpu().squeeze().detach().numpy()


    if print_3d_file:
        Data_PD.data = np.array(list(real_PD), dtype=float)
        make_a_mesh(Data_PD, '{}_real_{}.ply'.format(name, i), -0.1)

        Data_PD.data = np.array(list(fake_PD), dtype=float)
        make_a_mesh(Data_PD, '{}_fake_{}.ply'.format(name, i), -0.1)

    if print_cross_sections:

        plt.subplot(3, 3, 1)
        plt.title('z-axis')
        plt.ylabel('input')
        plt.imshow(real_CT[int(np.size(real_CT, 0)/2), :, :])

        plt.subplot(3, 3, 2)
        plt.title('x-axis')
        plt.imshow(real_CT[:, int(np.size(real_CT, 1)/2), :])

        plt.subplot(3, 3, 3)
        plt.title('y-axis')
        plt.ylabel('{}_{}'.format(name, i))
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

if number_check:
    mean_linear = list([])
    mean_cube = list([])
    for i in range(20):

        imgs = next(iter(val_dataloader))
        real_PD = imgs["PD"].type(Tensor)
        real_CT = imgs["CT"].type(Tensor)
        fake_PD = generator(real_CT)[0].cpu().squeeze().detach().numpy()
        real_PD = real_PD[0].cpu().squeeze().detach().numpy()
        real_CT = real_CT[0].cpu().squeeze().detach().numpy()
        print(i)
        mean_linear.append(np.mean(np.abs(fake_PD - real_PD)))
        mean_cube.append(np.mean(np.abs(fake_PD - real_PD)**3))

    print('#' * 10)
    print('{}'.format(name))
    print('pixel_offset_linear', np.mean(mean_linear), np.std(mean_linear))
    print('pixel_offset_cubic', np.mean(mean_cube), np.std(mean_cube))
    print('#' * 10)
