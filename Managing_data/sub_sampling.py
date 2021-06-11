import numpy as np
import torch

print('a')
from Def_functions import make_subsamples_3d
from Def_functions import make_a_mesh
print('b')
from Get_Mesh import Data_rot_CT as Data_CT
print('c')
from Read_dicom import Data_PD
print('d')
#%%
size = 6
print('e')
# make size dividable by 4, so make_subsamples_3d can handle it, further we neede to test, on which egde to cut, the samples,
# since when done by the ursprung it shifts the Dicom picture
a, b, c = np.asarray(np.shape(Data_PD.data))%size
d, e, f = np.asarray(np.shape(Data_CT.data))%size
#print(np.asarray(np.shape(Data_PD.data))%4, '######', np.asarray(np.shape(Data_CT.data))%4)
print('f')
# here a, c are not used since in this special case it is 0 and therefore would set the dimension also to 0
Data_PD.data = Data_PD.data[:-a,:-b,:-c]
Data_CT.data = Data_CT.data[:-d,:,:-f]

print('g')
#make_a_mesh(Data_PD, 'PD_Data_resized.ply', 6000)
sub_data_PD, sub_position_PD = make_subsamples_3d(Data_PD, size)
print('h')
sub_data_CT, sub_position_CT = make_subsamples_3d(Data_CT, size)
print(np.shape(sub_data_PD))
print(np.shape(sub_data_CT))

 #%%
#make the sub sample meshes

for i in range(1):
    # save a specific subsample
    sub_data_CT_PD = np.array([sub_data_CT[i], sub_data_PD[i]], dtype = object)
    np.save('../Bachelor_Thesis/Data/Dicom_Data_edited/subsample_CTandPD_{}'.format(i), sub_data_CT_PD)

    Data_PD.data = sub_data_PD[i]
    Data_CT.data = sub_data_CT[i]
    make_a_mesh(Data_CT, '/Users/konstantinscheffold/Desktop/dataloader_test/CT_Data_unedited{}.ply'.format(i), 40000)
    make_a_mesh(Data_PD, '/Users/konstantinscheffold/Desktop/dataloader_test/PD_Data_unedited{}.ply'.format(i), 6000)

#%%
from Pix2Pix.datasets import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

transforms_ = [
    transforms.ToTensor()
]

dataloader = DataLoader(
    ImageDataset('/Users/konstantinscheffold/PycharmProjects/Bachelor_Thesis/Data/Dicom_Data_edited', transforms_ = transforms_),
    batch_size = 6,
    shuffle=True,
)

imgs = next(iter(dataloader))
real_CT = torch.squeeze(imgs["CT"]).cpu().detach().numpy()
real_PD = torch.squeeze(imgs["PD"]).cpu().detach().numpy()
print(np.shape(real_CT))
Data_PD.data = real_PD
Data_CT.data = real_CT
make_a_mesh(Data_CT, '/Users/konstantinscheffold/Desktop/dataloader_test/CT_Data_edited{}.ply'.format(i), 40000)
make_a_mesh(Data_PD, '/Users/konstantinscheffold/Desktop/dataloader_test/PD_Data_edited{}.ply'.format(i), 6000)

