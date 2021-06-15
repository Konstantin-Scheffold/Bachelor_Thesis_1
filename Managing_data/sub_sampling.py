import numpy as np
import torch

print('a')
from Managing_data.Def_functions import make_subsamples_3d
from Managing_data.Def_functions import make_a_mesh
print('b')
from Managing_data.Get_Mesh import Data_rot_PD as Data_PD
print('c')
from Managing_data.Read_dicom import Data_CT
print('d')
#%%
size = 11
print('e')
# make size dividable by 4, so make_subsamples_3d can handle it, further we neede to test, on which egde to cut, the samples,
# since when done by the ursprung it shifts the Dicom picture
a, b, c = np.asarray(np.shape(Data_PD.data))%size
d, e, f = np.asarray(np.shape(Data_CT.data))%size

print('f')
# here a, c are not used since in this special case it is 0 and therefore would set the dimension also to 0
Data_PD.data = Data_PD.data[:-a, :-b, :-c]
Data_CT.data = Data_CT.data[:-d, :-e, :-f]

print('g')

sub_data_PD, sub_position_PD = make_subsamples_3d(Data_PD, size)
print('h')
sub_data_CT, sub_position_CT = make_subsamples_3d(Data_CT, size)
print(np.shape(sub_data_PD))
print(np.shape(sub_data_CT))

 #%%

#make the sub sample meshes
#
# for i in range(int(size**3 * 2/3)):
#     # save a specific subsample
#     sub_data_CT_PD = np.array([sub_data_CT[i], sub_data_PD[i]], dtype = object)
#     np.save('../Bachelor_Thesis/Data/Dicom_Data_edited/train/subsample_CTandPD_{}'.format(i), sub_data_CT_PD)
#
# for i in range(int(size**3 * 2/3), size**3):
#     # save a specific subsample
#     sub_data_CT_PD = np.array([sub_data_CT[i], sub_data_PD[i]], dtype = object)
#     np.save('../Bachelor_Thesis/Data/Dicom_Data_edited/val/subsample_CTandPD_{}'.format(i), sub_data_CT_PD)


#%%

a = np.load('../Bachelor_Thesis/Data/Dicom_Data_edited/train/subsample_CTandPD_1')