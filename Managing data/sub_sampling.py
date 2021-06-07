import numpy as np
from Def_functions import make_subsamples_3d
from Def_functions import make_a_mesh
from Get_Mesh import Data_rot_PD as Data_PD
from Read_dicom import Data_CT
from skimage.transform import downscale_local_mean
size = 4

# make size dividable by 4, so make_subsamples_3d can handle it, further we neede to test, on which egde to cut, the samples,
# since when done by the ursprung it shifts the Dicom picture
a, b, c = np.asarray(np.shape(Data_PD.data))%size
d, e, f = np.asarray(np.shape(Data_CT.data))%size
#print(np.asarray(np.shape(Data_PD.data))%4, '######', np.asarray(np.shape(Data_CT.data))%4)

# here a, c are not used since in this special case it is 0 and therefore would set the dimension also to 0
Data_PD.data = Data_PD.data[:,:-b,:]
Data_CT.data = Data_CT.data[:-d,:-e,:-f]

#make_a_mesh(Data_PD, 'PD_Data_resized.ply', 6000)
sub_data_PD, sub_position_PD = make_subsamples_3d(Data_PD, size)
sub_data_CT, sub_position_CT = make_subsamples_3d(Data_CT, size)
print(np.shape(sub_data_PD))
print(np.shape(sub_data_CT))

#%%
# make the sub sample meshes
for i in [0,1,20,40]:
    # set a specific sub sample, as the vol.data
    Data_PD.data = sub_data_PD[i]
    Data_CT.data = sub_data_CT[i]

# makes in this case for marching cubes no difference to calculate the relative position, since vol.position isnt use in marching cubes
#    Data_PD.position = -sub_position[i]*1000
    make_a_mesh(Data_CT, '/Users/konstantinscheffold/Desktop/working samples/CT_Data_subsample{}.ply'.format(i), 40000)
    make_a_mesh(Data_PD, '/Users/konstantinscheffold/Desktop/working samples/PD_Data_subsample{}.ply'.format(i), 6000)
