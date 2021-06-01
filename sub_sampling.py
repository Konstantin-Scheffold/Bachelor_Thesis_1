import numpy as np
from Def_functions import make_subsamples_3d
from Def_functions import make_a_mesh
from Read_dicom import Data_PD
from Read_dicom import Data_CT
from skimage.transform import downscale_local_mean

size = 4

# make size dividable by 4
Data_PD.data = Data_PD.data[:-3,:-1,:-1]

print(np.shape(Data_PD.data))
#make_a_mesh(Data_PD, 'PD_Data_resized.ply', 6000)
z = make_subsamples_3d(Data_PD.data, size)
print(np.shape(z))

for i in range(size**3-4):
    Data_PD.data = z[i]
    make_a_mesh(Data_PD, 'PD_Data_subsample{}.ply'.format(i), 6000)
