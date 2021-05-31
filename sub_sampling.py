import numpy as np
from Def_functions import make_subsamples_3d
from Def_functions import make_a_mesh
from Read_dicom import Data_PD
from Read_dicom import Data_CT

size = 2
print_mesh = Data_PD

for i in np.arange(size**3):
    print_mesh.data = make_subsamples_3d(Data_PD.data[i], size)
    make_a_mesh(print_mesh, 'Data_PD_subsample_{}'.format(i), 6000)
