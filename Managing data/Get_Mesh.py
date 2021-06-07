from Def_functions import make_a_mesh
from Read_dicom import Data_PD
from Read_dicom import Data_CT
import scipy


shiftz = 8
gradx = -0.35
grady = 0.5
gradz = 0
order = 5

Data_rot_PD = Data_PD
Data_rot_PD.data = scipy.ndimage.shift(Data_rot_PD.data, [shiftz, 0, 0], order = order)
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, gradz, axes = (1, 2), order = order, reshape = False)
Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_PD.data, grady, axes = (2, 0), order = order, reshape = False)
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, gradx, axes = (1, 0), order = order, reshape = False)

make_a_mesh(Data_rot_PD, '/Users/konstantinscheffold/Desktop/PD_ord{}_x{}°_y{}°_z{}°{}cm.ply'.format(order, gradx, grady, gradz, shiftz), 6000)
make_a_mesh(Data_CT, '/Users/konstantinscheffold/Desktop/CT_final.ply', 40000)

