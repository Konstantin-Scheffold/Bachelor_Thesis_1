from Def_functions import make_a_mesh
print(0)
from Read_dicom import Data_PD
from Read_dicom import Data_CT
import scipy

print(1)
shiftz = 8
gradx = -0.35
grady = 0.5
gradz = 0
order = 1
print(2)
Data_rot_CT = Data_CT
Data_rot_CT.data = scipy.ndimage.shift(Data_rot_CT.data, [shiftz, 0, 0], order = order)
print(3)
Data_rot_CT.data = scipy.ndimage.rotate(Data_rot_CT.data, gradz, axes = (1, 2), order = order)
print(4)
Data_rot_CT.data = scipy.ndimage.rotate(Data_rot_CT.data, grady, axes = (2, 0), order = order)
print(5)
#Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_CT.data, gradx, axes = (1, 0), order = order)
print(6)

# make_a_mesh(Data_rot_PD, '/Users/konstantinscheffold/Desktop/PD_ord{}_x{}°_y{}°_z{}°{}cm.ply'.format(order, gradx, grady, gradz, shiftz), 6000)
# make_a_mesh(Data_CT, '/Users/konstantinscheffold/Desktop/CT_final.ply', 40000)


