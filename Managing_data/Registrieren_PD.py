from Managing_data.Def_functions import make_a_mesh
print(0)
from Managing_data.Read_dicom import Data_PD
import scipy

print(1)
shiftz = 11
shifty = 0.5
gradx = -0.35
grady = 0.2
order = 5
print(2)
Data_rot_PD = Data_PD
Data_rot_PD.data = scipy.ndimage.shift(Data_rot_PD.data, [shiftz, 0, shifty], order=order)
print(3)
#Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_CT.data, gradz, axes = (1, 2), order = order, reshape = False)
print(4)
Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_PD.data, grady, axes = (2, 0), order = order, reshape = False)
print(5)
Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_PD.data, gradx, axes = (1, 0), order = order, reshape = False)
print(6)



