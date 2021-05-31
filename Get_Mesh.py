import trimesh
import numpy as np
from Def_functions import make_a_mesh
from Read_dicom import Data_PD
from Read_dicom import Data_CT
import scipy
from skimage.transform import downscale_local_mean

shiftz = 8
gradx = -0.35
grady = 0.5
gradz = 0
order = 1

Data_rot_PD = Data_PD
Data_rot_PD.data = scipy.ndimage.shift(Data_rot_PD.data, [shiftz, 0, 0], order = order)
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, gradz, axes = (1, 2), order = order)
Data_rot_PD.data = scipy.ndimage.rotate(Data_rot_PD.data, grady, axes = (2, 0), order = order)
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, gradx, axes = (1, 0), order = order)

make_a_mesh(Data_PD, '/Users/konstantinscheffold/Desktop/PD_ord{}_x{}°_y{}°_z{}°{}cm.ply'.format(order, gradx, grady, gradz, shiftz), 6000)

# rescale the CT Array

make_a_mesh(Data_CT, 'CT_final.ply', 40000)
# CT_resize = Data_CT
# a, b, c = np.size(Data_PD.data,0)/np.size(Data_CT.data,0), np.size(Data_PD.data,1)/np.size(Data_CT.data,1), np.size(Data_PD.data,2)/np.size(Data_CT.data,2)
#
# CT_resize.data = downscale_local_mean(CT_resize.data, ((int(a), int(b), int(c))))
# CT_resize.spacing =  Data_PD.spacing
#
# make_a_mesh(CT_resize, 'CT_final_res_spacing.ply', 40000)


