import trimesh
from skimage import measure
import numpy as np
from Read_dicom import Data_PD

def make_a_mesh(vol, name, iso):
    cm_verts, cm_faces, cm_normals, _ = measure.marching_cubes(
        vol.data, iso, gradient_direction='ascent', spacing=vol.spacing)
    cm_mesh = trimesh.base.Trimesh(vertices=cm_verts, faces=cm_faces, vertex_normals=cm_verts)
    cm_mesh.export(name)

def make_subsamples_3d(vol, size):
    subsamples_data = []
    # axis(Subsampled_data):[x,y,z]
    subsamples_splitting = np.array_split(vol, size, axis = 0)
    # axis(Subsampled_data):[sub_x, x, y, z]

    for i in np.arange(size):
        subsamples_splitting[i] = np.array_split(subsamples_splitting[i], size, axis=1)
    # axis(Subsampled_data):[sub_x, sub_y, sub_z x, y, z]

    for i in np.arange(size):
        for j in np.arange(size):
            subsamples_splitting[i][j] = np.array_split(subsamples_splitting[i][j], size, axis=2)
    # axis(Subsampled_data):[sub_x, sub_y, sub_z x, y, z]
    print(np.shape(subsamples_splitting))
    for i in np.arange(size):
        for j in np.arange(size):
            for k in np.arange(size):
                subsamples_data.append(subsamples_splitting[i][j][k])

    return subsamples_data


# print(np.shape(Data_PD.data))
# z = make_subsamples_3d(Data_PD.data, 2)
# print(np.shape(z))