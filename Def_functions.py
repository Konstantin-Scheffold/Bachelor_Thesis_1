import trimesh
from skimage import measure
import numpy as np


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
    for i in np.arange(size):
        for j in np.arange(size):
            for k in np.arange(size):
                subsamples_data.append(subsamples_splitting[i][j][k])

    subsamples_data = np.asarray(subsamples_data, dtype=object)

    return subsamples_data

# a = np.arange(364*308*308).reshape((364 , 308, 308))
# a = make_subsamples_3d(a, 2)
# print(np.shape(a))