import torch
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
    subsamples_spacing_calc = []
    subsamples_spacing_out = []

    # axis(Subsampled_data):[x,y,z]
    subsamples_splitting = np.split(vol.data, size, axis = 0)

    # axis(Subsampled_data):[sub_x, x, y, z]
    for i in np.arange(size):
        subsamples_splitting[i] = np.split(subsamples_splitting[i], size, axis=1)

    # axis(Subsampled_data):[sub_x, sub_y, sub_z x, y, z]
    for i in np.arange(size):
        for j in np.arange(size):
            subsamples_splitting[i][j] = np.split(subsamples_splitting[i][j], size, axis=2)

    # put the different blogs in one List, because until here it is ordered by the 3 dimensions
    for i in np.arange(size):
        for j in np.arange(size):
            for k in np.arange(size):
                subsamples_data.append(subsamples_splitting[i][j][k])
                # calculate the position, of the sub_block with step size 1
                subsamples_spacing_calc.append(np.asarray([i, j, k]))


    # calculate the Spacing in absolute numbers, the blog size is given, by the shape of vol, divided by the size cubed
    for i in range(size**3):
        subsamples_spacing_out.append(subsamples_spacing_calc[i] * np.shape(vol.data) / size)


    subsamples_data = np.asarray(subsamples_data)

    return subsamples_data, subsamples_spacing_out
