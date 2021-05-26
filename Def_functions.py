import trimesh
from skimage import measure

def make_a_mesh(vol, name, iso):
    cm_verts, cm_faces, cm_normals, _ = measure.marching_cubes(
        vol.data, iso, gradient_direction='ascent', spacing=vol.spacing)
    cm_mesh = trimesh.base.Trimesh(vertices=cm_verts, faces=cm_faces, vertex_normals=cm_verts)
    cm_mesh.export(name)