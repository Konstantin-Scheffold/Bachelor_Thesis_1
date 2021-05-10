import trimesh
from Def_functions import make_a_mesh
from Read_dicom import Data_PD
from Read_dicom import Data_CT
import scipy

shift = 1
grad = 0.375
order = 1
#make_a_mesh(Data_CT, 'CT_final.ply')
Data_rot_PD = Data_PD
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, -grad, axes = (1, 0), order = order, prefilter = False)
Data_rot_PD.data = scipy.ndimage.rotate(Data_PD.data, grad, axes = (2, 0), order = order, prefilter = False)
Data_rot_PD.data = scipy.ndimage.shift(Data_PD.data, [shift, 0, 0], order = 5, prefilter = False)

make_a_mesh(Data_rot_PD, '/Users/konstantinscheffold/Desktop/PD_shift_rot_ord{}_{}_{}.ply'.format(order, grad, shift))


# verts, faces, normals, values = measure.marching_cubes(Data_not_rot_PD.data, gradient_direction='ascent', spacing = Data_not_rot_PD.spacing)
#
# trimesh.util.attach_to_log()
#
# mesh = trimesh.Trimesh(vertices= verts,
#                        faces= faces,
#                        process=False)
#
# trimesh.exchange.export.export_mesh(mesh = mesh, file_obj = 'PD_3.ply')


#
# # some formats represent multiple meshes with multiple instances
# # the loader tries to return the datatype which makes the most sense
# # which will for scene-like files will return a `trimesh.Scene` object.
# # if you *always* want a straight `trimesh.Trimesh` you can ask the
# # loader to "force" the result into a mesh through concatenation
# ####mesh = trimesh.load('CesiumMilkTruck.glb', force='mesh')
#
# # mesh objects can be loaded from a file name or from a buffer
# # you can pass any of the kwargs for the `Trimesh` constructor
# # to `trimesh.load`, including `process=False` if you would like
# # to preserve the original loaded data without merging vertices
# # STL files will be a soup of disconnected triangles without
# # merging vertices however and will not register as watertight
# #mesh = trimesh.load('../models/featuretype.STL')
#
# # is the current mesh watertight?
# mesh.is_watertight
#
# # what's the euler number for the mesh?
# mesh.euler_number
#
# # the convex hull is another Trimesh object that is available as a property
# # lets compare the volume of our mesh with the volume of its convex hull
# print(mesh.volume / mesh.convex_hull.volume)
#
# # since the mesh is watertight, it means there is a
# # volumetric center of mass which we can set as the origin for our mesh
# mesh.vertices -= mesh.center_mass
#
# # what's the moment of inertia for the mesh?
# mesh.moment_inertia
#
# # if there are multiple bodies in the mesh we can split the mesh by
# # connected components of face adjacency
# # since this example mesh is a single watertight body we get a list of one mesh
# mesh.split()
#
# # facets are groups of coplanar adjacent faces
# # set each facet to a random color
# # colors are 8 bit RGBA by default (n, 4) np.uint8
# for facet in mesh.facets:
#     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
#
# # preview mesh in an opengl window if you installed pyglet with pip
# #mesh.show()
#
# # transform method can be passed a (4, 4) matrix and will cleanly apply the transform
# mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
#
# # axis aligned bounding box is available
# mesh.bounding_box.extents
#
# # a minimum volume oriented bounding box also available
# # primitives are subclasses of Trimesh objects which automatically generate
# # faces and vertices from data stored in the 'primitive' attribute
# mesh.bounding_box_oriented.primitive.extents
# mesh.bounding_box_oriented.primitive.transform
#
# # show the mesh appended with its oriented bounding box
# # the bounding box is a trimesh.primitives.Box object, which subclasses
# # Trimesh and lazily evaluates to fill in vertices and faces when requested
# # (press w in viewer to see triangles)
# #(mesh + mesh.bounding_box_oriented).show()
#
#
# trimesh.exchange.export.export_mesh(mesh = mesh, file_obj = 'PD_2.ply')
#
#
#
# #bounding spheres and bounding cylinders of meshes are also
# # available, and will be the minimum volume version of each
# # except in certain degenerate cases, where they will be no worse
# # than a least squares fit version of the primitive.
# print(mesh.bounding_box_oriented.volume,
#       mesh.bounding_cylinder.volume,
#       mesh.bounding_sphere.volume)