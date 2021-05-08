import numpy as np
from Read_dicom import Data_not_rot_CT
from Read_dicom import Data_not_rot_PD
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(Data_not_rot_CT.data, 0)
#vertsa, facesa, normalsa, valuesa = measure.marching_cubes(Data_not_rot_PD.data, 0)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
#mesha = Poly3DCollection(vertsa[facesa])
mesh.set_edgecolor('k')
#mesha.set_edgecolor('b')
ax.add_collection3d(mesh)
#ax.add_collection3d(mesha)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(-200, 350)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(50, 350)  # b = 10
ax.set_zlim(-350, 350)  # c = 16

plt.tight_layout()
plt.show()

