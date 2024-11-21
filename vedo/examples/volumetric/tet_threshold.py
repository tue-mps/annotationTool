"""Threshold the original TetMesh
with a scalar array"""
from vedo import *

settings.use_depth_peeling = True

tetm = TetMesh(dataurl+'limb_ugrid.vtk')
tetm.color('prism').alpha([0,1])

# Threshold the tetrahedral mesh for values in the range:
tetm.threshold(above=0.9, below=1)
tetm.add_scalarbar3d('chem_0  expression levels', c='k', italic=1)

show([(tetm,__doc__),
       tetm.tomesh(shrink=0.9),
     ], N=2, axes=1,
).close()
