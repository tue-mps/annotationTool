"""Histogram (or plot) in 3D.
The size of each cube is proportional to the value at that point"""
import numpy as np
from vedo import Volume, Cube, Glyph, show

# Make up some arbitrary data
X, Y, Z = np.mgrid[:10, :6, :4]
counts = 50 - ( (X-4)**2 + (Y-4)**2 + (Z-4)**2 )

# This is now a point cloud with an associated array of counts
pcloud = Volume(counts).topoints()

marker = Cube().scale(0.015)
glyphed_pcl = Glyph(pcloud, marker, scale_by_scalar=True)
glyphed_pcl.cmap('seismic').add_scalarbar('counts')

show(glyphed_pcl, __doc__, axes=1)
