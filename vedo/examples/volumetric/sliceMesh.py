"""Slice/probe a Volume with a Mesh"""
from vedo import *

vol = Volume(dataurl+'embryo.slc').mode(1).c('bone')
msh = Paraboloid(res=200).scale(200).pos(100,100,200)

scals = probe_points(vol, msh).pointdata[0]
msh.cmap('Spectral', scals).add_scalarbar()

show(vol, msh, __doc__, axes=True).close()
