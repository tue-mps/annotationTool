"""Use sliders to slice volume
Click button to change colormap"""
from vedo import dataurl, Volume, Text2D
from vedo.applications import Slicer3DPlotter

filename = dataurl + "embryo.slc"
# filename = dataurl+'embryo.tif'
# filename = dataurl+'vase.vti'

vol = Volume(filename)#.print()

plt = Slicer3DPlotter(
    vol,
    bg="white",
    bg2="lightblue",
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    use_slider3d=False,
)

# Can now add any other object to the Plotter scene:
# plt += Text2D('some message')
# plt.show().interactive()

plt.close()
