"""Create a Volume
from a numpy object using imread"""
from vedo import *
from skimage.io import imread


f = dataurl+'embryo.tif'

voriginal = Volume(f)
printc('voxel size is', voriginal.spacing(), c='cyan')

raw = imread(f)
narray = np.transpose(raw, axes=[2, 1, 0])

vraw = Volume(narray, spacing=(104,104,104))

# Compare loading the volume directly with the numpy volume:
# they should be the same
show(
    [(voriginal, __doc__),
    (vraw,"From imread\n(should be same as left)")],
    N=2,
    axes=1,
).close()

