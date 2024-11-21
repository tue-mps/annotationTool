"""Lock an object orientation
to constantly face the scene camera"""
from vedo import *

sp = Sphere().wireframe()

tx1 = Text3D("Fixed Text", sp.points(10), s=0.07, depth=0.1, c="lb")

tx2 = Text3D("Follower Text", sp.points(144), s=0.07, c="lg")
tx2.follow_camera() # a vtkCamera can also be passed as argument

fp = sp.flagpole("The\nNorth Pole", c='k6', rounded=True)
fp.scale(0.4).follow_camera()

show(sp, tx1, tx2, fp, __doc__, bg='bb', axes=1).close()
