"""Linearly transform a Mesh by defining how a specific
set of points (landmarks) must move"""
from vedo import dataurl, Mesh, Arrows, show

# note that landmark points do not need to belong to any mesh
landmarks1 = [
    [-0.067332, 0.177376, -0.05199058],
    [-0.004541, 0.085447,  0.05713107],
    [-0.011799, 0.175825, -0.02279279],
    [-0.081910, 0.117902,  0.04889364],
]

landmarks2 = [
    [0.1287002, 0.2651531, -0.0469673],
    [0.3338593, 0.0941488,  0.1243552],
    [0.1860555, 0.2626522, -0.0202493],
    [0.1149052, 0.1731894,  0.0474256],
]

s1 = Mesh(dataurl + "bunny.obj").c("gold")
s2 = s1.clone().c('orange4')

s2.transform_with_landmarks(landmarks1, landmarks2)

arrows = Arrows(landmarks1, landmarks2, s=0.5, c='black')
show(s1, s2, arrows, __doc__, axes=True).close()
