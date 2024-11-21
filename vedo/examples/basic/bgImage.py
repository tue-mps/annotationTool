"""
Set a jpeg background image
on a separate rendering layer
"""
from vedo import *

plt = Plotter(
    N=4,
    sharecam=False,
	bg=dataurl+"images/tropical.jpg",
	bg2='light blue',
)

a1 = load(dataurl+"flamingo.3ds").rotate_x(-90)

plt.at(2).show(__doc__)

# after first rendering, picture can be zoomed to fill the window:
plt.background_renderer.GetActiveCamera().Zoom(1.8)
plt.at(0).show(VedoLogo(distance=2))
plt.at(3).show(a1)
plt.interactive().close()
