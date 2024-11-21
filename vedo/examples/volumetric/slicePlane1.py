"""Slice a Volume with an arbitrary plane
hover the plane to get the scalar values"""
from vedo import dataurl, precision, Sphere, Volume, Plotter

def func(evt):
    if not evt.actor:
        return
    pid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
    txt = f"Probing:\n{precision(evt.actor.picked3d, 3)}\nvalue = {arr[pid]}"

    sph = Sphere(evt.actor.points(pid), c='orange7').pickable(False)
    fp = sph.flagpole(txt, s=7, offset=(-150,15), font=2).follow_camera()
    # remove old and add the two new objects
    plt.remove('Sphere', 'FlagPole').add(sph, fp)

vol = Volume(dataurl+'embryo.slc').alpha([0,0,0.8]).c('w').pickable(False)
vslice = vol.slice_plane(origin=vol.center(), normal=(0,1,1))
vslice.cmap('Purples_r').lighting('off').add_scalarbar('Slice', c='w')
arr = vslice.pointdata[0] # retrieve vertex array data

plt = Plotter(axes=9, bg='k', bg2='bb')
plt.add_callback('as my mouse moves please call', func) # be kind to vedo ;)
plt.show(vol, vslice, __doc__)
plt.close()
