from vedo import Points, Axes, show
import numpy as np

pts = np.random.randn(2000,3)*[3,2,4]-[1,2,3]
vpts1 = Points(pts).alpha(0.2).c('blue2')
vpts2 = vpts1.clone().shift(5,6,7).c('green2')

axs = Axes(
    [vpts1, vpts2],  # build axes for this set of objects
    xtitle="X-axis in \mum",
    ytitle="Variable Y in \mum",
    ztitle="_inverted Z in \mum",
    htitle='My \Gamma^2_ijk  plot',
    htitle_font='Kanopus',
    htitle_justify='bottom-right',
    htitle_color='red2',
    htitle_size=0.035,
    htitle_offset=(0,0.075,0),
    htitle_rotation=45,
    zhighlight_zero=True,
    xyframe_line=2, yzframe_line=1, zxframe_line=1,
    xyframe_color='red3',
    xyshift=1.05, # move xy 5% above the top of z-range
    yzgrid=True,
    zxgrid=True,
    zxshift=1.0,
    xtitle_justify='bottom-right',
    xtitle_offset=-1.175,
    xlabel_offset=-1.75,
    ylabel_rotation=90,
    z_inverted=True,
    tip_size=0.25,
)

show(vpts1, vpts2, axs, "Customizing Axes", viewup='z').close()
