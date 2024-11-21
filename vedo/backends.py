#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import settings
from vedo import colors
from vedo import utils
from vedo import shapes
from vedo.pointcloud import Points
from vedo.mesh import Mesh
from vedo.volume import Volume

__doc__ = """Submodule to delegate jupyter notebook rendering"""

__all__ = []


############################################################################################
def get_notebook_backend(actors2show=()):
    """Return the appropriate notebook viewer"""

    plt = vedo.plotter_instance
    if plt:
        if isinstance(plt.shape, str) or sum(plt.shape) > 2 and settings.default_backend != "2d":
            vedo.logger.error(
                f"Multirendering is not supported for jupyter backend: {settings.default_backend}"
            )

    #########################################
    if settings.default_backend == "2d":
        return start_2d()

    #########################################
    if settings.default_backend.startswith("itk"):
        return start_itkwidgets(actors2show)

    #########################################
    if settings.default_backend == "k3d":
        return start_k3d(actors2show)

    #########################################
    if settings.default_backend == "panel":
        return start_panel()

    #########################################
    if settings.default_backend.startswith("ipyvtk"):
        return start_ipyvtklink()

    #########################################
    if settings.default_backend == "ipygany":
        return start_ipygany(actors2show)

    #########################################
    # if settings.default_backend.startswith("pythree"): #todo
    #     return start_pythreejs(actors2show)

    vedo.logger.error(f"Unknown jupyter backend: {settings.default_backend}")
    return None


#####################################################################################
def start_2d():

    try:
        import PIL.Image
        import IPython
    except ImportError("PIL or IPython not available"):
        return None

    plt = vedo.plotter_instance

    if hasattr(plt, "window") and plt.window:
        if plt.renderer == plt.renderers[-1]:
            nn = vedo.io.screenshot(asarray=True, scale=settings.screeshot_scale)
            pil_img = PIL.Image.fromarray(nn)
            notebook_plotter = IPython.display.display(pil_img)
            vedo.notebook_plotter = notebook_plotter
            plt.close()
            return notebook_plotter


####################################################################################
def start_itkwidgets(actors2show):
    # https://github.com/InsightSoftwareConsortium/itkwidgets
    #  /blob/master/itkwidgets/widget_viewer.py
    try:
        from itkwidgets import view
    except ModuleNotFoundError("Cannot find itkwidgets, try:\n> pip install itkwidgets"):
        return None

    vedo.notebook_plotter = view(
        actors=actors2show,
        cmap="jet",
        ui_collapsed=True,
        gradient_opacity=False,
    )
    vedo.plotter_instance.close()
    return vedo.notebook_plotter


####################################################################################
def start_k3d(actors2show):
    # https://github.com/K3D-tools/K3D-jupyter
    try:
        import k3d
        if str(k3d.__version__) != "2.7.4":
            vedo.logger.warning("Only k3d version 2.7.4 is currently supported")
    except ModuleNotFoundError("Cannot find k3d, install with:  pip install k3d==2.7.4"):
        return None

    plt = vedo.plotter_instance

    actors2show2 = []
    for ia in actors2show:
        if not ia:
            continue
        if isinstance(ia, vtk.vtkAssembly):  # unpack assemblies
            acass = ia.unpack()
            actors2show2 += acass
        else:
            actors2show2.append(ia)

    vedo.notebook_plotter = k3d.plot(
        axes=["x", "y", "z"],
        menu_visibility=settings.k3d_menu_visibility,
        height=settings.k3d_plot_height,
        antialias=settings.k3d_antialias,
    )
    # vedo.notebook_plotter.grid = kgrid
    vedo.notebook_plotter.lighting = settings.k3d_lighting

    # set k3d camera
    vedo.notebook_plotter.camera_auto_fit = settings.k3d_camera_autofit
    vedo.notebook_plotter.grid_auto_fit = settings.k3d_grid_autofit
    vedo.notebook_plotter.axes_helper = settings.k3d_axes_helper

    if plt and plt.camera:
        k3dc = utils.vtkCameraToK3D(plt.camera)
        vedo.notebook_plotter.camera = k3dc

    if plt and not plt.axes:
        vedo.notebook_plotter.grid_visible = False

    for ia in actors2show2:

        if isinstance(ia, (vtk.vtkCornerAnnotation, vtk.vtkAssembly)):
            continue

        kobj = None
        kcmap = None
        name = None
        if hasattr(ia, "filename"):
            if ia.filename:
                name = os.path.basename(ia.filename)
            if ia.name:
                name = os.path.basename(ia.name)

        #####################################################################scalars
        # work out scalars first, Points Lines are also Mesh objs
        if isinstance(ia, (Mesh, shapes.Line, Points)):
            # print('scalars', ia.name, ia.npoints)
            iap = ia.GetProperty()

            if isinstance(ia, (shapes.Line, Points)):
                iapoly = ia.polydata()
            else:
                iapoly = ia.clone().clean().triangulate().compute_normals().polydata()

            vtkscals = None
            color_attribute = None
            if ia.mapper().GetScalarVisibility():
                vtkdata = iapoly.GetPointData()
                vtkscals = vtkdata.GetScalars()

                if vtkscals is None:
                    vtkdata = iapoly.GetCellData()
                    vtkscals = vtkdata.GetScalars()
                    if vtkscals is not None:
                        c2p = vtk.vtkCellDataToPointData()
                        c2p.SetInputData(iapoly)
                        c2p.Update()
                        iapoly = c2p.GetOutput()
                        vtkdata = iapoly.GetPointData()
                        vtkscals = vtkdata.GetScalars()

                if vtkscals is not None:
                    if not vtkscals.GetName():
                        vtkscals.SetName("scalars")
                    scals_min, scals_max = ia.mapper().GetScalarRange()
                    color_attribute = (vtkscals.GetName(), scals_min, scals_max)
                    lut = ia.mapper().GetLookupTable()
                    lut.Build()
                    kcmap = []
                    nlut = lut.GetNumberOfTableValues()
                    for i in range(nlut):
                        r, g, b, _ = lut.GetTableValue(i)
                        kcmap += [i / (nlut - 1), r, g, b]

        #####################################################################Volume
        if isinstance(ia, Volume):
            # print('Volume', ia.name, ia.dimensions())
            kx, ky, _ = ia.dimensions()
            arr = ia.pointdata[0]
            kimage = arr.reshape(-1, ky, kx)

            colorTransferFunction = ia.GetProperty().GetRGBTransferFunction()
            kcmap = []
            for i in range(128):
                r, g, b = colorTransferFunction.GetColor(i / 127)
                kcmap += [i / 127, r, g, b]

            kbounds = numpy.array(ia.imagedata().GetBounds()) + numpy.repeat(
                numpy.array(ia.imagedata().GetSpacing()) / 2.0, 2
            ) * numpy.array([-1, 1] * 3)

            kobj = k3d.volume(
                kimage.astype(numpy.float32),
                color_map=kcmap,
                # color_range=ia.imagedata().GetScalarRange(),
                alpha_coef=10,
                bounds=kbounds,
                name=name,
            )
            vedo.notebook_plotter += kobj

        #####################################################################text
        elif hasattr(ia, "info") and "formula" in ia.info.keys():
            pos = (ia.GetPosition()[0], ia.GetPosition()[1])
            kobj = k3d.text2d(ia.info["formula"], position=pos)
            vedo.notebook_plotter += kobj

        #####################################################################Mesh
        elif isinstance(ia, Mesh) and ia.npoints and len(ia.faces()):
            # print('Mesh', ia.name, ia.npoints, len(ia.faces()))
            kobj = k3d.vtk_poly_data(
                iapoly,
                name=name,
                # color=_rgb2int(iap.GetColor()),
                color_attribute=color_attribute,
                color_map=kcmap,
                opacity=iap.GetOpacity(),
                wireframe=(iap.GetRepresentation() == 1),
            )

            if iap.GetInterpolation() == 0:
                kobj.flat_shading = True
            vedo.notebook_plotter += kobj

        #####################################################################Points
        elif isinstance(ia, Points):
            # print('Points', ia.name, ia.npoints)
            kcols = []
            if color_attribute is not None:
                scals = utils.vtk2numpy(vtkscals)
                kcols = k3d.helpers.map_colors(scals, kcmap, [scals_min, scals_max]).astype(
                    numpy.uint32
                )
            # sqsize = numpy.sqrt(numpy.dot(sizes, sizes))

            kobj = k3d.points(
                ia.points().astype(numpy.float32),
                color=_rgb2int(iap.GetColor()),
                colors=kcols,
                opacity=iap.GetOpacity(),
                shader=settings.k3d_point_shader,
                point_size=iap.GetPointSize(),
                name=name,
            )
            vedo.notebook_plotter += kobj

        #####################################################################Lines
        elif isinstance(ia, vedo.Picture):
            vedo.logger.error("Sorry Picture objects are not supported in k3d.")

        #####################################################################Lines
        elif ia.polydata(False).GetNumberOfLines():
            # print('Line', ia.name, ia.npoints, len(ia.faces()),
            #       ia.polydata(False).GetNumberOfLines(), len(ia.lines()),
            #       color_attribute, [vtkscals])
            # kcols=[]
            # if color_attribute is not None:
            #     scals = utils.vtk2numpy(vtkscals)
            #     kcols = k3d.helpers.map_colors(scals, kcmap,
            #                                    [scals_min,scals_max]).astype(numpy.uint32)
            # sqsize = numpy.sqrt(numpy.dot(sizes, sizes))

            for i, ln_idx in enumerate(ia.lines()):
                if i > 200:
                    print("WARNING: K3D nr of line segments is limited to 200.")
                    break
                pts = ia.points()[ln_idx]
                kobj = k3d.line(
                    pts.astype(numpy.float32),
                    color=_rgb2int(iap.GetColor()),
                    opacity=iap.GetOpacity(),
                    shader=settings.k3d_line_shader,
                    # width=iap.GetLineWidth()*sqsize/1000,
                    name=name,
                )
                vedo.notebook_plotter += kobj
    if plt:
        plt.close()
    return vedo.notebook_plotter


#####################################################################################
def start_panel():

    try:
        import panel  # https://panel.pyviz.org/reference/panes/VTK.html
        panel.extension("vtk")
    except ImportError("panel is not installed, try:\n> pip install panel"):
        return None

    plt = vedo.plotter_instance

    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        vedo.notebook_plotter = panel.pane.VTK(
            plt.window,
            width=int(plt.size[0] / 1.5),
            height=int(plt.size[1] / 2),
        )
        return vedo.notebook_plotter
    vedo.logger.error("No window present for panel backend.")
    return None


#####################################################################################
def start_ipyvtklink():

    try:
        from ipyvtklink.viewer import ViewInteractiveWidget
    except ImportError("ipyvtklink is not installed, try:\n> pip install ipyvtklink"):
        return None

    plt = vedo.plotter_instance
    if hasattr(plt, "window") and plt.window:
        plt.renderer.ResetCamera()
        vedo.notebook_plotter = ViewInteractiveWidget(
            plt.window,
            allow_wheel=True,
            quality=100,
            quick_quality=50,
        )
        return vedo.notebook_plotter
    vedo.logger.error("No window present for panel backend.")
    return None


#####################################################################################
def start_ipygany(actors2show):
    try:
        from ipygany import PolyMesh, Scene, IsoColor, RGB, Component
        from ipygany import Alpha, ColorBar, colormaps, PointCloud
        from ipywidgets import FloatRangeSlider, Dropdown, VBox, AppLayout, jslink
    except ImportError("ipygany is not installed, try:\n> pip install ipygany"):
        return None

    plt = vedo.plotter_instance

    bgcol = colors.rgb2hex(colors.get_color(plt.backgrcol))

    actors2show2 = []
    for ia in actors2show:
        if not ia:
            continue
        if isinstance(ia, vedo.Assembly):  # unpack assemblies
            assacts = ia.unpack()
            for ja in assacts:
                if isinstance(ja, vedo.Assembly):
                    actors2show2 += ja.unpack()
                else:
                    actors2show2.append(ja)
        else:
            actors2show2.append(ia)

    pmeshes = []
    colorbar = None
    for obj in actors2show2:
        #            print("ipygany processing:", [obj], obj.name)
        if isinstance(obj, vedo.shapes.Line):
            lg = obj.diagonal_size() / 1000 * obj.GetProperty().GetLineWidth()
            vmesh = vedo.shapes.Tube(obj.points(), r=lg, res=4).triangulate()
            vmesh.c(obj.c())
            faces = vmesh.faces()
            # todo: Lines
        elif isinstance(obj, Mesh):
            vmesh = obj.triangulate()
            faces = vmesh.faces()
        elif isinstance(obj, Points):
            vmesh = obj
            faces = []
        elif isinstance(obj, Volume):
            vmesh = obj.isosurface()
            faces = vmesh.faces()
        elif isinstance(obj, vedo.TetMesh):
            vmesh = obj.tomesh(fill=False)
            faces = vmesh.faces()
        else:
            print("ipygany backend: cannot process object type", [obj])
            continue

        vertices = vmesh.points()
        scals = vmesh.inputdata().GetPointData().GetScalars()
        if scals and not colorbar:  # there is an active array, only pick the first
            aname = scals.GetName()
            arr = vmesh.pointdata[aname]
            parr = Component(name=aname, array=arr)
            if len(faces):
                pmesh = PolyMesh(vertices=vertices, triangle_indices=faces, data={aname: [parr]})
            else:
                pmesh = PointCloud(vertices=vertices, data={aname: [parr]})
            rng = scals.GetRange()
            colored_pmesh = IsoColor(pmesh, input=aname, min=rng[0], max=rng[1])
            if obj.scalarbar:
                colorbar = ColorBar(colored_pmesh)
                colormap_slider_range = FloatRangeSlider(
                    value=rng,
                    min=rng[0],
                    max=rng[1],
                    step=(rng[1] - rng[0]) / 100.0,
                )
                jslink((colored_pmesh, "range"), (colormap_slider_range, "value"))
                colormap = Dropdown(options=colormaps, description="Colormap:")
                jslink((colored_pmesh, "colormap"), (colormap, "index"))

        else:
            if len(faces):
                pmesh = PolyMesh(vertices=vertices, triangle_indices=faces)
            else:
                pmesh = PointCloud(vertices=vertices)
            if vmesh.alpha() < 1:
                colored_pmesh = Alpha(RGB(pmesh, input=tuple(vmesh.color())), input=vmesh.alpha())
            else:
                colored_pmesh = RGB(pmesh, input=tuple(vmesh.color()))

        pmeshes.append(colored_pmesh)

    if colorbar:
        scene = AppLayout(
            left_sidebar=Scene(pmeshes, background_color=bgcol),
            right_sidebar=VBox((colormap_slider_range, colorbar, colormap)),  # not working
            pane_widths=[2, 0, 1],
        )
    else:
        scene = Scene(pmeshes, background_color=bgcol)

    vedo.notebook_plotter = scene
    plt.close()
    return scene


#####################################################################################
def _rgb2int(rgb_tuple):
    # Return the int number of a color from (r,g,b), with 0<r<1 etc.
    rgb = (int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))
    return 65536 * rgb[0] + 256 * rgb[1] + rgb[2]
