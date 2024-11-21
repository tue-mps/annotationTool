import glob
import os
from deprecated import deprecated

import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import utils
from vedo.base import Base3DProp
from vedo.base import BaseGrid
from vedo.mesh import Mesh

__doc__ = """
Work with volumetric datasets (voxel data) <br>
.. image:: https://vedo.embl.es/images/volumetric/slicePlane2.png
"""

__all__ = [
    "BaseVolume",  # included to generate documentation in pydoc
    "Volume",
    "VolumeSlice",
]


##########################################################################
class BaseVolume:
    """Base class. Do not instantiate."""

    def __init__(self):
        self._data = None
        self._mapper = None
        self.transform = None

    def _update(self, img):
        self._data = img
        self._data.GetPointData().Modified()
        self._mapper.SetInputData(img)
        self._mapper.Modified()
        self._mapper.Update()
        return self

    def clone(self):
        """Return a clone copy of the Volume."""
        newimg = vtk.vtkImageData()
        newimg.CopyStructure(self._data)
        newimg.CopyAttributes(self._data)
        newvol = Volume(newimg)
        prop = vtk.vtkVolumeProperty()
        prop.DeepCopy(self.GetProperty())
        newvol.SetProperty(prop)
        newvol.SetOrigin(self.GetOrigin())
        newvol.SetScale(self.GetScale())
        newvol.SetOrientation(self.GetOrientation())
        newvol.SetPosition(self.GetPosition())
        return newvol

    def imagedata(self):
        """Return the underlying `vtkImagaData` object."""
        return self._data

    def tonumpy(self):
        """
        Get read-write access to voxels of a Volume object as a numpy array.

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.

        Example: `arr[:] = arr*2 + 15`

        If the array is modified call:

        *volume.imagedata().GetPointData().GetScalars().Modified()*

        when all your modifications are completed.
        """
        narray_shape = tuple(reversed(self._data.GetDimensions()))

        scals = self._data.GetPointData().GetScalars()
        comps = scals.GetNumberOfComponents()
        if comps == 1:
            narray = utils.vtk2numpy(scals).reshape(narray_shape)
            narray = np.transpose(narray, axes=[2, 1, 0])
        else:
            narray = utils.vtk2numpy(scals).reshape(*narray_shape, comps)
            narray = np.transpose(narray, axes=[2, 1, 0, 3])

        # narray = utils.vtk2numpy(self._data.GetPointData().GetScalars()).reshape(narray_shape)
        # narray = np.transpose(narray, axes=[2, 1, 0])

        return narray

    def dimensions(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return np.array(self._data.GetDimensions())

    @deprecated(reason=vedo.colors.red + "Please use scalar_range()" + vedo.colors.reset)
    def scalarRange(self):
        "Deprecated. Please use scalar_range()"
        return self.scalar_range()

    def scalar_range(self):
        """Return the range of the scalar values."""
        return np.array(self._data.GetScalarRange())

    def spacing(self, s=None):
        """Set/get the voxels size in the 3 dimensions."""
        if s is not None:
            self._data.SetSpacing(s)
            return self
        return np.array(self._data.GetSpacing())

    def origin(self, s=None):
        """Set/get the origin of the volumetric dataset."""
        ### superseedes base.origin()
        ### DIFFERENT from base.origin()!
        if s is not None:
            self._data.SetOrigin(s)
            return self
        return np.array(self._data.GetOrigin())

    def center(self, center=None):
        """Set/get the volume coordinates of its center.
        Position is reset to (0,0,0)."""
        if center is not None:
            cn = self._data.GetCenter()
            self._data.SetOrigin(-np.array(cn) / 2)
            self._update(self._data)
            self.pos(0, 0, 0)
            return self
        return np.array(self._data.GetCenter())

    def permute_axes(self, x, y, z):
        """Reorder the axes of the Volume by specifying
        the input axes which are supposed to become the new X, Y, and Z."""
        imp = vtk.vtkImagePermute()
        imp.SetFilteredAxes(x, y, z)
        imp.SetInputData(self.imagedata())
        imp.Update()
        return self._update(imp.GetOutput())

    def resample(self, new_spacing, interpolation=1):
        """
        Resamples a ``Volume`` to be larger or smaller.

        This method modifies the spacing of the input.
        Linear interpolation is used to resample the data.

        Parameters
        ----------
        new_spacing : list
            a list of 3 new spacings for the 3 axes

        interpolation : int
            0=nearest_neighbor, 1=linear, 2=cubic
        """
        rsp = vtk.vtkImageResample()
        oldsp = self.spacing()
        for i in range(3):
            if oldsp[i] != new_spacing[i]:
                rsp.SetAxisOutputSpacing(i, new_spacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.Update()
        return self._update(rsp.GetOutput())

    def interpolation(self, itype):
        """
        Set interpolation type.

        0 = nearest neighbour, 1 = linear
        """
        self.property.SetInterpolationType(itype)
        return self

    def threshold(self, above=None, below=None, replace=None, replace_value=None):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain a value above/below the input values
        and replace them with a new value (default is 0).
        """
        th = vtk.vtkImageThreshold()
        th.SetInputData(self.imagedata())

        # sanity checks
        if above is not None and below is not None:
            if above == below:
                return self
            if above > below:
                vedo.logger.warning("in volume.threshold(), above > below, skip.")
                return self

        ## cases
        if below is not None and above is not None:
            th.ThresholdBetween(above, below)

        elif above is not None:
            th.ThresholdByUpper(above)

        elif below is not None:
            th.ThresholdByLower(below)

        ##
        if replace is not None:
            th.SetReplaceIn(True)
            th.SetInValue(replace)
        else:
            th.SetReplaceIn(False)

        if replace_value is not None:
            th.SetReplaceOut(True)
            th.SetOutValue(replace_value)
        else:
            th.SetReplaceOut(False)

        th.Update()
        out = th.GetOutput()
        return self._update(out)

    def crop(
            self,
            left=None, right=None,
            back=None, front=None,
            bottom=None, top=None,
            VOI=()
        ):
        """
        Crop a ``Volume`` object.

        Parameters
        ----------
        left : float
            fraction to crop from the left plane (negative x)

        right : float
            fraction to crop from the right plane (positive x)

        back : float
            fraction to crop from the back plane (negative y)

        front : float
            fraction to crop from the front plane (positive y)

        bottom : float
            fraction to crop from the bottom plane (negative z)

        top : float
            fraction to crop from the top plane (positive z)

        VOI : list
            extract Volume Of Interest expressed in voxel numbers

        Example:
            `vol.crop(VOI=(xmin, xmax, ymin, ymax, zmin, zmax)) # all integers nrs`
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.imagedata())

        if VOI:
            extractVOI.SetVOI(VOI)
        else:
            d = self.imagedata().GetDimensions()
            bx0, bx1, by0, by1, bz0, bz1 = 0, d[0]-1, 0, d[1]-1, 0, d[2]-1
            if left is not None:   bx0 = int((d[0]-1)*left)
            if right is not None:  bx1 = int((d[0]-1)*(1-right))
            if back is not None:   by0 = int((d[1]-1)*back)
            if front is not None:  by1 = int((d[1]-1)*(1-front))
            if bottom is not None: bz0 = int((d[2]-1)*bottom)
            if top is not None:    bz1 = int((d[2]-1)*(1-top))
            extractVOI.SetVOI(bx0, bx1, by0, by1, bz0, bz1)
        extractVOI.Update()
        return self._update(extractVOI.GetOutput())

    def append(self, volumes, axis="z", preserve_extents=False):
        """
        Take the components from multiple inputs and merges them into one output.
        Except for the append axis, all inputs must have the same extent.
        All inputs must have the same number of scalar components.
        The output has the same origin and spacing as the first input.
        The origin and spacing of all other inputs are ignored.
        All inputs must have the same scalar type.

        Parameters
        ----------
        axis : int, str
            axis expanded to hold the multiple images

        preserve_extents : bool
            if True, the extent of the inputs is used to place
            the image in the output. The whole extent of the output is the union of the input
            whole extents. Any portion of the output not covered by the inputs is set to zero.
            The origin and spacing is taken from the first input.

        Example:
            .. code-block:: python

                from vedo import Volume, dataurl
                vol = Volume(dataurl+'embryo.tif')
                vol.append(vol, axis='x').show()
        """
        ima = vtk.vtkImageAppend()
        ima.SetInputData(self.imagedata())
        if not utils.is_sequence(volumes):
            volumes = [volumes]
        for volume in volumes:
            if isinstance(volume, vtk.vtkImageData):
                ima.AddInputData(volume)
            else:
                ima.AddInputData(volume.imagedata())
        ima.SetPreserveExtents(preserve_extents)
        if axis == "x":
            axis = 0
        elif axis == "y":
            axis = 1
        elif axis == "z":
            axis = 2
        ima.SetAppendAxis(axis)
        ima.Update()
        return self._update(ima.GetOutput())

    def resize(self, *newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        old_dims = np.array(self.imagedata().GetDimensions())
        old_spac = np.array(self.imagedata().GetSpacing())
        rsz = vtk.vtkImageResize()
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.imagedata())
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self._data = rsz.GetOutput()
        new_spac = old_spac * old_dims / newdims  # keep aspect ratio
        self._data.SetSpacing(new_spac)
        return self._update(self._data)

    def normalize(self):
        """Normalize that scalar components for each point."""
        norm = vtk.vtkImageNormalize()
        norm.SetInputData(self.imagedata())
        norm.Update()
        return self._update(norm.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror flip along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.
        """
        img = self.imagedata()

        ff = vtk.vtkImageFlip()
        ff.SetInputData(img)
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        elif axis.lower() == "z":
            ff.SetFilteredAxis(2)
        else:
            vedo.logger.error("mirror must be set to either x, y, z or n")
            raise RuntimeError()
        ff.Update()
        return self._update(ff.GetOutput())

    def operation(self, operation, volume2=None):
        """
        Perform operations with ``Volume`` objects. Keyword `volume2` can be a constant float.

        Possible operations are: ``+``, ``-``, ``/``, ``1/x``, ``sin``, ``cos``, ``exp``, ``log``,
        ``abs``, ``**2``, ``sqrt``, ``min``, ``max``, ``atan``, ``atan2``, ``median``,
        ``mag``, ``dot``, ``gradient``, ``divergence``, ``laplacian``.

        .. hint:: examples/volumetric/volumeOperations.py
        """
        op = operation.lower()
        image1 = self._data

        if op in ["median"]:
            mf = vtk.vtkImageMedian3D()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        if op in ["mag"]:
            mf = vtk.vtkImageMagnitude()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        if op in ["dot", "dotproduct"]:
            mf = vtk.vtkImageDotProduct()
            mf.SetInput1Data(image1)
            mf.SetInput2Data(volume2.imagedata())
            mf.Update()
            return Volume(mf.GetOutput())
        if op in ["grad", "gradient"]:
            mf = vtk.vtkImageGradient()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        if op in ["div", "divergence"]:
            mf = vtk.vtkImageDivergence()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        if op in ["laplacian"]:
            mf = vtk.vtkImageLaplacian()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())

        mat = vtk.vtkImageMathematics()
        mat.SetInput1Data(image1)

        K = None

        if utils.is_number(volume2):
            K = volume2
            mat.SetConstantK(K)
            mat.SetConstantC(K)

        elif volume2 is not None:  # assume image2 is a constant value
            mat.SetInput2Data(volume2.imagedata())

        # ###########################
        if op in ["+", "add", "plus"]:
            if K:
                mat.SetOperationToAddConstant()
            else:
                mat.SetOperationToAdd()

        elif op in ["-", "subtract", "minus"]:
            if K:
                mat.SetConstantC(-float(K))
                mat.SetOperationToAddConstant()
            else:
                mat.SetOperationToSubtract()

        elif op in ["*", "multiply", "times"]:
            if K:
                mat.SetOperationToMultiplyByK()
            else:
                mat.SetOperationToMultiply()

        elif op in ["/", "divide"]:
            if K:
                mat.SetConstantK(1.0 / K)
                mat.SetOperationToMultiplyByK()
            else:
                mat.SetOperationToDivide()

        elif op in ["1/x", "invert"]:
            mat.SetOperationToInvert()
        elif op in ["sin"]:
            mat.SetOperationToSin()
        elif op in ["cos"]:
            mat.SetOperationToCos()
        elif op in ["exp"]:
            mat.SetOperationToExp()
        elif op in ["log"]:
            mat.SetOperationToLog()
        elif op in ["abs"]:
            mat.SetOperationToAbsoluteValue()
        elif op in ["**2", "square"]:
            mat.SetOperationToSquare()
        elif op in ["sqrt", "sqr"]:
            mat.SetOperationToSquareRoot()
        elif op in ["min"]:
            mat.SetOperationToMin()
        elif op in ["max"]:
            mat.SetOperationToMax()
        elif op in ["atan"]:
            mat.SetOperationToATAN()
        elif op in ["atan2"]:
            mat.SetOperationToATAN2()
        else:
            vedo.logger.error(f"unknown operation {operation}")
            raise RuntimeError()
        mat.Update()
        return self._update(mat.GetOutput())

    def frequency_pass_filter(self, low_cutoff=None, high_cutoff=None, order=1):
        """
        Low-pass and high-pass filtering become trivial in the frequency domain.
        A portion of the pixels/voxels are simply masked or attenuated.
        This function applies a high pass Butterworth filter that attenuates the
        frequency domain image.

        The gradual attenuation of the filter is important.
        A simple high-pass filter would simply mask a set of pixels in the frequency domain,
        but the abrupt transition would cause a ringing effect in the spatial domain.

        Parameters
        ----------
        low_cutoff : list
            the cutoff frequencies for x, y and z

        high_cutoff : list
            the cutoff frequencies for x, y and z

        order : int
            order determines sharpness of the cutoff curve
        """
        # https://lorensen.github.io/VTKExamples/site/Cxx/ImageProcessing/IdealHighPass
        fft = vtk.vtkImageFFT()
        fft.SetInputData(self._data)
        fft.Update()
        out = fft.GetOutput()

        if high_cutoff:
            blp = vtk.vtkImageButterworthLowPass()
            blp.SetInputData(out)
            blp.SetCutOff(high_cutoff)
            blp.SetOrder(order)
            blp.Update()
            out = blp.GetOutput()

        if low_cutoff:
            bhp = vtk.vtkImageButterworthHighPass()
            bhp.SetInputData(out)
            bhp.SetCutOff(low_cutoff)
            bhp.SetOrder(order)
            bhp.Update()
            out = bhp.GetOutput()

        rfft = vtk.vtkImageRFFT()
        rfft.SetInputData(out)
        rfft.Update()

        ecomp = vtk.vtkImageExtractComponents()
        ecomp.SetInputData(rfft.GetOutput())
        ecomp.SetComponents(0)
        ecomp.Update()
        return self._update(ecomp.GetOutput())

    def smooth_gaussian(self, sigma=(2, 2, 2), radius=None):
        """
        Performs a convolution of the input Volume with a gaussian.

        Parameters
        ----------
        sigma : float, list
            standard deviation(s) in voxel units.
            A list can be given to smooth in the three direction differently.

        radius : float, list
            radius factor(s) determine how far out the gaussian
            kernel will go before being clamped to zero. A list can be given too.
        """
        gsf = vtk.vtkImageGaussianSmooth()
        gsf.SetDimensionality(3)
        gsf.SetInputData(self.imagedata())
        if utils.is_sequence(sigma):
            gsf.SetStandardDeviations(sigma)
        else:
            gsf.SetStandardDeviation(sigma)
        if radius is not None:
            if utils.is_sequence(radius):
                gsf.SetRadiusFactors(radius)
            else:
                gsf.SetRadiusFactor(radius)
        gsf.Update()
        return self._update(gsf.GetOutput())

    def smooth_median(self, neighbours=(2, 2, 2)):
        """
        Median filter that replaces each pixel with the median value
        from a rectangular neighborhood around that pixel.
        """
        imgm = vtk.vtkImageMedian3D()
        imgm.SetInputData(self.imagedata())
        if utils.is_sequence(neighbours):
            imgm.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        else:
            imgm.SetKernelSize(neighbours, neighbours, neighbours)
        imgm.Update()
        return self._update(imgm.GetOutput())

    def erode(self, neighbours=(2, 2, 2)):
        """
        Replace a voxel with the minimum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        .. hint:: examples/volumetric/erode_dilate.py
        """
        ver = vtk.vtkImageContinuousErode3D()
        ver.SetInputData(self._data)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())

    def dilate(self, neighbours=(2, 2, 2)):
        """
        Replace a voxel with the maximum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        .. hint:: examples/volumetric/erode_dilate.py
        """
        ver = vtk.vtkImageContinuousDilate3D()
        ver.SetInputData(self._data)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())

    def magnitude(self):
        """Colapses components with magnitude function."""
        imgm = vtk.vtkImageMagnitude()
        imgm.SetInputData(self.imagedata())
        imgm.Update()
        return self._update(imgm.GetOutput())

    def topoints(self):
        """
        Extract all image voxels as points.
        This function takes an input ``Volume`` and creates an ``Mesh``
        that contains the points and the point attributes.

        .. hint:: examples/volumetric/vol2points.py
        """
        v2p = vtk.vtkImageToPoints()
        v2p.SetInputData(self.imagedata())
        v2p.Update()
        mpts = vedo.Points(v2p.GetOutput())
        return mpts

    def euclidean_distance(self, anisotropy=False, max_distance=None):
        """
        Implementation of the Euclidean DT (Distance Transform) using Saito's algorithm.
        The distance map produced contains the square of the Euclidean distance values.
        The algorithm has a O(n^(D+1)) complexity over n x n x...x n images in D dimensions.

        Check out also: https://en.wikipedia.org/wiki/Distance_transform

        Parameters
        ----------
        anisotropy : bool
            used to define whether Spacing should be used in the computation of the distances.

        max_distance : bool
            any distance bigger than max_distance will not be
            computed but set to this specified value instead.

        .. hint:: examples/volumetric/euclDist.py
        """
        euv = vtk.vtkImageEuclideanDistance()
        euv.SetInputData(self._data)
        euv.SetConsiderAnisotropy(anisotropy)
        if max_distance is not None:
            euv.InitializeOn()
            euv.SetMaximumDistance(max_distance)
        euv.SetAlgorithmToSaito()
        euv.Update()
        return Volume(euv.GetOutput())

    def correlation_with(self, vol2, dim=2):
        """
        Find the correlation between two volumetric data sets.
        Keyword `dim` determines whether the correlation will be 3D, 2D or 1D.
        The default is a 2D Correlation.

        The output size will match the size of the first input.
        The second input is considered the correlation kernel.
        """
        imc = vtk.vtkImageCorrelation()
        imc.SetInput1Data(self._data)
        imc.SetInput2Data(vol2.imagedata())
        imc.SetDimensionality(dim)
        imc.Update()
        return Volume(imc.GetOutput())

    def scale_voxels(self, scale=1):
        """Scale the voxel content by factor `scale`."""
        rsl = vtk.vtkImageReslice()
        rsl.SetInputData(self.imagedata())
        rsl.SetScalarScale(scale)
        rsl.Update()
        return self._update(rsl.GetOutput())


##########################################################################
class Volume(vtk.vtkVolume, BaseGrid, BaseVolume):
    """
    Derived class of ``vtkVolume``.
    Can be initialized with a numpy object, a ``vtkImageData``
    or a list of 2D bmp files.

    Parameters
    ----------
    c : list, str
        sets colors along the scalar range, or a matplotlib color map name

    alphas : float, list
         sets transparencies along the scalar range

    alpha_unit : float
        low values make composite rendering look brighter and denser

    origin : list
        set volume origin coordinates

    spacing : list
        voxel dimensions in x, y and z.

    dims : list
        specify the dimensions of the volume.

    mapper : str
        either 'gpu', 'opengl_gpu', 'fixed' or 'smart'

    mode : int
        define the volumetric rendering style:

            - 0, composite rendering
            - 1, maximum projection
            - 2, minimum projection
            - 3, average projection
            - 4, additive mode

        The default mode is "composite" where the scalar values are sampled through
        the volume and composited in a front-to-back scheme through alpha blending.
        The final color and opacity is determined using the color and opacity transfer
        functions specified in alpha keyword.

        Maximum and minimum intensity blend modes use the maximum and minimum
        scalar values, respectively, along the sampling ray.
        The final color and opacity is determined by passing the resultant value
        through the color and opacity transfer functions.

        Additive blend mode accumulates scalar values by passing each value
        through the opacity transfer function and then adding up the product
        of the value and its opacity. In other words, the scalar values are scaled
        using the opacity transfer function and summed to derive the final color.
        Note that the resulting image is always grayscale i.e. aggregated values
        are not passed through the color transfer function.
        This is because the final value is a derived value and not a real data value
        along the sampling ray.

        Average intensity blend mode works similar to the additive blend mode where
        the scalar values are multiplied by opacity calculated from the opacity
        transfer function and then added.
        The additional step here is to divide the sum by the number of samples
        taken through the volume.
        As is the case with the additive intensity projection, the final image will
        always be grayscale i.e. the aggregated values are not passed through the
        color transfer function.

    Example:
        .. code-block:: python

            from vedo import Volume
            vol = Volume("path/to/mydata/rec*.bmp", c='jet', mode=1)
            vol.show(axes=1)

    .. note:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range of the scalar.

    .. hint:: examples/volumetric/numpy2volume1.py
        .. image:: https://vedo.embl.es/images/volumetric/numpy2volume1.png

    .. hint:: examples/volumetric/read_volume2.py
        .. image:: https://vedo.embl.es/images/volumetric/read_volume2.png
    """

    def __init__(
        self,
        inputobj=None,
        c="RdBu_r",
        alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1.0),
        alpha_gradient=None,
        alpha_unit=1,
        mode=0,
        spacing=None,
        dims=None,
        origin=None,
        mapper="smart",
    ):

        vtk.vtkVolume.__init__(self)
        BaseGrid.__init__(self)
        BaseVolume.__init__(self)

        ###################
        if isinstance(inputobj, str):

            if "https://" in inputobj:
                inputobj = vedo.io.download(inputobj, verbose=False)  # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        if "gpu" in mapper:
            self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif "opengl_gpu" in mapper:
            self._mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif "smart" in mapper:
            self._mapper = vtk.vtkSmartVolumeMapper()
        elif "fixed" in mapper:
            self._mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self._mapper = mapper
        else:
            print("Error unknown mapper type", [mapper])
            raise RuntimeError()
        self.SetMapper(self._mapper)

        ###################
        inputtype = str(type(inputobj))

        # print('Volume inputtype', inputtype, c='b')

        if inputobj is None:
            img = vtk.vtkImageData()

        elif utils.is_sequence(inputobj):

            if isinstance(inputobj[0], str) and ".bmp" in inputobj[0].lower():
                # scan sequence of BMP files
                ima = vtk.vtkImageAppend()
                ima.SetAppendAxis(2)
                pb = utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    if "_rec_spr.bmp" in f:
                        continue
                    picr = vtk.vtkBMPReader()
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtk.vtkImageMagnitude()
                    mgf.SetInputData(picr.GetOutput())
                    mgf.Update()
                    ima.AddInputData(mgf.GetOutput())
                    pb.print("loading...")
                ima.Update()
                img = ima.GetOutput()

            else:

                if len(inputobj.shape) == 1:
                    varr = utils.numpy2vtk(inputobj)
                else:
                    varr = utils.numpy2vtk(inputobj.ravel(order="F"))
                varr.SetName("input_scalars")

                img = vtk.vtkImageData()
                if dims is not None:
                    img.SetDimensions(dims)
                else:
                    if len(inputobj.shape)==1:
                        vedo.logger.error("must set dimensions (dims keyword) in Volume")
                        raise RuntimeError()
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

        elif "ImageData" in inputtype:
            img = inputobj

        elif isinstance(inputobj, Volume):
            img = inputobj.inputdata()

        elif "UniformGrid" in inputtype:
            img = inputobj

        elif hasattr(inputobj, "GetOutput"):  # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.io.download(inputobj, verbose=False)
            img = vedo.io.loadImageData(inputobj)

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        if dims is not None:
            img.SetDimensions(dims)

        if origin is not None:
            img.SetOrigin(origin)  ### DIFFERENT from volume.origin()!

        if spacing is not None:
            img.SetSpacing(spacing)

        self._data = img
        self._mapper.SetInputData(img)

        if img.GetPointData().GetScalars():
            if img.GetPointData().GetScalars().GetNumberOfComponents() == 1:
                self.mode(mode).color(c).alpha(alpha).alpha_gradient(alpha_gradient)
                self.GetProperty().SetShade(True)
                self.GetProperty().SetInterpolationType(1)
                self.GetProperty().SetScalarOpacityUnitDistance(alpha_unit)

        # remember stuff:
        self._mode = mode
        self._color = c
        self._alpha = alpha
        self._alpha_grad = alpha_gradient
        self._alpha_unit = alpha_unit

    def _update(self, data):
        self._data = data
        self._data.GetPointData().Modified()
        self._mapper.SetInputData(data)
        self._mapper.Modified()
        self._mapper.Update()
        return self

    def mode(self, mode=None):
        """
        Define the volumetric rendering style.

            - 0, composite rendering
            - 1, maximum projection rendering
            - 2, minimum projection rendering
            - 3, average projection rendering
            - 4, additive mode

        The default mode is "composite" where the scalar values are sampled through
        the volume and composited in a front-to-back scheme through alpha blending.
        The final color and opacity is determined using the color and opacity transfer
        functions specified in alpha keyword.

        Maximum and minimum intensity blend modes use the maximum and minimum
        scalar values, respectively, along the sampling ray.
        The final color and opacity is determined by passing the resultant value
        through the color and opacity transfer functions.

        Additive blend mode accumulates scalar values by passing each value
        through the opacity transfer function and then adding up the product
        of the value and its opacity. In other words, the scalar values are scaled
        using the opacity transfer function and summed to derive the final color.
        Note that the resulting image is always grayscale i.e. aggregated values
        are not passed through the color transfer function.
        This is because the final value is a derived value and not a real data value
        along the sampling ray.

        Average intensity blend mode works similar to the additive blend mode where
        the scalar values are multiplied by opacity calculated from the opacity
        transfer function and then added.
        The additional step here is to divide the sum by the number of samples
        taken through the volume.
        As is the case with the additive intensity projection, the final image will
        always be grayscale i.e. the aggregated values are not passed through the
        color transfer function.
        """
        if mode is None:
            return self._mapper.GetBlendMode()

        if isinstance(mode, str):
            if "comp" in mode:
                mode = 0
            elif "proj" in mode:
                if "max" in mode:
                    mode = 1
                elif "min" in mode:
                    mode = 2
                elif "ave" in mode:
                    mode = 3
                else:
                    vedo.logger.warning(f"unknown mode {mode}")
                    mode = 0
            elif "add" in mode:
                mode = 4
            else:
                vedo.logger.warning(f"unknown mode {mode}")
                mode = 0

        self._mapper.SetBlendMode(mode)
        self._mode = mode
        return self

    def shade(self, status=None):
        """
        Set/Get the shading of a Volume.
        Shading can be further controlled with ``volume.lighting()`` method.

        If shading is turned on, the mapper may perform shading calculations.
        In some cases shading does not apply
        (for example, in maximum intensity projection mode).
        """
        if status is None:
            return self.GetProperty().GetShade()
        self.GetProperty().SetShade(status)
        return self

    def cmap(self, c, alpha=None, vmin=None, vmax=None):
        """Same as color().

        Parameters
        ----------
        alpha : list
            use a list to specify transparencies along the scalar range

        vmin : float
            force the min of the scalar range to be this value

        vmax : float
            force the max of the scalar range to be this value
        """
        return self.color(c, alpha, vmin, vmax)

    def jittering(self, status=None):
        """
        If `jittering` is `True`, each ray traversal direction will be perturbed slightly
        using a noise-texture to get rid of wood-grain effects.
        """
        if hasattr(self._mapper, "SetUseJittering"):  # tetmesh doesnt have it
            if status is None:
                return self._mapper.GetUseJittering()
            self._mapper.SetUseJittering(status)
        return self

    def mask(self, data):
        """
        Mask a volume visualization with a binary value.
        Needs to specify keyword mapper='gpu'.

        Example:
            .. code-block:: python

                from vedo import np, Volume, show

                data_matrix = np.zeros([75, 75, 75], dtype=np.uint8)
                # all voxels have value zero except:
                data_matrix[0:35,   0:35,  0:35] = 1
                data_matrix[35:55, 35:55, 35:55] = 2
                data_matrix[55:74, 55:74, 55:74] = 3
                vol = Volume(data_matrix, c=['white','b','g','r'], mapper='gpu')

                data_mask = np.zeros_like(data_matrix)
                data_mask[10:65, 10:45, 20:75] = 1
                vol.mask(data_mask)

                show(vol, axes=1).close()
        """
        mask = Volume(data.astype(np.uint8))
        try:
            self.mapper().SetMaskTypeToBinary()
            self.mapper().SetMaskInput(mask.inputdata())
        except AttributeError:
            vedo.logger.error("volume.mask() must create the volume with Volume(..., mapper='gpu')")
        return self

    def alpha_gradient(self, alpha_grad, vmin=None, vmax=None):
        """
        Assign a set of tranparencies to a volume's gradient
        along the range of the scalar value.
        A single constant value can also be assigned.
        The gradient function is used to decrease the opacity
        in the "flat" regions of the volume while maintaining the opacity
        at the boundaries between material types.  The gradient is measured
        as the amount by which the intensity changes over unit distance.

        The format for alpha_grad is the same as for method `volume.alpha()`.
        """
        if vmin is None:
            vmin, _ = self._data.GetScalarRange()
        if vmax is None:
            _, vmax = self._data.GetScalarRange()
        self._alpha_grad = alpha_grad
        volumeProperty = self.GetProperty()

        if alpha_grad is None:
            volumeProperty.DisableGradientOpacityOn()
            return self

        volumeProperty.DisableGradientOpacityOff()

        gotf = volumeProperty.GetGradientOpacity()
        if utils.is_sequence(alpha_grad):
            alpha_grad = np.array(alpha_grad)
            if len(alpha_grad.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha_grad):
                    xalpha = vmin + (vmax - vmin) * i / (len(alpha_grad) - 1)
                    # Create transfer mapping scalar value to gradient opacity
                    gotf.AddPoint(xalpha, al)
            elif len(alpha_grad.shape) == 2:  # user passing [(x0,alpha0), ...]
                gotf.AddPoint(vmin, alpha_grad[0][1])
                for xalpha, al in alpha_grad:
                    # Create transfer mapping scalar value to opacity
                    gotf.AddPoint(xalpha, al)
                gotf.AddPoint(vmax, alpha_grad[-1][1])
            # print("alpha_grad at", round(xalpha, 1), "\tset to", al)
        else:
            gotf.AddPoint(vmin, alpha_grad)  # constant alpha_grad
            gotf.AddPoint(vmax, alpha_grad)
        return self

    def component_weight(self, i, weight):
        """Set the scalar component weight in range [0,1]."""
        self.GetProperty().SetComponentWeight(i, weight)
        return self

    def xslice(self, i):
        """Extract the slice at index `i` of volume along x-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if i > nx - 1:
            i = nx - 1
        vslice.SetExtent(i, i, 0, ny, 0, nz)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    def yslice(self, j):
        """Extract the slice at index `j` of volume along y-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if j > ny - 1:
            j = ny - 1
        vslice.SetExtent(0, nx, j, j, 0, nz)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    def zslice(self, k):
        """Extract the slice at index `i` of volume along z-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if k > nz - 1:
            k = nz - 1
        vslice.SetExtent(0, nx, 0, ny, k, k)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    @deprecated(reason=vedo.colors.red + "Please use slice_plane()" + vedo.colors.reset)
    def slicePlane(self, *a, **b):
        "Deprecated. Please use scalar_range()"
        return self.slice_plane(*a, **b)

    def slice_plane(self, origin=(0, 0, 0), normal=(1, 1, 1), autocrop=False):
        """
        Extract the slice along a given plane position and normal.

        .. hint:: examples/volumetric/slicePlane1.py
            .. image:: https://vedo.embl.es/images/volumetric/slicePlane1.gif
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self._data)
        reslice.SetOutputDimensionality(2)
        newaxis = utils.versor(normal)
        pos = np.array(origin)
        initaxis = (0, 0, 1)
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(pos)
        M = T.GetMatrix()
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToLinear()
        reslice.SetAutoCropOutput(not autocrop)
        reslice.Update()
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(reslice.GetOutput())
        vslice.Update()
        msh = Mesh(vslice.GetOutput())
        msh.SetOrientation(T.GetOrientation())
        msh.SetPosition(pos)
        return msh

    def warp(self, source, target, sigma=1, mode="3d", fit=False):
        """
        Warp volume scalars within a Volume by specifying source and target sets of points.

        Parameters
        ----------
        source : Points, list
            the list of source points

        target : Points, list
            the list of target points

        fit : bool
            fit/adapt the old bounding box to the warped geometry
        """
        if isinstance(source, vedo.Points):
            source = source.points()
        if isinstance(target, vedo.Points):
            target = target.points()

        ns = len(source)
        ptsou = vtk.vtkPoints()
        ptsou.SetNumberOfPoints(ns)
        for i in range(ns):
            ptsou.SetPoint(i, source[i])

        nt = len(target)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        pttar = vtk.vtkPoints()
        pttar.SetNumberOfPoints(nt)
        for i in range(ns):
            pttar.SetPoint(i, target[i])

        T = vtk.vtkThinPlateSplineTransform()
        if mode.lower() == "3d":
            T.SetBasisToR()
        elif mode.lower() == "2d":
            T.SetBasisToR2LogR()
        else:
            vedo.logger.error(f"unknown mode {mode}")
            raise RuntimeError()

        T.SetSigma(sigma)
        T.SetSourceLandmarks(ptsou)
        T.SetTargetLandmarks(pttar)
        T.Inverse()
        self.transform = T
        self.apply_transform(T, fit=fit)
        return self

    def apply_transform(self, T, fit=False):
        """
        Apply a VTK transform to the scalars in the volume.

        Parameters
        ----------
        T : vtkTransform, matrix
            The transformation to be applied

        fit : bool
            fit/adapt the old bounding box to the warped geometry
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
            T = tr

        elif utils.is_sequence(T):
            M = vtk.vtkMatrix4x4()
            n = len(T[0])
            for i in range(n):
                for j in range(n):
                    M.SetElement(i, j, T[i][j])
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            T = tr

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self._data)
        reslice.SetResliceTransform(T)
        reslice.SetOutputDimensionality(3)
        reslice.SetInterpolationModeToLinear()

        spacing = self._data.GetSpacing()
        origin = self._data.GetOrigin()

        if fit:
            bb = self.box()
            if isinstance(T, vtk.vtkThinPlateSplineTransform):
                TI = vtk.vtkThinPlateSplineTransform()
                TI.DeepCopy(T)
                TI.Inverse()
            else:
                TI = vtk.vtkTransform()
                TI.DeepCopy(T)
            bb.apply_transform(TI)
            bounds = bb.GetBounds()
            bounds = (
                bounds[0]/spacing[0], bounds[1]/spacing[0],
                bounds[2]/spacing[1], bounds[3]/spacing[1],
                bounds[4]/spacing[2], bounds[5]/spacing[2],
            )
            bounds = np.round(bounds).astype(int)
            reslice.SetOutputExtent(bounds)
            reslice.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
            reslice.SetOutputOrigin(origin[0], origin[1], origin[2])

        reslice.Update()
        return self._update(reslice.GetOutput())


##########################################################################
class VolumeSlice(vtk.vtkImageSlice, Base3DProp, BaseVolume):
    """
    Derived class of `vtkImageSlice`.
    This class is equivalent to ``Volume`` except for its representation.
    The main purpose of this class is to be used in conjunction with `Volume`
    for visualization using `mode="image"`.
    """

    def __init__(self, inputobj=None):

        vtk.vtkImageSlice.__init__(self)
        Base3DProp.__init__(self)
        BaseVolume.__init__(self)

        self._mapper = vtk.vtkImageResliceMapper()
        self._mapper.SliceFacesCameraOn()
        self._mapper.SliceAtFocalPointOn()
        self._mapper.SetAutoAdjustImageQuality(False)
        self._mapper.BorderOff()

        self.lut = None

        self.property = vtk.vtkImageProperty()
        self.property.SetInterpolationTypeToLinear()
        self.SetProperty(self.property)

        ###################
        if isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.io.download(inputobj, verbose=False)  # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        inputtype = str(type(inputobj))

        if inputobj is None:
            img = vtk.vtkImageData()

        if isinstance(inputobj, Volume):
            img = inputobj.imagedata()
            self.lut = utils.ctf2lut(inputobj)

        elif utils.is_sequence(inputobj):

            if isinstance(inputobj[0], str):  # scan sequence of BMP files
                ima = vtk.vtkImageAppend()
                ima.SetAppendAxis(2)
                pb = utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    picr = vtk.vtkBMPReader()
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtk.vtkImageMagnitude()
                    mgf.SetInputData(picr.GetOutput())
                    mgf.Update()
                    ima.AddInputData(mgf.GetOutput())
                    pb.print("loading...")
                ima.Update()
                img = ima.GetOutput()

            else:
                if "ndarray" not in inputtype:
                    inputobj = np.array(inputobj)

                if len(inputobj.shape) == 1:
                    varr = utils.numpy2vtk(inputobj, dtype=float)
                else:
                    if len(inputobj.shape) > 2:
                        inputobj = np.transpose(inputobj, axes=[2, 1, 0])
                    varr = utils.numpy2vtk(inputobj.ravel(order="F"), dtype=float)
                varr.SetName("input_scalars")

                img = vtk.vtkImageData()
                img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

        elif "ImageData" in inputtype:
            img = inputobj

        elif isinstance(inputobj, Volume):
            img = inputobj.inputdata()

        elif "UniformGrid" in inputtype:
            img = inputobj

        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.io.download(inputobj, verbose=False)
            img = vedo.io.loadImageData(inputobj)

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        self._data = img
        self._mapper.SetInputData(img)
        self.SetMapper(self._mapper)

    def bounds(self):
        """Return the bounding box as [x0,x1, y0,y1, z0,z1]"""
        bns = [0, 0, 0, 0, 0, 0]
        self.GetBounds(bns)
        return bns

    def colorize(self, lut=None, fix_scalar_range=False):
        """
        Assign a LUT (Look Up Table) to colorize the slice, leave it ``None``
        to reuse an existing Volume color map.
        Use "bw" for automatic black and white.
        """
        if lut is None and self.lut:
            self.property.SetLookupTable(self.lut)
        elif isinstance(lut, vtk.vtkLookupTable):
            self.property.SetLookupTable(lut)
        elif lut == "bw":
            self.property.SetLookupTable(None)
        self.property.SetUseLookupTableScalarRange(fix_scalar_range)
        return self

    def alpha(self, value):
        """Set opacity to the slice"""
        self.property.SetOpacity(value)
        return self

    def auto_adjust_quality(self, value=True):
        """Automatically reduce the rendering quality for greater speed when interacting"""
        self._mapper.SetAutoAdjustImageQuality(value)
        return self

    def slab(self, thickness=0, mode=0, sample_factor=2):
        """
        Make a thick slice (slab).

        Parameters
        ----------
        thickness : float
            set the slab thickness, for thick slicing

        mode : int
            The slab type:
            0 = min
            1 = max
            2 = mean
            3 = sum

        sample_factor : float
            Set the number of slab samples to use as a factor of the number of input slices
            within the slab thickness. The default value is 2, but 1 will increase speed
            with very little loss of quality.
        """
        self._mapper.SetSlabThickness(thickness)
        self._mapper.SetSlabType(mode)
        self._mapper.SetSlabSampleFactor(sample_factor)
        return self

    def face_camera(self, value=True):
        """Make the slice always face the camera or not."""
        self._mapper.SetSliceFacesCameraOn(value)
        return self

    def jump_to_nearest_slice(self, value=True):
        """This causes the slicing to occur at the closest slice to the focal point,
        instead of the default behavior where a new slice is interpolated between the original slices.
        Nothing happens if the plane is oblique to the original slices."""
        self.SetJumpToNearestSlice(value)
        return self

    def fill_background(self, value=True):
        """Instead of rendering only to the image border, render out to the viewport boundary with
        the background color. The background color will be the lowest color on the lookup
        table that is being used for the image."""
        self._mapper.SetBackground(value)
        return self

    def lighting(self, window, level, ambient=1.0, diffuse=0.0):
        """Assign the values for window and color level."""
        self.property.SetColorWindow(window)
        self.property.SetColorLevel(level)
        self.property.SetAmbient(ambient)
        self.property.SetDiffuse(diffuse)
        return self
