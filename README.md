# annotationTool
Contains the script for the annotation tool and a modified vedo library to add functionalities.
The annotation tool loads a dataset, it takes images, labels (in kitti format) pointclouds, and the calibration matrices of the camera.

The dataset should be placed in the folder dataForViewing/dataset1, inside the dataset1 folder, you have to put 4 folders named as "calib" for the calibration matrices, "Image_2" for the images, "labels_2" for the labels, and "velodyne" for the pointclouds, all data samples have to have the same name convention across the whole dataset, meaning that the files for let's say the first data sample, should be like this: the corresponding calibration matrix file should be named "000001.txt", the corresponding image should be "000001.png", the corresponding label file should be "000001.txt" and the corresponding pointcloud should be "000001.bin", all in their respective folders.

## How to use it

When the script starts, it loads the first data sample, this will show you a BEV's, a lateral right view and a lateral left view from the pointcloud scene with the 3D bounding boxes overlayed in the scene, also the corresponding image from the camera with projections of the 3D bounding boxes, and finally a GUI with the labels of that data sample.

The GUI allows you to go through the data samples with the "previous" and "next" buttons, it shows a numbered list of the detections so that one can identify which label belongs to which detection, the original label files contain more information but GUI shows only what is meaningful for the user in terms of the position of the detection in the 3D scene, for example, the height, width and length of the 3D bounding box, the x, y and z position of the centroid of this boxes in camera coordinates and the rotation of the object around the y-axis, in camera coordinates, in radians.

![annotation-Tool](https://github.com/user-attachments/assets/156df593-7a4f-4c4c-8189-850052b15cdd)

To add a label, one has to click in the "Add label" button, this will create an extra row in the list that the user can fill, then to save it, one has to click the "Save changes" button, this will reload and show the current datasample and the new label will appear in the scene 

<div align="center">
  <img src="https://github.com/user-attachments/assets/043a5936-fe9d-470b-8074-fc831e05f8e2" alt="add-label" />
</div>

![add-label-scene](https://github.com/user-attachments/assets/f8cf6115-a9cd-4aa2-bf82-fda6e3eeef21)

To delete a label, one has to select any of the cells of the label to be deleted, then click "Delete Label", this will highlight the entire row so that the user can verify it, and to delete it, he has to sabe changes, if the user only clicks "Delete Label" without saving changes, the label won't be deleted, by saving the changes the data sample is reloaded and changes will be visible.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1a03ddd7-24d7-4030-baae-f1c172a17b45" alt="delete-highlight" />
</div>

![deleted-label](https://github.com/user-attachments/assets/01f0ddc2-fa6f-414b-82d9-c51f44203524)


## About the environment

It was tested in ubuntu 20.04, using WSL and since this is a graphical tool, consider installing and setting VcXsrv so that linux applications can use the display of the host machine.

If that is the case, then after installing VcXsrv in the host machine, in your linux system:
```
$ sudo apt update
$ sudo apt install mesa-utils
$ sudo apt install x11-apps
```

Then you will need to modify the ~/.bashrc file to add:
```
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
```
Save the file and the changes in the ~/.bashrc and then launch:

```
$ source ~/.bashrc
```

# Create conda environment 
```
$ conda create -n annotation-tool python=3.8
```

## Dependencies 

```
$ conda install numpy=1.21.19
$ conda install pyat vtk deprecated opencv matplotlib
```

If packages are not in the default channel then try:

```
$ conda install -c conda-forge numpy=1.21.19
$ conda install -c conda-forge pyat vtk deprecated opencv matplotlib
```

Finally, run the labelEditorTool.py script 
