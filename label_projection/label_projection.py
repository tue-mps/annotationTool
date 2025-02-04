import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from os import walk
import re

# Calibration params
CAMERA_MATRIX = np.array([[1.84541929e+03, 0.0, 8.55802458e+02], [0.0, 1.78869210e+03, 6.07342667e+02], [0.0, 0.0, 1.0]]) 
DISTORTION_COEFFICIENTS = np.array([ 2.51771602e-01, -1.32561698e+01,  4.33607564e-03, -6.94637533e-03, 5.95513933e+01])    
CAMERA_TO_LIDAR_ROTATION = np.array([1.61803058,  0.03365624, -0.04003127])
CAMERA_TO_LIDAR_TRANSLATION = np.array([0.09138029, 1.38369885, 1.43674736])
RADAR_TO_LIDAR_ROTATION = np.array([0.0, 0.0, 0.0]) # Dummy for now
RADAR_TO_LIDAR_TRANSLATION = np.array([0.0, 0.0, 0.0]) # Dummy for now


# Data location   
IMAGES_DIR = "/home/danil/RADIalHD/Radial_imagesHD"
LABELS_DIR = "/home/danil/RADIalHD/Radial_imagesHD_labels"
LASER_PCL_DIR = "/home/danil/data/RADIal/laser_PCL"
OUTPUT_IMAGES_DIR = "/home/danil/data/RADIal/projected_labels"
OUTPUT_LABELS_DIR = "/home/danil/data/RADIal/labelled_range_azimuth"


COLORS_ARRAY = np.array(['pink', 'red', 'green', 'blue', 'purple', 'orange'])


def pcl_to_range_azimuth(point_cloud):
    # Convert to polar coordinates (range and azimuth)
    ranges = np.sqrt(point_cloud[:, 0]**2 + point_cloud[:, 1]**2)  # Range
    azimuths = np.arctan2(point_cloud[:, 1], point_cloud[:, 0]) # Azimuth in degrees
    markers = point_cloud[:,3]

    # Normalize azimuths to [0, 360) for consistency
    #azimuths = (azimuths + 360) % 360

    # Convert to degrees, front is 0, left-hand are positive values, and right-hand are negative
    azimuths = np.rad2deg(azimuths) - 90

    # Create a 2D array with each point's range and azimuth
    range_azimuth_array = np.column_stack((ranges, azimuths, markers))

    return range_azimuth_array  

def cluster_pc(pc, num_labels, eps = 0.2, ingore_z = False):  
    for label_index in range(0, num_labels): 
        filter_by_label = pc[:,3] == label_index
        labelled_points = pc[filter_by_label]
        labelled_indicies = np.where(filter_by_label)[0]

        if len(labelled_points) == 0:
            print("No points for the label ", label_index)
            continue

        print("Clustering points for label", label_index)
        clustering = None
        if ingore_z:
            points_2d = labelled_points[:,[0,1]]
            print("Points 2D:", points_2d.shape)
            clustering = DBSCAN(eps=eps, min_samples=1).fit(points_2d)
        else:    
            clustering = DBSCAN(eps=eps, min_samples=1).fit(labelled_points)
        cluster_labels = clustering.labels_
        
        print("Labels:", cluster_labels.shape)
        label_max = cluster_labels.max()
        if label_max == -1:
            print("Could not cluster, returning original point cloud")
            continue

        print("Label max: ", label_max)

        label_with_max_points = 0
        max_num_points = 0

        for i in range(0, label_max + 1):
            num_points = len(cluster_labels[cluster_labels[:] == i])
            if (num_points > max_num_points):
                max_num_points = num_points
                label_with_max_points = i

        print("The biggest cluster has ", max_num_points, " points")  

        for j, cluster_label in enumerate(cluster_labels):
            if cluster_label != label_with_max_points:
                pc[labelled_indicies[j],3] = -1 # Remove label 

    return pc                                 

def get_sample_pc(id):       
    filename = os.path.join(LASER_PCL_DIR, "pcl_{:s}.npy".format(id))
    return np.load(filename,allow_pickle=True)  

def read_lables(width, height, id):
    filename = os.path.join(LABELS_DIR, "image_{:s}.txt".format(id))       
    labels_data = np.loadtxt(filename)
    if (len(labels_data) == 0):
        return np.array([])
    if len(labels_data.shape) == 1:
        labels_data = np.array([labels_data])
    print("Labels: ", labels_data)
    labels = np.empty([len(labels_data), 4], dtype=int) 
    for index, label in enumerate(labels_data):
        center_x = width * label[1]
        center_y = height * label[2]
        label_width = width * label[3]
        label_height = height * label[4]
        box = [int(center_x - label_width/2), int(center_y - label_height/2), int(center_x + label_width/2), int(center_y + label_height/2)]
        labels[index] = box
    return labels    

def label_point_cloud(pc, points_2d, labels):
    """
    Labels point cloud (pc) by checking the corresponding prohected points (points_2d) to be inside the bounding box specified by each label
    """    
    count_marked = 0

    for index, e in enumerate(points_2d): 
        for label_index, label in enumerate(labels):
            # For TU/e data points in front of the camera are with X < 0 (lidar X axis points backwards)
            # For the RADIal data the X points forward and there are no points behind
            if (e[0] >= label[0] and e[0] <= label[2] and e[1] >= label[1] and e[1] <= label[3]):
                pc[index, 3] = label_index  
                count_marked += 1

    print(count_marked, "points are inside the labelled windows") 
    return pc    

def show_range_azimuth(pc, num_labels, id):
    ra = pcl_to_range_azimuth(pc)
    print("Range-Azimuth", ra.shape)

    # As a maximum range, take the furthest labelled point
    labeled_points = ra[ra[:,2] != -1]
    max_range = max(labeled_points[:,0] if len(labeled_points) > 0 else ra[:,0])

    print("Maximum range for a labelled point =", max_range)

    filter = ra[:,0] < max_range
    ra = ra[filter]

    print("Filtered Range-Azimuth", ra.shape)

    az = ra[:,1]
    r = ra[:,0]

    # Visualize the point cloud in a polar plot
  
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': 'polar'})

    # Shift marker values [-1, len(labels)) 1 position to the right to have an array of values [0, len(labels) + 1)
    colors = ra[:,2].astype(int) + 1
    colormap = []
    colormap.append('black')
    for i in range(0, num_labels):
        colormap.append(COLORS_ARRAY[i % len(COLORS_ARRAY)])
    colormap = np.array(colormap)  

    # Create array of sizes, labelled points are bigger
    sizes = np.where(ra[:,2] == -1, 1, 5)

    ax.set_thetamax(180)  
    ax.set_thetamin(0)
    
    ax.scatter(az, r, c=colormap[colors], s=sizes)

    # Plot the selected region as a red mark
    plt.title("Sample {:s}".format(id))
    plt.show()   

def save_range_azimuth(pc, num_labels, id):
    f = open(os.path.join(OUTPUT_LABELS_DIR, "{:s}.txt".format(id)), "w")
    ra = pcl_to_range_azimuth(pc)
    print("Range-Azimuth", ra.shape)

    for i in range(0, num_labels):
        points_i = ra[ra[:,2] == i]
        if points_i.size == 0:
            continue
        r_i = points_i[:,0] 
        az_i = points_i[:,1]
        f.write("{:d} {:f} {:f}\n".format(i, np.average(r_i), np.average(az_i)))

    f.close()  

def save_image(image, labels, pc, width, height, id):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      

    # Add markers to 2D points
    image_points = np.hstack((points_2d, pc[:,3].reshape(-1, 1)))
    print("Labelled points:", len(image_points[image_points[:,2] >= 0]))
    # Consider only points in the image
    filter = (image_points[:, 0] >= 0) & (image_points[:, 0] <= width) & (image_points[:, 1] >= 0) & (image_points[:, 1] <= height)
    image_points = image_points[filter]

    print("Points from point cloud which belongs to the image:", image_points.shape)
    print("Labelled points on image:", len(image_points[image_points[:,2] >= 0]))

    # Define DPI (dots per inch)
    dpi = 100  # Common screen DPI; adjust as needed

    # Compute figure size in inches
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)

    for label_index, label in enumerate(labels):
        # Create a Rectangle patch
        rect = patches.Rectangle((label[0], label[1]), label[2] - label[0], label[3] - label[1], linewidth=1, edgecolor=COLORS_ARRAY[label_index % len(COLORS_ARRAY)], facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add text at position (x=50, y=50)
        ax.text(label[0], label[1], "{:d}".format(label_index), fontsize=12, color=COLORS_ARRAY[label_index % len(COLORS_ARRAY)], backgroundcolor='white')

    # Hide axes
    ax.axis("off")

    #plt.show(block=True)

    for point in image_points:
        label_index = int(point[2])
        color = 'white' if label_index < 0 else COLORS_ARRAY[label_index % len(COLORS_ARRAY)]
        size = 1 if label_index < 0 else 4
        ax.scatter(point[0], point[1], color=color, s=size)

    plt.savefig(os.path.join(OUTPUT_IMAGES_DIR, "{:s}.jpg".format(id)), format='jpg', dpi=200, bbox_inches='tight', pad_inches=0)


image_files = []
label_files = []

for (dirpath, dirnames, filenames) in walk(IMAGES_DIR):
    image_files.extend(filenames)
    break
print("Found", len(image_files), "images")

for (dirpath, dirnames, filenames) in walk(LABELS_DIR):
    label_files.extend(filenames)
    break
print("Found", len(label_files), "labels")

for image_file in image_files:
    id = re.search("\d+", image_file).group()

    if id != "014016":
        continue

    pc = get_sample_pc(id)
    if len(pc) == 0:
        print("Could not extract PC for sample", id)
        continue

    print("Initial PC: ", pc.shape)   
    
    # Keep only x,y,z
    pc = pc[:,[0,1,2]]
    # Transform lidar PC from the RADIal sane way as they do
    pc[:,[0, 1, 2]] = pc[:,[1, 0,2]] # Swap the order
    pc[:,0]*=-1 # Left is positive

    # marker for the labels. '-1' means the point does not belong to any label 
    no_labels = -1 * np.ones((pc.shape[0], 1))              
    pc = np.hstack((pc, no_labels))
    print("PC shape after modification: ", pc.shape)
    
    # Get 2D points from the point cloud to project onto the image
    points_2d,_ = cv2.projectPoints(np.array(pc[:,:3]), 
                                    CAMERA_TO_LIDAR_ROTATION, 
                                    CAMERA_TO_LIDAR_TRANSLATION,
                                    CAMERA_MATRIX,
                                    DISTORTION_COEFFICIENTS)

    points_2d = points_2d.squeeze(1).astype('int')
    print("2D points:", points_2d.shape)

    # Get the image and labels   

    image = cv2.imread(os.path.join(IMAGES_DIR, image_file)) 
    width = image.shape[1]
    height = image.shape[0]

    print("Image size:", width, "x", height)
    labels = read_lables(width, height, id)

    if labels.size == 0:
        print("No labels found for sample", id)
        continue

    print("Original PC shape:", pc.shape)
    pc = label_point_cloud(pc, points_2d, labels)  
    print("Labelled PC shape:", pc.shape)
    pc = cluster_pc(pc, len(labels), eps=0.35, ingore_z=True)   
    print("Clustered PC shape:", pc.shape)

    save_image(image, labels, pc, width, height, id)
    #save_range_azimuth(pc, len(labels), id)
    #show_range_azimuth(pc, len(labels), id)
   
  
    





    
