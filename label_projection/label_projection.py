import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from os import walk
import re
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Calibration params
CAMERA_MATRIX = np.array([[1.84541929e+03, 0.0, 8.55802458e+02], [0.0, 1.78869210e+03, 6.07342667e+02], [0.0, 0.0, 1.0]]) 
DISTORTION_COEFFICIENTS = np.array([ 2.51771602e-01, -1.32561698e+01,  4.33607564e-03, -6.94637533e-03, 5.95513933e+01])    
CAMERA_TO_LIDAR_ROTATION = np.array([1.61803058,  0.03365624, -0.04003127])
CAMERA_TO_LIDAR_TRANSLATION = np.array([0.09138029, 1.38369885, 1.43674736])

# Whether to visualize radar point cloud (true) or lidar point cloud (false)
PROJECT_ON_RADAR = False

# The maximum elevation angle caught by radar
RADAR_MAX_ELEVATION_DEGREES = 12

# Whether to use polygon-shaped labels instead of rectangular bounding boxes
USE_POLYGON_LABLES = False

# When set to True, a biggest cluster is taken as a pole when doing clustering.
# Otherwise, the closest cluster is taken (with the smallest range) after which there is no higher cluster
USE_BIGGEST_CLUSTER = False

# Minumum height to be considered for the pole detection
MINIMUM_POLE_HEIGHT = 0.5

# Data location   
IMAGES_DIR = "/home/danil/RADIalHD/Radial_imagesHD"
LABELS_DIR = "/home/danil/RADIalHD/Radial_imagesHD_labels"
POLYGON_LABELS_DIR = "/home/danil/data/testFolder_labels_polygons"
LASER_PCL_DIR = "/home/danil/data/RADIal/laser_PCL"
RADAR_PCL_DIR = "/home/danil/data/RADIal/radar_PCL"
PREDICTED_LABELS_DIR = "/home/danil/data/RADIal/predicted_labels"
OUTPUT_IMAGES_DIR = "/home/danil/data/RADIal/projected_labels"
OUTPUT_LABELS_DIR = "/home/danil/data/RADIal/labelled_range_azimuth"


COLORS_ARRAY = np.array(['pink', 'red', 'green', 'blue', 'purple', 'orange'])

def belongs_to_polygon(polygon: np.array, point):
    return Path(polygon).contains_point(point)

def rotation2d(xyz,roll,yaw,pitch):
    
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    
    #xyz = np.hstack([xy,np.zeros( (len(xy),1))])
    
    rotation_vector = np.array([roll,pitch,yaw])
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(xyz)
    
    return rotated_vec[:,:3]

def pcl_to_range_azimuth(point_cloud, with_conversion):
    # Convert to polar coordinates (range and azimuth)
    ranges = np.sqrt(point_cloud[:, 0]**2 + point_cloud[:, 1]**2)  # Range
    azimuths = np.arctan2(point_cloud[:, 1], point_cloud[:, 0]) # Azimuth in degrees
    markers = point_cloud[:,3]

    # Normalize azimuths to [0, 360) for consistency
    #azimuths = (azimuths + 360) % 360

    # Convert to degrees, front is 0, left-hand are positive values, and right-hand are negative
    if with_conversion:
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

        print(" ------- Clustering points for label", label_index)
        clustering = None
        if ingore_z:
            points_2d = labelled_points[:,[0,1]]
            print("Points 2D:", points_2d.shape)
            clustering = DBSCAN(eps=eps, min_samples=2).fit(points_2d)
        else:    
            clustering = DBSCAN(eps=eps, min_samples=2).fit(labelled_points)
        cluster_labels = clustering.labels_
        
        print("Labels:", cluster_labels.shape)
        label_max = cluster_labels.max()
        if label_max == -1:
            print("Could not cluster, returning original point cloud")
            continue

        print("Label max: ", label_max)

        if USE_BIGGEST_CLUSTER:

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

        else:

            cluster_average_ranges = []
            cluster_heights = []
            cluster_sizes = []

            for i in range(0, label_max + 1):
                indicies = np.where(cluster_labels == i)[0]
                cumulative_range = 0.0
                min_height = 1000.0
                max_height = -1000.0
                for index in indicies:
                    point = labelled_points[index]
                    r = np.sqrt(point[0]**2 + point[1]**2)
                    cumulative_range += r
                    if point[2] < min_height:
                        min_height = point[2]
                    if point[2] > max_height:
                        max_height = point[2]

                cluster_size = len(indicies)
                cluster_average_ranges.append(cumulative_range/cluster_size)
                cluster_heights.append(max_height - min_height)
                cluster_sizes.append(cluster_size)

            best_cluster_index = 0
            closest_cluster_index = 0
            tallest_cluster_index = 0
            biggest_cluster_index = 0
            heaviest_cluster_index = 0

            min_range = min(cluster_average_ranges)
            max_height = max(cluster_heights)
            max_size = max(cluster_sizes)
            max_weight = 0

            for i, r in enumerate(cluster_average_ranges):
                h = cluster_heights[i]
                s = cluster_sizes[i]
                w = (h*s)/r
                if r == min_range:
                    closest_cluster_index = i
                if h == max_height:
                    tallest_cluster_index = i
                if s == max_size:
                    biggest_cluster_index = i  
                if w > max_weight:
                    max_weight = w
                    heaviest_cluster_index = i     
                print("Cluster ", i, ": range =", r, ", height =", h, ", size =", s, ", weight =", w)

            print("Nearest cluster:", closest_cluster_index, ", tallest cluster:", tallest_cluster_index, ", biggest cluster:", biggest_cluster_index, ", heaviest cluster:", heaviest_cluster_index)

            
            # if tallest_cluster_index == biggest_cluster_index:
            #     # (1) The biggest and tallest cluster is the firsts good candidate
            #     best_cluster_index = biggest_cluster_index    
            # else:
            #     # (2) If no such cluster, consider the cluster with the highest weight
            #     best_cluster_index = heaviest_cluster_index

            best_cluster_index = heaviest_cluster_index    

            print("Chose the best cluster:", best_cluster_index)    

            for j, cluster_label in enumerate(cluster_labels):
                if cluster_label != best_cluster_index:
                    pc[labelled_indicies[j],3] = -1 # Remove label    


    return pc  

def cluster_pc_with_region_growing(original_pc, num_labels, eps = 0.2):

    for i in range(0, num_labels): 

        marked_pc = original_pc[original_pc[:,3] == i]
        pc = np.transpose(np.array([marked_pc[:,0], marked_pc[:,1], marked_pc[:,2]]))

        # Convert the NumPy array to an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Access the computed normals
        normals = np.asarray(pcd.normals)    

        # Define parameters for region growing
        distance_threshold = eps  # Maximum distance between neighbors
        angle_threshold = np.pi / 8  # Maximum angle (in radians) between normals
        curvature_threshold = 0.1  # Curvature threshold (optional)

        # Compute curvatures (optional)
        # Note: Open3D does not directly compute curvature; here is a simple approximation
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        curvatures = np.zeros(len(pc))
        for i, point in enumerate(pc):
            [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)
            neighbors = pc[idx, :]
            # Ensure enough neighbors to compute covariance
            if len(neighbors) < 3:  # At least 3 points needed to compute a 2D covariance matrix
                curvatures[i] = 0  # Assign default curvature for isolated points
                continue

            # Compute covariance matrix of neighbors
            covariance_matrix = np.cov(neighbors - neighbors.mean(axis=0), rowvar=False)
            
            # Check if covariance_matrix is valid
            if covariance_matrix.shape != (3, 3):  # Ensure it's a 3x3 matrix
                curvatures[i] = 0  # Assign default curvature
                continue

            # Compute eigenvalues
            eigenvalues, _ = np.linalg.eigh(covariance_matrix)
            curvatures[i] = eigenvalues.min() / eigenvalues.sum()  # Curvature as a ratio of smallest eigenvalue

        # Initialize clustering
        visited = np.zeros(len(pc), dtype=bool)
        clusters = []

        # Run region growing
        for i in range(len(pc)):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True

                while queue:
                    current_idx = queue.pop(0)
                    cluster.append(current_idx)

                    # Find neighbors of the current point
                    [_, neighbor_indices, _] = kd_tree.search_radius_vector_3d(pc[current_idx], distance_threshold)
                    for neighbor_idx in neighbor_indices:
                        if not visited[neighbor_idx]:
                            # Check normal angle criterion
                            normal_angle = np.arccos(np.clip(np.dot(normals[current_idx], normals[neighbor_idx]), -1.0, 1.0))
                            if normal_angle < angle_threshold:
                                # Check curvature criterion (optional)
                                if curvatures[neighbor_idx] < curvature_threshold:
                                    queue.append(neighbor_idx)
                                    visited[neighbor_idx] = True

                if len(cluster) > 10:  # Minimum cluster size
                    clusters.append(cluster) 

        # Choose cluster
        max_cluster = None
        for cluster in clusters:
            print("Cluster ", len(cluster))  
            if max_cluster == None or len(cluster) > len(max_cluster):
                max_cluster = cluster

        if max_cluster == None:
            print("Could not cluster for label", i, ", skipping to the next label")
            continue          

        for index, e in enumerate(marked_pc):
            if index not in max_cluster:
                marked_pc[index,3] = -1      

        filter_out_labels = original_pc[:,3] != i
        original_pc = np.concatenate((original_pc[filter_out_labels], marked_pc), axis=0)  

    return original_pc                                                               

def get_sample_pc(id):       
    filename = os.path.join(RADAR_PCL_DIR if PROJECT_ON_RADAR else LASER_PCL_DIR, "pcl_{:s}.npy".format(id))
    return np.load(filename,allow_pickle=True)  

def read_lables(width, height, id):
    filename = os.path.join(LABELS_DIR, "image_{:s}.txt".format(id))       
    labels_data = np.loadtxt(filename)
    if (len(labels_data) == 0):
        return np.array([])
    if len(labels_data.shape) == 1:
        labels_data = np.array([labels_data])
    print("Labels: ", labels_data)
    labels = []
    for i, label in enumerate(labels_data):
        center_x = width * label[1]
        center_y = height * label[2]
        label_width = width * label[3]
        label_height = height * label[4]

        # For poles, skip wide labels as they are most probably irrelevant
        if label_width > label_height:
            continue

        box = [int(center_x - label_width/2), int(center_y - label_height/2), int(center_x + label_width/2), int(center_y + label_height/2)]
        labels.append(box)
    
    return np.array(labels)    

def read_polygon_labels(width, height, id):
    filename = os.path.join(POLYGON_LABELS_DIR, "image_{:s}.txt".format(id))  
    labels_data = np.genfromtxt(filename, dtype=object, delimiter="\n")
    labels = [list(map(float, row.split()[1:])) for row in labels_data]    

    # Convert labels from normalized to pixels
    for i, label in enumerate(labels):
        for j, point in enumerate(label):
            labels[i][j] = point * width if i%2 == 0 else point * height

    # Convert each row into a list of (x, y) points
    polygons = [np.array(row).reshape(-1, 2) for row in labels if len(row) % 2 == 0]
    # Convert each polygon into a Path object
    paths = [Path(polygon) for polygon in polygons]        
    return paths       


def label_point_cloud(pc, points_2d, labels):
    """
    Labels point cloud (pc) by checking the corresponding projected points (points_2d) to be inside the bounding box specified by each label
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

def label_point_cloud_for_polygons(pc, points_2d, polygons):
    """
    Labels point cloud (pc) by checking the corresponding projected points (points_2d) to be inside the specified polygon
    """    
    count_marked = 0

    for index, e in enumerate(points_2d): 
        for label_index, path in enumerate(polygons):
            if (path.contains_point(e)):
                pc[index, 3] = label_index  
                count_marked += 1

    print(count_marked, "points are inside the labelled windows") 
    return pc       

def show_range_azimuth(pc, num_labels, id):
    ra = pcl_to_range_azimuth(pc, False)
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

def show_predicted_range_azimuth(ra, id):

    az = ra[:,1]
    r = ra[:,0]

    # Visualize the point cloud in a polar plot
  
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': 'polar'})

    ax.set_thetamax(180)  
    ax.set_thetamin(0)
    
    ax.scatter(az, r, c='black', s=1)

    # Plot the selected region as a red mark
    plt.title("Sample {:s}".format(id))
    plt.show()     

def save_range_azimuth(pc, num_labels, id):
    f = open(os.path.join(OUTPUT_LABELS_DIR, "{:s}.txt".format(id)), "w")
    ra = pcl_to_range_azimuth(pc, True)
    print("Range-Azimuth", ra.shape)

    for i in range(0, num_labels):
        points_i = ra[ra[:,2] == i]
        if points_i.size == 0:
            continue
        r_i = points_i[:,0] 
        az_i = points_i[:,1]
        f.write("{:d} {:f} {:f}\n".format(i, np.average(r_i), np.average(az_i)))

    f.close()  

def save_image(image, labels, points_2d, markers, width, height, id):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      

    # Add markers to 2D points
    image_points = np.hstack((points_2d, markers.reshape(-1, 1)))
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


def save_image_with_polygons(image, paths, points_2d, markers, width, height, id):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      

    # Add markers to 2D points
    image_points = np.hstack((points_2d, markers.reshape(-1, 1)))
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

    for label_index, path in enumerate(paths):
        # Create a Rectangle patch
        patch = patches.PathPatch(path, linewidth=1, edgecolor=COLORS_ARRAY[label_index % len(COLORS_ARRAY)], facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(patch)
        # Add text at position (x=50, y=50)
        #ax.text(path., "{:d}".format(label_index), fontsize=12, color=COLORS_ARRAY[label_index % len(COLORS_ARRAY)], backgroundcolor='white')

    # Hide axes
    ax.axis("off")

    #plt.show(block=True)

    for point in image_points:
        label_index = int(point[2])
        color = 'white' if label_index < 0 else COLORS_ARRAY[label_index % len(COLORS_ARRAY)]
        size = 1 if label_index < 0 else 4
        ax.scatter(point[0], point[1], color=color, s=size)

    plt.savefig(os.path.join(OUTPUT_IMAGES_DIR, "{:s}.jpg".format(id)), format='jpg', dpi=200, bbox_inches='tight', pad_inches=0)    


# Converts range and azimuth to x,y,z.
# Here, z in the maximum elevation that potentially can be caught by radar for such range 
def range_azimuth_to_3d(ra):
    az = np.deg2rad(ra[:,1] + 90) # Do a reversed conversion, see pcl_to_range_azimuth()
    x = ra[:,0] * np.cos(az)
    y = ra[:,0] * np.sin(az)
    z = ra[:,0] * np.sin(np.deg2rad(RADAR_MAX_ELEVATION_DEGREES))
    return np.stack([x,y,z],axis=1)

# There are 11 parameters per point described sctructured as follow:
# [X, Y, Z, intensity, radialDistance, elevation_Angle, azimuth_angle, layer_index]
# Here is the trick, the Laser Scanner has 2 mirrors, one even and one odd. 
# There is a slight elevation angle difference between the 2 mirrors, so we have to compensate that angle
def compensate_layer_angle(pcl, index, sensor_height):
    
    offset=0
    if(index%2==0):
        offset = np.deg2rad(.6)

    x = pcl[:,4] * np.cos(pcl[:,5]+offset) * np.cos(pcl[:,6])
    y = pcl[:,4] * np.cos(pcl[:,5]+offset) * np.sin(pcl[:,6])
    z = pcl[:,4] * np.sin(pcl[:,5]+offset) + sensor_height
    
    pcl[:,0] = x
    pcl[:,1] = y
    pcl[:,2] = z
    
    return pcl

def process_labeled_images():
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

        if id != "000863":
            continue

        pc = get_sample_pc(id)
        print("PC point:", pc[0,:])
        if len(pc) == 0:
            print("Could not extract PC for sample", id)
            continue

        print("Initial PC: ", pc.shape)   

        if PROJECT_ON_RADAR:
            pc = np.stack([pc[5], pc[6], pc[7] + 0.7], axis=1)
            pc = rotation2d(pc,0,0,-2)
            x = -pc[:,1] # lateral
            y = pc[:,0] # longi
            z = pc[:,2] # longi
            pc = np.stack([x,y,z],axis=1)

        else:      
            # Keep only x,y,z
            pc = compensate_layer_angle(pc, 0, 0.42)[:,:3]
            
            # Transform lidar PC from the RADIal sane way as they do
            pc[:,[0, 1, 2]] = pc[:,[1, 0,2]] # Swap the order
            pc[:,0]*=-1 # Left is positive           
        
        print("PC point:", pc[0,:])
        # Get 2D points from the point cloud to project onto the image
        points_2d,_ = cv2.projectPoints(np.array(pc), 
                                        CAMERA_TO_LIDAR_ROTATION, 
                                        CAMERA_TO_LIDAR_TRANSLATION,
                                        CAMERA_MATRIX,
                                        DISTORTION_COEFFICIENTS)

        points_2d = points_2d.squeeze(1).astype('int')
        print("2D points:", points_2d.shape)

        print("Original PC shape:", pc.shape)

        # marker for the labels. '-1' means the point does not belong to any label 
        no_labels = -1 * np.ones((pc.shape[0], 1))   
        pc = np.hstack((pc[:,[0,1,2]], no_labels))
        print("PC shape before labelling: ", pc.shape)

        # Get the image and labels   

        image = cv2.imread(os.path.join(IMAGES_DIR, image_file)) 
        width = image.shape[1]
        height = image.shape[0]

        print("Image size:", width, "x", height)

        if USE_POLYGON_LABLES:
            paths = read_polygon_labels(width, height, id)
            if len(paths) == 0:
                print("No polygon labels found for sample", id)
                continue

            pc = label_point_cloud_for_polygons(pc, points_2d, paths)  
            print("Labelled PC shape:", pc.shape)
            save_image_with_polygons(image, paths, points_2d, pc[:,3], width, height, id)
            save_range_azimuth(pc, len(paths), id)
            show_range_azimuth(pc, len(paths), id)

        else:
            labels = read_lables(width, height, id)

            if labels.size == 0:
                print("No labels found for sample", id)
                continue

            pc = label_point_cloud(pc, points_2d, labels)  
            print("Labelled PC shape:", pc.shape)
            pc = cluster_pc(pc, len(labels), eps=0.2, ingore_z=True)   
            #pc = cluster_pc_with_region_growing(pc, len(labels), eps=0.2)  
            print("Clustered PC shape:", pc.shape)

            save_image(image, labels, points_2d, pc[:,3], width, height, id)
            save_range_azimuth(pc, len(labels), id)
            show_range_azimuth(pc, len(labels), id)


def project_predicted_labels():
    label_files = []

    for (dirpath, dirnames, filenames) in walk(PREDICTED_LABELS_DIR):
        label_files.extend(filenames)
        break
    print("Found", len(label_files), "labels")

    for label_file in label_files:
        id = re.search("\d+", label_file).group()
        ra = np.loadtxt(os.path.join(PREDICTED_LABELS_DIR, label_file))
        show_predicted_range_azimuth(ra, id)
        points_3d = range_azimuth_to_3d(ra)
        print("3D points:", points_3d.shape)  

        image = cv2.imread(os.path.join(IMAGES_DIR, "image_{:s}.jpg".format(id))) 
        width = image.shape[1]
        height = image.shape[0]

        # Get 2D points from the point cloud to project onto the image
        points_2d,_ = cv2.projectPoints(points_3d, 
                                        CAMERA_TO_LIDAR_ROTATION, 
                                        CAMERA_TO_LIDAR_TRANSLATION,
                                        CAMERA_MATRIX,
                                        DISTORTION_COEFFICIENTS)

        points_3d_no_elevation =  points_3d.copy()
        points_3d_no_elevation[:,2] = 0                             

        points_2d = points_2d.squeeze(1).astype('int')
        print("2D points:", points_2d.shape)

        points_2d_no_elevation,_ = cv2.projectPoints(points_3d_no_elevation, 
                                        CAMERA_TO_LIDAR_ROTATION, 
                                        CAMERA_TO_LIDAR_TRANSLATION,
                                        CAMERA_MATRIX,
                                        DISTORTION_COEFFICIENTS)
        points_2d_no_elevation = points_2d_no_elevation.squeeze(1).astype('int')                               
        print("2D points (no elevation):", points_2d_no_elevation.shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      

        # Define DPI (dots per inch)
        dpi = 100  # Common screen DPI; adjust as needed

        # Compute figure size in inches
        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(image)

        # Hide axes
        ax.axis("off")

        #plt.show(block=True)

        for i, point in enumerate(points_2d):
            if point[0] > 0 and point[0] < width and point[1] > 0 and point[1] < height:
                ground_point = points_2d_no_elevation[i,:]
                ground_x = ground_point[0] if ground_point[0] < width else width - 1
                ground_y = ground_point[1] if ground_point[1] < height else height - 1
                ax.plot([point[0], ground_x], [point[1], ground_y], color='green', linewidth=1, linestyle='-', alpha=0.4)

        plt.savefig(os.path.join(OUTPUT_IMAGES_DIR, "{:s}_.jpg".format(id)), format='jpg', dpi=200, bbox_inches='tight', pad_inches=0)
   


# Main program
process_labeled_images()    
#project_predicted_labels() 


  
    





    
