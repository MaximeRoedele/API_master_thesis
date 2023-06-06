import numpy as np

import os
import collections
import re

def ConvertPathUnityOpenCV(elevation_data: np.array, 
                            path_waypoints: str):
    """
    ConvertPathUnityOpenCV converts the waypoints of the maritime vessel from Unity's left hand coordinate
    system to OpenCVs right hand coordinate system. It also makes sure they fit inside the mesh-boundaries.

    :param elevation_data: (np.array) A 2D numpy array denoting the elevation of the terrain surrounding the
    USV. elevation_data[row, column] = elevation_data[y, x] = elevation_yx
    :param path_waypoints: (string) The path to the .txt file containing the path-waypoints in Unity
    conventions. 
    :return intended_path: (np.array) A 2D numpy array with waypoints along the intended path. Each
    row denotes a separate point. Column 0 are x-values and column 1 are y-values. 
    :return actual_path: (np.array) A 2D numpy array with waypoints along the actual path. Each
    row denotes a separate point. Column 0 are x-values and column 1 are y-values. 
    """ 
    # Load the intended path as a list of strings
    path = np.genfromtxt(path_waypoints, dtype='str')

    # Define output-arrays 
    intended_path = path[0:path.shape[0]//2, :].astype(int)
    actual_path = path[path.shape[0]//2:, :].astype(int)

    # Get bounds of the elevation-data
    geotiff_bounds = elevation_data.shape

    # Adjust the paths form Unity-conventions to openCV conventions
    intended_path[:, 0] = np.ones((1, intended_path.shape[0]))[0]*geotiff_bounds[0]//2 - intended_path[:, 0]
    intended_path[:, 1] = np.ones((1, intended_path.shape[0]))[0]*geotiff_bounds[1]//2 + intended_path[:, 1]

    actual_path[:, 0] = np.ones((1, actual_path.shape[0]))[0]*geotiff_bounds[0]//2 - actual_path[:, 0]
    actual_path[:, 1] = np.ones((1, actual_path.shape[0]))[0]*geotiff_bounds[1]//2 + actual_path[:, 1]

    return intended_path, actual_path

def LoadCapturePaths(path_segmented_images_folder: str, 
                    filetype: str = 'png'):
    """
    LoadCapturePaths generates two numpy arrays of filepaths for semantic segmentations captures at 
    the intended and actual waypoints of the maritime vessel, in order of capture. Assumes all 
    captures are in one folder and that there is an equal amount of captures.

    :param path_segmented_images_folder: (string) filepath to the folder containing *all* captures.
    :param filetype: (string = 'png') The filetype of the captures, to distinguish between images and
    meta-data files.
    :return intended_segmentation_filepaths: (np.array) A 1D numpy array of filepaths to semantic segmentations
    along the intended USV path in order. 
    :return actual_segmentation_filepaths: (np.array) A 1D numpy array of filepaths to semantic segmentations
    along the actual USV path in order.  
    """ 
    # Define dictionary for ease of sorting later on
    segmentation_pathdict = {}

    #Loop through all data, extracting the image-paths
    for root, _ , files in os.walk(path_segmented_images_folder):
        for name in files:
            # Only assess files of type 'filetype'
            if str(name[-len(filetype):]) == filetype:
                    # find the number of the segmented image
                    key = int(re.findall(r"\d+", name)[0])
                    # use this number as a key for the segmentation dictionary. Use the image path as the value
                    segmentation_pathdict[key] = os.path.join(root, name)

    # Sort the dictionary based on the keys
    segmentation_pathdict = list(collections.OrderedDict(sorted(segmentation_pathdict.items())).values())

    # Split the dictionary into 2 lists
    intended_segmentation_filepaths = segmentation_pathdict[0:len(segmentation_pathdict)//2]
    actual_segmentation_filepaths = segmentation_pathdict[(len(segmentation_pathdict)//2):len(segmentation_pathdict)]

    return np.array(intended_segmentation_filepaths), np.array(actual_segmentation_filepaths)

def BuildCameraIntrinsicMatrix(path_camparameters: str):
    """
    BuildCameraIntrinsicMatrix loads a list of camera parameters from a txt file and converts it into a 
    camera intrinsic matrix for a physical camera. The list is formatted as follows:

        [camera focal length [mm], camera sensor size x [mm], camera sensor size y[mm], 
         actual image width [pixels], actual image heigth [pixels], camera principal point x [pixels],
         camera principal point y [pixels]]

    The output camera intrinsic matrix is on the following format: 
             | f_x   0   o_x |
        K =  |  0   f_y  o_y |
             |  0    0    0  |

    :param path_camparameters: (string) filepath to the .txt file containing the camera parameters.
    :return camera_matrix: (np.array) A 2D numpy array representing the camera intrinsic matrix.
    """ 
    # Load the file containing the camera parameters to a list of strings
    cam_parameters = np.genfromtxt(path_camparameters, dtype='str')

    # Replace all ',' with '.' for correct conversion in the next step
    cam_parameters = np.char.replace(cam_parameters, ",", ".")
        
    # convert the matrix to a numpy array of floats
    cam_parameters = cam_parameters.astype(float)

    # extract camera_parameters (for readability)
    cam_focal_length = cam_parameters[0]
    cam_sensor_size_x = cam_parameters[1]
    cam_sensor_size_y = cam_parameters[2]
    cam_pixel_width = cam_parameters[3]
    cam_pixel_height = cam_parameters[4]
    cam_principal_point_x = cam_parameters[5]
    cam_principal_point_y = cam_parameters[6]

    # Define the camera intrinsic matrix from the parameters
    camera_matrix = np.zeros((3, 3), dtype=float)
    camera_matrix[0, 0] = cam_focal_length*(cam_pixel_width/cam_sensor_size_x)
    camera_matrix[1, 1] = cam_focal_length*(cam_pixel_height/cam_sensor_size_y)
    camera_matrix[0, 2] = cam_principal_point_x
    camera_matrix[1, 2] = cam_principal_point_y
    camera_matrix[2, 2] = 1.0

    return camera_matrix
