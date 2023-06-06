import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from matplotlib.path import Path
from math_utils import AngularAddition, ProjectVectorToImageBorder, DrawLine2D, HorizontalCameraFrustumProjectionToImageEdge, IdentifyImageCornersInCameraFrustum

def DEMSkylineFromImageSkyline(elevation: np.array, 
                               image_skyline_element_intended: np.array,
                               image_skyline_element_actual: np.array,
                               current_waypoint: np.array, 
                               last_waypoint: np.array,  
                               camera_horizontal_FOV: float,
                               camera_horizontal_dimension: int, 
                               plot_results: bool = False):
    """
    DEMSkylineFromImageSkyline extracts the 3D position of skyline elements in an image skyline element, 
    which can be part of, or the entire skyline captured from a camera. An important assumption is that
    the skyline element has a consistent width of 1 pixel, which can be found through f.ex 
    image_utils.SkylineContourSegmentedImage.
    The location of the skyline in 3D is done by iterating through the N pixels of the skyline element,
    drawing a ray into the DEM per nonzero pixel. Each line is masked with the DEM to form a cross-section
    of the elevation profile along that line. The nonzero point with the highest angle from 
    'current_waypoint' is selected to be the 3D position of the skyline element inside the DEM. 

    :param elevation: (np.array) 2D numpy array containing the heightmap information of the terrain
    surrounding the USV. 
    :param image_skyline_element_intended: (np.array) A 2D numpy array with the points of the skyline element. 
    Each row indicates a point in the image plane. Column 0 are column-values (x) and column 1 are 
    row-values (y).
    :param current_waypoint: (np.array) 1D numpy array containing the x and y values of the current
    USV position along the linear path. USV coordinates are assumed to be on OpenCV conventions. 
    :param last_waypoint: (np.array) 1D numpy array containing the x and y values of the previous
    USV position along the linear path. USV coordinates are assumed to be on OpenCV conventions. 
    :param camera_horizontal_FOV: (float) floating point value indicating the horizontal Field Of 
    View of the camera aboard the USV.
    :param camera_horizontal_dimension: (int) The horizontal size of the image. 
    :param plot_results: (bool = False) Boolean flag indicating wheter or not to visualize the results of
    the algorithm upon completion.
    :return dem_skyline_elements: (np.array) 2D numpy array containing the 3D points matching the
    supplied skyline element. Each row is a point. Column 0 are x-values, column 1 are y values and 
    column 2 are z-values, or the elevation at (x, y). 
    """ 
    # Define an output array for the skyline elements
    dem_skyline_elements = []

    # Calculate the current heading of the USV
    directional_vector = current_waypoint - last_waypoint
    heading = np.arctan2(directional_vector[1], directional_vector[0])

    # iterate through the horizontal dimension of the camera image. For each pixel (x-value in the image) 
    # draw a line in the raster data and fetch the elevations at each point along this line. Extract the
    # point along this elevation cross-section with the highest angle (i.e the point that will be visible
    # on the skyline). 
    # NOTE: Iterating through the image from left -> right = iterating through the elevation map clockwise.
    # The initial angle is then *always* the heading minus half the horizontal FOV.
    # NOTE 2: Note the use for the helper-function DivideCameraFrustumToAngles to calculate the
    # appropriate angular deviation from one line to the next. This is necessary as the angles are not
    # uniform.
    frustum_line_angles = DivideCameraFrustumToAngles(camera_horizontal_dimension, camera_horizontal_FOV, heading)   

    for i in range(0, camera_horizontal_dimension):
        # first, check if the skyline contains an element at x-value = i
        if(i not in image_skyline_element_intended[:, 0]):
            continue
        
        # find the projection of the current line to the boundary of the elevation-data
        line_angle = frustum_line_angles[i]

        line_elevation_boundary_projection = ProjectVectorToImageBorder(elevation, current_waypoint, line_angle)
        
        # find the elevation cross-section by spatially defining the line from the current waypoint to the 
        # boundary of the elevation map using a line-drawing algorithm. Then, mask the elevation map to fit
        # the line.
        line = DrawLine2D(current_waypoint, line_elevation_boundary_projection)
        elevation_cross_section = elevation[line[:, 1], line[:, 0]]

        # extract all non-zero elevations along this line.
        elevation_cross_section_nonzero_indexes = np.nonzero(elevation_cross_section)[0]

        # if the line contain heights in the elevation map that are non-zero, find which 3D point has
        # the greatest angle to the current point. This should be the visible skyline element along the line.
        if elevation_cross_section_nonzero_indexes != []:
            # isolate an x and z value for each elevation point. x is the euclidean distance from the current
            # waypoint at height 0 to the 0-height value of the elevation point and z is the height of the point.
            
            # find the x-values by finding the relevant coordinates of the cross section on the raster map
            line = line[elevation_cross_section_nonzero_indexes, :]

            # calculate the x value as the euclidean distance of all these from the current waypoint
            x = np.abs(np.linalg.norm(line - np.full(line.shape, current_waypoint), axis = 1))
            
            # define the z-values as the non-zero heights along the elevation-cross section
            z = elevation_cross_section[elevation_cross_section_nonzero_indexes]

            # calculate the angles of all x-z pairs
            angles = np.arctan2(z, x)

            # append the point with the greatest angle to be the skyline element found along the line
            highest_angle = np.argmax(angles)
            x = line[highest_angle, 0]
            y = line[highest_angle, 1]
            z = elevation[y, x]

            dem_skyline_elements.append(np.array([x, y, z]))
        
        # if the pixel exists in the skyline element but not in the line drawn from the DEM, remove the 
        # skyline element and the potential match. This is to avoid mismatches and keep the number of
        # matching 2D and 3D features the same. 
        else:
            # find the index of the row to remove
            index = np.argwhere(image_skyline_element_intended[:, 0] == i)[0][0]
            
            # delete the item from the intended skyline element
            image_skyline_element_intended = np.delete(image_skyline_element_intended, index, axis = 0)

            # and from the matching skyline element in the actual image
            image_skyline_element_actual = np.delete(image_skyline_element_actual, index, axis = 0)

            continue    
    # convert the dem_skyline indices to a numpy array
    dem_skyline_elements = np.array(dem_skyline_elements)

    # if 'plot_results', visualize the resulting skyline elements and the lines of sight from the USV
    if plot_results:

        USV_direction_indicator = np.array([last_waypoint,
                                            current_waypoint])
        plt.figure()

        plt.subplot(1, 3, 1).set_title("Original elevation data w. 3 DOF pose of vessel", fontsize = 7)
        plt.imshow(elevation, cmap = 'gray')
        plt.arrow(USV_direction_indicator[0, 0], USV_direction_indicator[0, 1], USV_direction_indicator[1, 0] - USV_direction_indicator[0, 0], USV_direction_indicator[1, 1] - USV_direction_indicator[0, 1], head_width  = 20, ec = 'cyan')


        plt.subplot(1, 3, 2).set_title("Rays from vessel and extracted 3D coordinates", fontsize = 7)
        plt.imshow(elevation, cmap = 'gray')
        plt.arrow(USV_direction_indicator[0, 0], USV_direction_indicator[0, 1], USV_direction_indicator[1, 0] - USV_direction_indicator[0, 0], USV_direction_indicator[1, 1] - USV_direction_indicator[0, 1], head_width  = 20, ec = 'cyan')

        # plot the located skyline-elements and rays
        for i in range(0, len(dem_skyline_elements)):
            plt.plot([current_waypoint[0], dem_skyline_elements[i, 0]], [current_waypoint[1], dem_skyline_elements[i, 1]], color = 'blue', zorder = 0)
        plt.scatter(x = dem_skyline_elements[:, 0], y = dem_skyline_elements[:, 1], s=1, c = 'r', zorder = 1)

        plt.subplot(1, 3, 3).set_title("Vessel and extracted 3D coordinates", fontsize = 7)
        plt.imshow(elevation, cmap = 'gray')
        plt.arrow(USV_direction_indicator[0, 0], USV_direction_indicator[0, 1], USV_direction_indicator[1, 0] - USV_direction_indicator[0, 0], USV_direction_indicator[1, 1] - USV_direction_indicator[0, 1], head_width  = 20, ec = 'cyan')

        plt.scatter(x = dem_skyline_elements[:, 0], y = dem_skyline_elements[:, 1], s=1, c = 'r', zorder = 1)

        plt.show()
    
    return dem_skyline_elements, image_skyline_element_intended, image_skyline_element_actual

def DivideCameraFrustumToAngles(N: int, FOV_h: float, heading: float):
    # Define the output list: 
    subdivision_angles = np.zeros(N)

    # find the angle of the first line. 
    # NOTE: Assuming that we iterate through the image from left -> right = iterating through the elevation
    # map clockwise, the initial angle is *always* the heading minus half the horizontal FOV.
    half_horizontal_camera_FOV = (FOV_h*np.pi/180)/2   # Half of FOV_h in rads
    initial_angle_rad = AngularAddition(heading, -half_horizontal_camera_FOV, degrees = False) 

    subdivision_angles[0] = initial_angle_rad

    # Subdivision algorithm using subtriangles and the cosine sentence to extract angular deviations.
    # For a thourough description of this process, consult the report accomodating this software. 
    # Calculate the image plane subdivision length
    l = (np.tan(half_horizontal_camera_FOV)*2)/(N-1)

    for i in range(1, N):
        # calculate the length along the image plane for the previous and current line
        l_prev, l_curr = l * (i-1), l * i

        # Use these to calculate the current and previous lines
        if l_prev > np.tan(half_horizontal_camera_FOV):
            d_prev = np.sqrt(1 + (l_prev - np.tan(half_horizontal_camera_FOV))**2)
        else: 
            d_prev = np.sqrt(1 + (np.tan(half_horizontal_camera_FOV) - l_prev)**2)

        if l_curr > np.tan(half_horizontal_camera_FOV):
            d_curr = np.sqrt(1 + (l_curr - np.tan(half_horizontal_camera_FOV))**2)
        else: 
            d_curr = np.sqrt(1 + (np.tan(half_horizontal_camera_FOV) - l_curr)**2)
        
        # calculate the angular addition from the last subdivided triangle
        divisor = d_prev**2 + d_curr**2 - l**2
        denominator = 2*d_prev*d_curr
        subdivision_angles[i] = AngularAddition(np.arccos(divisor/denominator), subdivision_angles[i-1])

    return subdivision_angles