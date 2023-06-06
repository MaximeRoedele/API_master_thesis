import cv2

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.optimize import least_squares

def ReprojectWorldPoints(points: np.array,
                         camera_position: np.array, 
                         last_position:np.array,
                         camera_intrinsics: np.array, 
                         plot_results:bool = False, 
                         image_skyline_element: np.array = None):
    """
    ReprojectWorldPoints projects 3D points in the environment to a camera image plane using the cameras
    intrinsic and extrinsic parameters. To ease the problem of reconstructing the camera rendering from 
    Unity, all points and frames are transformed so as to match the left hand coordinate system present 
    during generation of the original data. Then, all points are transformed by a camera projection
    matrix before being subjected to a linear perspective projection.  

    :param points: (np.array) 2D numpy array containing the 3D points to be re-projected. 
    Each row is a point. Column 0 are x-values, column 1 are y values and column 2 are z-values, 
    or the elevation at (x, y). 
    :param camerea_position: (np.array) 1D numpy array containing the x and y coorinates of the camera. 
    :param last_position: (np.array) 1D numpy array containing the x and y coorinates of the previous 
    camera position. 
    :param camera_intrinsics: (np.array) A 2D numpy array representing the camera intrinsic matrix. 
    :param plot_results: (bool = False) Boolean flag indicating wheter or not to visualize the results of
    the algorithm upon completion.
    :param image_skyline_element: (np.array = None) Image skyline element to compare reprojection results
    with. 
    :return reprojected_pixels: (np.array) A 2D numpy array containing the UV coordinates of the 
    reprojected 3D points through the camera. 
    """ 
    # Declare the output array as a 2D array of same length as the input points
    reprojected_pixels = np.zeros((points.shape[0], 2))

    # Convert points from the XY plane to the XZ plane
    camera_position = np.array([[camera_position[0]], [0], [camera_position[1]]])
    last_position = np.array([[last_position[0]], [0], [last_position[1]]])
    points = points[:, [0, 2, 1]]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis = 1)
    
    # Calculate camera extrinsic matrix: 
    directional_vector = camera_position - last_position
    heading = np.arctan2(directional_vector[2], directional_vector[0])
    camera_R_wtheta = np.array([[-np.sin(heading), 0, np.cos(heading)],
                                [0, -1, 0],
                                [np.cos(heading), 0, np.sin(heading)]])
    
    camera_t = np.dot(-camera_R_wtheta, camera_position)

    extrinsic_matrix = np.concatenate((camera_R_wtheta, camera_t), axis = 1)

    # Multiply the intrinsic and extrinsic matrises to form the projection matrix
    projection_matrix = np.dot(camera_intrinsics, extrinsic_matrix)

    # Go through all points, reprojecting them to the image plane
    for i in range(0, points.shape[0]):
        pixel_coords = np.dot(projection_matrix, points[i, :])
        pixel_coords = pixel_coords/pixel_coords[2]
        reprojected_pixels[i, 0] = pixel_coords[0]
        reprojected_pixels[i, 1] = pixel_coords[1]

    # convert all pixels to integer values by rounding to the closest one.
    # NOTE: Rounding is done by np.rint, rounding an array elementwise to closest integer
    reprojected_pixels = np.rint(reprojected_pixels)

    if plot_results and image_skyline_element is not None:
        plt.figure()
        plt.subplot(2, 1, 1)
        # Show a blank canvas in the dimensions of the original image
        plt.imshow(np.zeros((1080, 1920)))
        plt.scatter(x = image_skyline_element[:, 0], y = image_skyline_element[:, 1], s = 1, c = 'g', label = "Original skyline contour")
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')
        plt.xlim((np.min(image_skyline_element[:, 0]), np.max(image_skyline_element[:, 0])))
        plt.ylim((np.max(image_skyline_element[:, 1]) + 30, np.min(image_skyline_element[:, 1]) - 30))
        # show a legend defining objects in the image
        plt.legend(prop={'size': 6})

        plt.subplot(2, 1, 2)
        plt.imshow(np.zeros((1080, 1920)))
        plt.scatter(x = reprojected_pixels[:, 0], y = reprojected_pixels[:, 1], s = 1, c = 'r', label = "Reprojected skyline contour")
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')
        plt.xlim((np.min(image_skyline_element[:, 0]), np.max(image_skyline_element[:, 0])))
        plt.ylim((np.max(image_skyline_element[:, 1]) + 30, np.min(image_skyline_element[:, 1]) - 30))
        # show a legend defining objects in the image
        plt.legend(prop={'size': 6})

        plt.show()

        plt.figure()
        plt.imshow(np.zeros((1080, 1920)))
        vertical_error = image_skyline_element[:, 1] - reprojected_pixels[:, 1]
        plt.errorbar(x = image_skyline_element[:, 0], y = image_skyline_element[:, 1], yerr = vertical_error, label = "Deviation of reprojection from original contour", ecolor='r', errorevery = 10)
        # show a legend defining objects in the image
        plt.legend(prop={'size': 6})
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')
        plt.xlim((np.min(image_skyline_element[:, 0]), np.max(image_skyline_element[:, 0])))
        plt.ylim((np.max(image_skyline_element[:, 1]) + 30, np.min(image_skyline_element[:, 1]) - 30))
        plt.show()
    return reprojected_pixels

def BundleAdjustmentPoseEstimation(world_points: np.array,
                                   desired_skyline_element: np.array,
                                   current_waypoint: np.array,
                                   last_waypoint: np.array, 
                                   camera_intrinsics: np.array,
                                   raster_dimensions: np.array):
    
    # optimize using scipy's nonlinear solver
    sol = least_squares(BundleAdjustmentReprojectionError, 
                        x0 = (current_waypoint[0], current_waypoint[1]), 
                        args = (world_points, desired_skyline_element, last_waypoint, camera_intrinsics), 
                        bounds = ([0, 0], [raster_dimensions[0], raster_dimensions[1]]), 
                        method = 'trf')  

    return sol

def BundleAdjustmentReprojectionError(parameters,
                                      world_points: np.array, 
                                      desired_skyline_element: np.array,  
                                      last_waypoint: np.array,
                                      camera_intrinsics: np.array):
    
    # reproject the world points to the current USV position given by 'parameters'
    reprojected_pixels = ReprojectWorldPoints(world_points,
                                              np.array([parameters[0], parameters[1]]),
                                              last_waypoint,
                                              camera_intrinsics)

    # calculate the reprojection error as the euclidean distance between each points. 
    # NOTE: This works due to previous algorithms sorting the output data in ascending x-values and
    # only keeping similar matches. Distance is then calculated for each respective point-pair. 
    distance_matrix = np.linalg.norm(reprojected_pixels - desired_skyline_element, axis = 1)
    total_error = np.sum(distance_matrix)

    return total_error

def GridSearchPoseEstimation(world_points: np.array, 
                             skyline_match_actual: np.array, 
                             current_waypoint: np.array,
                             last_waypoint: np.array, 
                             camera_intrinsics: np.array, 
                             error_gradient_threshold: float = 100, 
                             iteration_threshold: int = 1000):
    # initialize some necessary logistical variables
    iterations = 0
    error_improvement = np.inf

    # initialize the error as:
    reprojected_world_points = ReprojectWorldPoints(world_points, current_waypoint, last_waypoint, camera_intrinsics)
    distance_matrix = np.linalg.norm(reprojected_world_points - skyline_match_actual, axis = 1)
    error = np.sum(distance_matrix)

    # iterate until you have the best possible solution
    current_position = current_waypoint 
    while error_improvement > error_gradient_threshold and iterations < iteration_threshold:
        # find the other possible waypoints from the current waypoint
        possible_positions = np.mgrid[current_position[0]-1:current_position[0]+2:1, current_position[1]-1:current_position[1]+2:1].reshape(2, -1).T

        # go through each possible position, find the one with the greatest error_improvement
        error_improvement_temp = 0
        new_position = None
        for i in range(0, possible_positions.shape[0]):
            # extract the currently evaluated position
            evaluated_position = possible_positions[i]
            
            # reproject points and find the error
            reprojected_world_points = ReprojectWorldPoints(world_points, evaluated_position, last_waypoint, camera_intrinsics)
            error_temp = np.sum(np.abs(skyline_match_actual[:, 0] - reprojected_world_points[:, 0]) + np.abs(skyline_match_actual[:, 1] - reprojected_world_points[:, 1]))

            # if the error-improvement is the greatest so far, save the error and position
            if error - error_temp > error_improvement_temp:
                error_improvement_temp = error - error_temp
                new_position = evaluated_position
        
        if new_position.all() == None:
            print("Grid search converged")
            break

        # update the position and error improvement
        current_position = new_position
        error_improvement = error_improvement_temp
        
        iterations += 1

    return current_position

def AdaptiveGridSearchPoseEstimation(f_DEM: np.array, 
                                     f_actual: np.array,
                                     current_intended_waypoint: np.array, 
                                     last_actual_waypoint: np.array, 
                                     camera_intrinsic_matrix: np.array,
                                     error_gradient_threshold: float = 0.001,
                                     iteration_threshold: int = 1000, 
                                     plot_results: bool = False, 
                                     actual_position: np.array = np.zeros((1))):

    # initialize the error at the current waypoint
    reprojected_world_points = ReprojectWorldPoints(f_DEM, current_intended_waypoint, last_actual_waypoint, camera_intrinsic_matrix)
    # distance_matrix = reprojected_world_points[:, 0] - f_actual[:, 0] # Distance matrix only in x-direction
    distance_matrix = np.linalg.norm(reprojected_world_points - f_actual, axis = 1) # uniformly weighted euclidean distance
    error_curr = np.sum(distance_matrix)

    # Declare some data containers in case plotting is desired
    position_progression = [current_intended_waypoint]
    reprojection_progression = [reprojected_world_points]
    # iterate through a while loop till one or more conditions are satisfied
    i = 0
    error_last = np.inf
    neighborhood_radius = 32
    considered_position = current_intended_waypoint
    while np.abs(error_last-error_curr) > error_gradient_threshold and i <= iteration_threshold:  

        # define the considered vertices given the neighborhood radius and considered position
        grid = []
        grid.append(np.linspace((current_intended_waypoint[0] - neighborhood_radius//2, current_intended_waypoint[1] - neighborhood_radius//2),
                                (current_intended_waypoint[0] - neighborhood_radius//2, current_intended_waypoint[1] + neighborhood_radius//2),
                                neighborhood_radius+1))
        grid.append(np.linspace((current_intended_waypoint[0] + neighborhood_radius//2, current_intended_waypoint[1] - neighborhood_radius//2),
                                (current_intended_waypoint[0] + neighborhood_radius//2, current_intended_waypoint[1] + neighborhood_radius//2),
                                neighborhood_radius+1))
        grid.append(np.linspace((current_intended_waypoint[0] - neighborhood_radius//2, current_intended_waypoint[1] - neighborhood_radius//2),
                                (current_intended_waypoint[0] + neighborhood_radius//2, current_intended_waypoint[1] - neighborhood_radius//2),
                                neighborhood_radius+1))
        grid.append(np.linspace((current_intended_waypoint[0] - neighborhood_radius//2, current_intended_waypoint[1] + neighborhood_radius//2),
                                (current_intended_waypoint[0] + neighborhood_radius//2, current_intended_waypoint[1] + neighborhood_radius//2),
                                neighborhood_radius+1))
        
        grid = np.array(grid)
        grid = grid.reshape((grid.shape[0]*grid.shape[1], grid.shape[2]))
        
        # Go through all possible solutions for this iteration
        best_error_this_iteration = error_curr
        best_position_this_iteration = considered_position
        best_reprojection_this_iteration = np.zeros((1, 1))
        for potential_pos in grid:
            # calculate the reprojection error
            reprojected_world_points = ReprojectWorldPoints(f_DEM, potential_pos, last_actual_waypoint, camera_intrinsic_matrix)
            distance_matrix = np.linalg.norm(reprojected_world_points - f_actual, axis = 1)
            temp_error = np.sum(distance_matrix)
            
            # if the error is the best recorded this iteration, replace the best iteration
            if temp_error < best_error_this_iteration:
                best_error_this_iteration = temp_error
                best_position_this_iteration = potential_pos
                best_reprojection_this_iteration = reprojected_world_points

        # if no new solution was found, either:
        if np.abs(error_curr - best_error_this_iteration) == 0:
            # ... Lessen the radius
            if neighborhood_radius is not 2:
                neighborhood_radius = neighborhood_radius//2
                continue
            
            # ... or terminate
            else: 
                print("Grid search converged on position after", i, "iterations")
                break

        # update algorithm variables
        i += 1
        considered_position = best_position_this_iteration
        error_last = error_curr   
        error_curr = best_error_this_iteration

        # update data containers for plotting
        position_progression.append(considered_position)
        reprojection_progression.append(best_reprojection_this_iteration)

    if plot_results and actual_position.shape[0] == 2:
        # Convert the plotting data-containers into numpy arrays
        position_progression = np.array(position_progression)
        reprojection_progression = np.array(reprojection_progression)

        # define a gradient colormap
        cmap = cm.get_cmap('summer')

        plt.figure()

        plt.subplot(1, 2, 1).set_title("Reprojection progression", fontsize = 7)
        # plot a subset of reprojections from start to end
        for i in range(1, reprojection_progression.shape[0]):
            plt.scatter(x = reprojection_progression[i, :, 0], y = reprojection_progression[i, :, 1], s = 1, c = cmap(i/reprojection_progression.shape[0]))
        # plot the reprojections at the start
        plt.scatter(x = reprojection_progression[0][:, 0], y = reprojection_progression[0][:, 1], s = 1, c = 'g', label = "Initial f_dem reprojection")
        # plot the reprojections at the end
        plt.scatter(x = f_actual[:, 0], y = f_actual[:, 1], s = 1, c = 'r', label = "Target f_dem reprojection")

        plt.xlabel('[px]')
        plt.ylabel('[px]')
        plt.xlim((np.min(reprojection_progression[:, :, 0]) - 10 , np.max(reprojection_progression[:, :, 0]) + 10))
        plt.ylim((np.max(reprojection_progression[:, :, 1]) + 10, np.min(reprojection_progression[:, :, 1]) - 10))

        plt.legend(prop={'size': 6})

        plt.subplot(1, 2, 2).set_title("Position progression", fontsize = 7)
        plt.imshow(np.zeros((4000, 4000)))
        
        #plot the start and endpoint
        plt.scatter(x = position_progression[0, 0], y = position_progression[0, 1], s = 5, c = 'g', label = "Initial position")
        plt.scatter(x = actual_position[0], y =actual_position[1], s = 5, c = 'r', label = "Target position")

        # plot the progression from one 
        for i in range(0, position_progression.shape[0]-1):
            
            if i == position_progression.shape[0]-2:
                plt.scatter(x = position_progression[-1, 0], y = position_progression[-1, 1], s = 5, c = cmap(1.0), label = "Final estimation")
            else: 
                plt.scatter(x = position_progression[i+1, 0], y = position_progression[i+1, 1], s = 2, c = cmap((i+1)/position_progression.shape[0]), label = "Estimation iteration " + str(i + 1))
            
            plt.plot(position_progression[i:i+2, 0], position_progression[i:i+2, 1], linestyle = '--', linewidth = 1, color = cmap((i+1)/position_progression.shape[0]))
        
        plt.xlabel('[m]')
        plt.ylabel('[m]')
        plt.xlim((np.min(position_progression[:, 0]) - 10 , np.max(position_progression[:, 0]) + 10))
        plt.ylim((np.max(position_progression[:, 1]) + 10, np.min(position_progression[:, 1]) - 10))

        plt.legend(prop={'size': 6})

        plt.show()

    return considered_position