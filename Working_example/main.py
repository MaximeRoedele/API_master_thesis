# Import all necessary packages
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import time
import math

# Import utility functions
from data_utils import ConvertPathUnityOpenCV, LoadCapturePaths, BuildCameraIntrinsicMatrix
from image_utils import SkylineContourSegmentedImage, SkylineHilltopMatching, LocateHilltopsInSkylineContour
from raster_utils import DEMSkylineFromImageSkyline
from bundle_adjustment_utils import ReprojectWorldPoints, AdaptiveGridSearchPoseEstimation, GridSearchPoseEstimation

def ReprojectEntireSkyline(current_waypoint_number: int):
    # import all relevant data
    geotiff_path = '../data/dtm1_33_122_109_cropped.tif'
    path_waypoints = '../data/path_3/paths.txt'
    path_camera_parameters = '../data\path_3\cameraParameters.txt'
    path_segmentation = '../data/path_3/sequence.0'
    
    camera_intrinsics = BuildCameraIntrinsicMatrix(path_camera_parameters)
    h_fov = 70

    # Load raster data and extract the elevation band
    geotiff_data = rasterio.open(geotiff_path)
    terrain_elevation = geotiff_data.read(1)
    terrain_elevation[terrain_elevation < 0] = 0    # Set all negative values to 0

    # Load the paths to all segmentation images
    intended_segmentation_filepaths, actual_segmentation_filepaths = LoadCapturePaths(path_segmentation)

    # Load the intended and actual path
    path_intended, path_actual = ConvertPathUnityOpenCV(terrain_elevation, path_waypoints)

    # Define the current and last waypoint given the supplied waypoint number
    last_waypoint = path_actual[current_waypoint_number - 1]
    current_waypoint = path_intended[current_waypoint_number]
    
    # NOTE: As there are no images at the starting position (i=0), images at waypoint i are at index i-1. 
    capture_intended_waypoint = cv2.imread(intended_segmentation_filepaths[current_waypoint_number - 1])
    capture_actual_waypoint = cv2.imread(actual_segmentation_filepaths[current_waypoint_number - 1])

    # Extract the skyline contour from the intended image
    skyline_contour_intended = SkylineContourSegmentedImage(capture_intended_waypoint, remove_coastline = True, remove_noise = True, plot_results = False) 
    skyline_contour_actual = SkylineContourSegmentedImage(capture_actual_waypoint, remove_coastline = True, remove_noise = True, plot_results = False) 

    # Find matching skyline elements in the DEM
    dem_skyline_elements, skyline_contour_intended, _ = DEMSkylineFromImageSkyline(terrain_elevation, skyline_contour_intended, skyline_contour_actual, current_waypoint, last_waypoint, h_fov, 1920, plot_results = False)
    
    # Reproject DEM skyline elements to the USV camera
    ReprojectWorldPoints(dem_skyline_elements, current_waypoint, last_waypoint, camera_intrinsics, plot_results = True, image_skyline_element = skyline_contour_intended)

def ReprojectFeature(current_waypoint_number: int):
    # import all relevant data
    geotiff_path = '../data/dtm1_33_122_109_cropped.tif'
    path_waypoints = '../data/path_2/paths.txt'
    path_camera_parameters = '../data\path_2\cameraParameters.txt'
    path_segmentation = '../data/path_2/sequence.0'
    
    camera_intrinsics = BuildCameraIntrinsicMatrix(path_camera_parameters)
    h_fov = 70

    # Load raster data and extract the elevation band
    geotiff_data = rasterio.open(geotiff_path)
    terrain_elevation = geotiff_data.read(1)
    terrain_elevation[terrain_elevation < 0] = 0    # Set all negative values to 0

    # Load the paths to all segmentation images
    intended_segmentation_filepaths, actual_segmentation_filepaths = LoadCapturePaths(path_segmentation)

    # Load the intended and actual path
    path_intended, path_actual = ConvertPathUnityOpenCV(terrain_elevation, path_waypoints)

    # Define the current and last waypoint given the supplied waypoint number
    last_waypoint = path_actual[current_waypoint_number - 1]
    current_waypoint = path_intended[current_waypoint_number]
    
    # NOTE: As there are no images at the starting position (i=0), images at waypoint i are at index i-1. 
    capture_intended_waypoint = cv2.imread(intended_segmentation_filepaths[current_waypoint_number - 1])
    capture_actual_waypoint = cv2.imread(actual_segmentation_filepaths[current_waypoint_number - 1])

    # Extract the skyline contour from the intended image
    skyline_contour_intended = SkylineContourSegmentedImage(capture_intended_waypoint, remove_coastline = True, remove_noise = True, plot_results = False) 
    skyline_contour_actual = SkylineContourSegmentedImage(capture_actual_waypoint, remove_coastline = True, remove_noise = True, plot_results = False) 

    # Extract a matching feature between the two skylines:
    intended_skyline_match, actual_skyline_match = SkylineHilltopMatching(skyline_contour_intended, 
                                                                              skyline_contour_actual, 
                                                                              hilltop_width_threshold = 25,
                                                                              hilltop_height_threshold = 5, 
                                                                              skyline_horizontal_boundaries = np.array([0, capture_intended_waypoint.shape[1]]), 
                                                                              plot_results = False)

    # Find matching skyline elements in the DEM
    dem_skyline_elements, intended_skyline_match, _ = DEMSkylineFromImageSkyline(terrain_elevation, intended_skyline_match, actual_skyline_match, current_waypoint, last_waypoint, h_fov, 1920, plot_results = False)
    
    # Reproject DEM skyline elements to the USV camera
    ReprojectWorldPoints(dem_skyline_elements, current_waypoint, last_waypoint, camera_intrinsics, plot_results = True, image_skyline_element = intended_skyline_match)

def main():
    # Load data
    geotiff_path = '../data/dtm1_33_122_109_cropped.tif'
    path_waypoints = '../data/path_3/paths.txt'
    path_camera_parameters = '../data\path_3\cameraParameters.txt'
    path_segmentation = '../data/path_3/sequence.0'
    
    camera_intrinsics = BuildCameraIntrinsicMatrix(path_camera_parameters)
    h_fov = 70

    # Load the dem and extract the elevation band
    geotiff_data = rasterio.open(geotiff_path)
    terrain_elevation = geotiff_data.read(1)
    terrain_elevation[terrain_elevation < 0] = 0 # Set all negative values to 0

    # Load the paths to all segmentation images
    intended_segmentation_filepaths, actual_segmentation_filepaths = LoadCapturePaths(path_segmentation)

    # Load the intended and actual path
    path_intended, path_actual = ConvertPathUnityOpenCV(terrain_elevation, path_waypoints)

    # calculate the average error between the paths in euclidean distance
    dist = 0
    for e in range(1, path_actual.shape[0]):
        dist += np.linalg.norm(path_actual[e, :]-path_intended[e, :])
    
    print("Original MSE:", dist/(path_actual.shape[0] - 1))

    # Define a list in which to keep all indexes of the paths where no matches could be found
    infeasible_matches = []

    # Define lists for extracted times
    elapsed_time = []
    iterations = []

    # Loop thorugh all path points and estimate the pose of the USV
    estimated_positions = np.zeros(path_actual.shape)
    estimated_positions[0] = path_actual[0, :]
    
    for i in range(1, len(path_intended)):
        print("=== ON WAYPOINT", i, "===")
        # define the current and last waypoint
        last_waypoint = path_actual[i - 1]
        current_waypoint = path_intended[i]

        # load the USV images at the current waypoint of the intended and actual path
        # NOTE: As there are no images at the starting position (i=0), images at waypoint i are at index i-1. 
        capture_intended_waypoint = cv2.imread(intended_segmentation_filepaths[i-1])
        capture_actual_waypoint = cv2.imread(actual_segmentation_filepaths[i-1])

        # extract the skyline contours of both images
        skyline_contour_intended = SkylineContourSegmentedImage(capture_intended_waypoint, 
                                                                remove_coastline = True, 
                                                                remove_noise = True, 
                                                                plot_results = False) 
        
        skyline_contour_actual = SkylineContourSegmentedImage(capture_actual_waypoint, 
                                                              remove_coastline = True, 
                                                              remove_noise = True, 
                                                              plot_results = False) 

        intended_skyline_match, actual_skyline_match = SkylineHilltopMatching(skyline_contour_intended, 
                                                                              skyline_contour_actual, 
                                                                              hilltop_width_threshold = 25,
                                                                              hilltop_height_threshold = 5, 
                                                                              skyline_horizontal_boundaries = np.array([0, capture_intended_waypoint.shape[1]]), 
                                                                              plot_results = False)
        
        if intended_skyline_match is None or actual_skyline_match is None:
            print("No matches could be found!")
            infeasible_matches.append(i)
            continue
        
        dem_skyline_elements, intended_skyline_match, actual_skyline_match = DEMSkylineFromImageSkyline(terrain_elevation, 
                                                                                                        intended_skyline_match, 
                                                                                                        actual_skyline_match, 
                                                                                                        current_waypoint, 
                                                                                                        last_waypoint, 
                                                                                                        h_fov, 
                                                                                                        1920, 
                                                                                                        plot_results = False)
        estimation = AdaptiveGridSearchPoseEstimation(dem_skyline_elements, 
                                                                  actual_skyline_match, 
                                                                  current_waypoint, 
                                                                  last_waypoint, 
                                                                  camera_intrinsics,
                                                                  plot_results = False,
                                                                  actual_position = path_actual[i])
        
        estimated_positions[i, :] = np.array([estimation])
    
    # Print a scatterplot of the elapsed time used for the skyline contour algorithm
    # print("Mean elapsed time of skyline contour extraction: ", np.mean(np.array(elapsed_time)))
    # plt.figure()
    # plt.grid()
    # plt.scatter(x = np.arange(1, len(elapsed_time) + 1), y = elapsed_time, s = 3, c = 'b', label = 'Elapsed time at instance')
    # plt.axhline(y = np.mean(np.array(elapsed_time)), c = 'r', linestyle = '--', label = 'Mean elapsed time')
    # plt.xlabel("Waypoint")
    # plt.ylabel("Time elapsed [s]")
    # plt.xticks(range(1, math.ceil(len(elapsed_time)) + 1))
    # plt.legend(prop={'size': 8})
    # plt.show()

    # remove infeasible matches if there exist any
    if len(infeasible_matches) > 0:
        estimated_positions = np.delete(estimated_positions, np.array(infeasible_matches), axis = 0)
        path_actual = np.delete(path_actual, np.array(infeasible_matches), axis = 0)
        print("Amount of infeasible matches:", len(infeasible_matches))

    # Plot the errors at each waypoint i
    plt.figure()
    distance_matrix = np.linalg.norm(estimated_positions[1:, :] - path_actual[1:, :], axis = 1)
    MSE = np.sum(distance_matrix)/(estimated_positions.shape[0] - 1)

    plt.scatter(x = np.arange(1, estimated_positions.shape[0]),
                y = distance_matrix,
                  s = 3, c = 'b', label = 'Estimation error at each waypoint')
    plt.axhline(y = MSE, c = 'r', linestyle = '--', label = 'MSE of the path')
    print(MSE)
    plt.xlabel("Waypoint")
    plt.ylabel(r'$||\hat{p}_i-\vec{p}_i||_2$')
    plt.legend(prop={'size': 8})
    plt.xticks(range(1, math.ceil(estimated_positions.shape[0])))
    plt.grid()
    plt.show()

    # Plot the results
    plt.figure()
    plt.imshow(terrain_elevation, cmap = 'gray')
    plt.colorbar(label = "Altitude above sealevel")

    # plot the intended USV path
    plt.scatter(x = path_intended[:, 0], y = path_intended[:, 1], c = 'g', s = 1, label = 'Intended waypoints')
    # plot the actual USV path
    plt.scatter(x = path_actual[:, 0], y = path_actual[:, 1], c = 'y', s = 1, label = 'Actual waypoints')
    # plot the estimated USV path
    plt.scatter(x = estimated_positions[:, 0], y = estimated_positions[:, 1], c = 'r', s = 1, label = 'Estimated waypoints')

    # display a line between the estimated and actual waypoints, denoting the error
    for i in range(0, estimated_positions.shape[0]):
        if i == 0:
            plt.plot(np.array([estimated_positions[i, 0], path_actual[i, 0]]), np.array([estimated_positions[i, 1], path_actual[i, 1]]), 'r-', label = "Estimation error")
        else: 
            plt.plot(np.array([estimated_positions[i, 0], path_actual[i, 0]]), np.array([estimated_positions[i, 1], path_actual[i, 1]]), 'r-')

    # Display a legend to make sense of the results
    plt.legend(prop={'size': 6})
    # Plot axis
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    # Plot the figure
    plt.show()

    return

#ReprojectEntireSkyline(19)
#ReprojectFeature(2)
main()