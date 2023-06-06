import numpy as np
import matplotlib.pyplot as plt

import cv2

def SkylineContourSegmentedImage(segmented_image: np.array, 
                                remove_coastline: bool = False, 
                                remove_noise:bool = True, 
                                noise_structuring_element: np.array = np.ones((3, 3), np.uint8), 
                                plot_results: bool = False):
    """
    SkylineContourSegmentedImage extracts the skyline contour from a semantic segmentation of terrain using
    a set of image-processing specific morphological operations. The resulting skyline is of width 1px.
    The function is capable of visualizing the results through the 'plot_results' flag.

    :param segmented_image: (np.array) 2D numpy array of the pixel data making up the semantic segmentation
    to be processed.
    :param remove_coastline: (bool = False) Boolean flag indicating wheter or not to remove the coastline
    of the terrain-segmentation. 
    :param remove_noise: (bool = True) Boolean flag indicating wheter or not to perform morphological
    operations to reduce noise in the image.
    :param noise_structuring_element: (np.array = np.ones((3, 3))) 2D numpy array denoting the
    structuring element used by noise-removal operations.
    :param plot_results: (bool = False) Boolean flag indicating wheter or not to visualize the results of
    the algorithm upon completion. 
    :return skyline_contour: (np.array) A 2D numpy array with the points of the contour. Each row indicates
    a point in the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    """ 
    # transform image into a binary image
    segmented_image_binary = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, segmented_image_binary = cv2.threshold(segmented_image_binary,127,255,cv2.THRESH_BINARY)

    # if 'remove_noise', perform open/close morphological operations to remove noise/fill holes
    if remove_noise:
        segmented_image_binary = cv2.morphologyEx(segmented_image_binary, cv2.MORPH_CLOSE, noise_structuring_element)
        segmented_image_binary = cv2.morphologyEx(segmented_image_binary, cv2.MORPH_OPEN, noise_structuring_element)

    # extract the skyline contour of the semantic segmentation by first eroding the image with a 3x3
    # cross-shaped structuring element before masking against the original image through a bitwise XOR
    # operation.
    erosion_kernel = np.array([[0, 1, 0], 
                               [1, 1, 1],
                               [0, 1, 0]], dtype = np.uint8)
    contour = cv2.bitwise_xor(segmented_image_binary, cv2.erode(segmented_image_binary, erosion_kernel, iterations = 1))
    contour = np.array([np.nonzero(contour)[1], np.nonzero(contour)[0]])    # remove non-zero elements and format
    contour = np.transpose(contour)

    # if 'remove_coastline', go through all columns of the contour and remove the bottom value where more
    # one exists. This is the highest value in OpenCV conventions. 
    if remove_coastline:
        # declare an array to contain points in the skyline
        skyline = []
        # go through all columns, returning all but the bottommost element of each. 
        for i in range(contour.shape[0]):
            # extract the current column (x-value)
            column_value = contour[i, 0]
            # find all rows (y-values) corresponding the column (x-value)
            row_index_same_column_value = np.where(contour[:, 0] == contour[i,0])[0]
            corresponding_rows = contour[row_index_same_column_value, 1]
            # if there are multiple rows containing the same column-values, save the one on top to the skyline array
            if corresponding_rows.shape[0] > 1:
                # remove the bottom row from the corresponding_rows list
                rows_to_keep = np.delete(corresponding_rows, np.argmax(corresponding_rows), axis = 0)
                # construct the matrix of remaining elements
                elements_to_keep = np.concatenate((np.full((corresponding_rows.shape[0]-1, 1), column_value), np.reshape(rows_to_keep, (len(rows_to_keep), 1))), axis = 1)[0]
                # add to the current skyline
                skyline.append(elements_to_keep)
            # else, return the only value
            else: 
                skyline.append([column_value, corresponding_rows[0]])

        # replace the current contour with the skyline w.o a coastline
        contour = np.array(skyline)

    # make sure the skyline only contains unique elements
    contour = np.unique(contour, axis = 0)

    # sort skyline by x-values
    contour = contour[contour[:, 0].argsort()]

    # if 'plot_results', visualize the resulting skyline contour on top of the semantic segmentation
    if plot_results:
        # instanciate the figure
        plt.figure()
        # show the semantic segmentation of the terrain
        plt.imshow(segmented_image, cmap='gray')
        # scatter the skyline on top of the terrain
        plt.scatter(x = contour[:, 0], y = contour[:, 1], s = 0.1, c = 'r', label = "Skyline contour")
        # show a legend defining objects in the image
        plt.legend(prop={'size': 6})
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')
        # show figure
        plt.show()
    
    return contour

def LocateHilltopsInSkylineContour(skyline_contour: np.array,
                                   hilltop_width_threshold: int,
                                   hilltop_height_threshold: int,
                                   skyline_horizontal_boundaries: np.array,
                                   plot_results: bool = False):
    """
    LocateHilltopsInSkylineContour is an extensive feature extraction function designed to extract hilltops 
    present in an 2D skyline contour for use as features describing the current pose of the maritime vessel.
    Hilltops are located by a top-down, depth first search of the skyline contour and adhere to the 
    following conditions:
        - Hilltops are convex hulls with ever decreasing sides. 
        - Hilltops are never located at the image boundary, as this would result in ambigouous features. 
        - Hilltops are not slopes, meaning the top of the hilltop is never at the the edge of the hilltop. 
        - Hilltops are minimum 'hilltop_width_threshold' wide.
        - Hilltops are minimum 'hilltop_height_threshold' high. 
        - Hilltops do not have any overlapping areas with other hilltops. 

    :param skyline_contour: (np.array) A 2D numpy array with the points of the contour. Each row indicates
    a point in the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    :param hilltop_width_threshold: (int) Integer denoting the minimum width of a hilltop.
    :param hilltop_height_threshold: (int) Integer denoting the minimum height of a hilltop.
    :param skyline_horizontal_boundaries: (np.array) A 1D numpy array containing the horizontal boundaries
    of the skyline image. Element 0 is the leftmost boundary and element 1 is the rightmost boundary. 
    :param plot_results: (bool = False) Boolean flag indicating wheter or not to visualize the results of
    the algorithm upon completion. 
    :return hilltops: (Dict{int, np.array}) A dictioary containing the extracted hilltops.
    """ 
    # Define an output dictionary to save the hilltops found in the skyline. A dictionary is used to
    # seamlessly store arrays of different lengths at different keys.
    # Also define a hilltop_counter to easily index new hilltops in the dictionary. 
    hilltops = {}
    hilltop_counter = 0

    # To extract hilltops, iterate through the rows of the image in descending order. Isolate hilltops one at a
    # time by iterating through neighboring, descending pixels until none can be found, then delete the
    # hilltop from the skyline not to evaluate the same elements again. 
    # NOTE: A copy of the skyline contour is used to ensure the original contour is un-altered. 
    skyline = skyline_contour.copy()

    skyline_min, skyline_max = np.min(skyline[:, 1]), np.max(skyline[:, 1])
    for row in range(skyline_min, skyline_max):
        # extract all points with the current row-value
        skyline_row = skyline[np.where(skyline[:, 1] == row), :][0]
        
        # don't consider the row if there are no elements remaining with the current row-value
        if not skyline_row.any():
            continue

        # Separate the skyline row into different subarrays based on if there is a jump in the column-
        # value from one pixel to the next. This indicates elements at different places in the skyline
        # -> different hilltops starting at the same vertical value. 
        skyline_row = skyline_row[skyline_row[:, 0].argsort()] # sort based on the column-values

        skyline_row_segments, last_pixel = [[]], skyline_row[0, :]
        for pixel in skyline_row:
            # if the next pixel is within a pixel length, add it to the current segment
            if np.abs(last_pixel[0] - pixel[0]) <= 1:
                skyline_row_segments[-1].append([pixel[0], pixel[1]])
            # if not, instanciate a new segment
            else:
                skyline_row_segments.append([[pixel[0], pixel[1]]])
            # update the last pixel
            last_pixel = pixel
        
        # Go through each segment and try to extract a hilltop. This is done by steadily iterating down the 
        # sides of the segment until it no longer descends. If the hilltop found satisfy the function thresholds
        # for heigth and width, it is added to the dictionary of hilltops and all pixels within the hilltop are
        # removed from the skyline contour as not to be repeated. 
        for segment in skyline_row_segments:
            # define boolean flags to cover certain cases
            hilltop_descending_right = True    # flag to see if hilltop is extending downwards to the right
            hilltop_descending_left = True     # flag to see if hilltop is extending downwards to the left
            hilltop_on_image_edge = False      # flag to see if hilltop connects to the bounds of the image

            # add to the hilltop as long as it's extending downwards in some direction
            while hilltop_descending_right or hilltop_descending_left:
                # find the left and rightmost extremity of the list
                leftmost_extremity = segment[0]
                rightmost_extremity = segment[-1]

                # check if the extremities intersect with the maximal bounds of the camera image. If
                # so, check the appropriate flag
                if not hilltop_on_image_edge and (leftmost_extremity[0] == skyline_horizontal_boundaries[0] or rightmost_extremity[0] == skyline_horizontal_boundaries[1]-1):
                    hilltop_on_image_edge = True

                # denote possible extensions of the hilltop in both directions by predefined neighborhoods.
                # If the edges of the segment are on the image bounds, do not generate neighborhoods.
                if leftmost_extremity[0] != skyline_horizontal_boundaries[0]:
                    possible_l_extension = [[leftmost_extremity[0] - 1, leftmost_extremity[1]],     # directly to the left
                                            [leftmost_extremity[0] - 1, leftmost_extremity[1] + 1], # diagonally to the left
                                            [leftmost_extremity[0], leftmost_extremity[1] + 1]]     # directly below
                else: 
                    possible_l_extension = []

                if rightmost_extremity[0] != skyline_horizontal_boundaries[1] - 1:
                    possible_r_extension = [[rightmost_extremity[0] + 1, rightmost_extremity[1]],     # directly to the right
                                            [rightmost_extremity[0] + 1, rightmost_extremity[1] + 1], # diagonally to the right
                                            [rightmost_extremity[0], rightmost_extremity[1] + 1]]     # directly below
                else: 
                    possible_r_extension = []

                # check if any of the extensions exist within the skyline contour. Done by conversions to
                # maps and sets to check for intersections. Update the flags according to which direction
                # pixels are added to the current hilltop
                skyline_set = set(map(tuple, skyline))
                possible_l_extension_map = map(tuple, possible_l_extension)
                possible_r_extension_map = map(tuple, possible_r_extension)

                l_extensions = list(set(skyline_set).intersection(set(possible_l_extension_map)))
                r_extensions = list(set(skyline_set).intersection(set(possible_r_extension_map)))

                # add any confirmed extensions to the segment. If no confirmed extensions exist, denote
                # that the segment is no longer descending in the corresponding direction by setting the 
                # appropriate flag.
                if hilltop_descending_left and l_extensions != []:
                    # add the new leftmost pixel of the hilltop to the start of the segment 
                    segment = [[l_extensions[0][0], l_extensions[0][1]]] + segment
                else: 
                    hilltop_descending_left = False

                if hilltop_descending_right and r_extensions != []:
                    # add the new rightmost pixel of the hilltop to the end of the segment
                    segment.append([r_extensions[0][0], r_extensions[0][1]]) 
                else: 
                    hilltop_descending_right = False
            
            # convert segment to numpy array for ease of indexation
            segment = np.array(segment)
            
            # Remove the current segment from the skyline. Done with an insanely fast algorithm
            # provided by the author below:
            # CREDIT: https://stackoverflow.com/a/40056251
            cumdims = (np.maximum(skyline.max(),segment.max())+1)**np.arange(segment.shape[1])
            skyline = skyline[~np.in1d(skyline.dot(cumdims),segment.dot(cumdims))]
            
            # check if the segment should be added as a hilltop. This is done in two steps for readability.
            # First, check if the hilltop is not located on the image edge and if it satisfies height and 
            # width thresholds.
            segment_height = np.abs(np.max(segment[:, 1]) - np.min(segment[:, 1]))
            segment_width = np.abs(np.max(segment[:, 0]) - np.min(segment[:, 0]))

            if hilltop_on_image_edge is False and segment_height > hilltop_height_threshold and segment_width > hilltop_width_threshold:
                # secondly, check if the segment is a slope. That is, either the leftmost or rightmost 
                # element is located at the top of the hilltop. 
                # NOTE: Since y is downwards in opencv images, the top of the hilltop is the smallest y-value.
                is_segment_slope = (segment[0, 1] == np.min(segment[:, 1]) or segment[-1, 1] == np.min(segment[:, 1]))
                if not is_segment_slope:
                    # if all requirements are met: Add the segment as a hilltop. 
                    hilltops[hilltop_counter] = segment
                    hilltop_counter += 1
    
    # if 'plot_results', visualize the skyline and the extracted hilltops of the skyline
    if plot_results:
        # instanciate the figure
        plt.figure()
        # instanciate subplot number 1
        plt.subplot(1, 2, 1).set_title("Skyline Contour", fontsize = 7)
        # show a blank canvas as a background
        plt.imshow(np.zeros((1080, 1920)), cmap='gray')
        # Show the original skyline contour as a scatterplot
        plt.scatter(x = skyline_contour[:, 0], y = skyline_contour[:, 1], s = 0.1, c = 'g')
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')

        # instanciate subplot number 2
        plt.subplot(1, 2, 2).set_title("Extracted hilltops", fontsize = 7)
        # show a blank canvas as a background
        plt.imshow(np.zeros((1080, 1920)), cmap='gray')
        # show the hilltopss found as scatterplots
        for hilltop in range(0, len(hilltops.keys())):
            plt.scatter(hilltops[hilltop][:, 0], hilltops[hilltop][:, 1], s = 1)
        # set axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')
        # show figure
        plt.show()

    return hilltops

def SkylineHilltopMatching(skyline_1_contour: np.array, 
                           skyline_2_contour: np.array,
                           hilltop_width_threshold: int,
                           hilltop_height_threshold: int,
                           skyline_horizontal_boundaries: np.array,
                           plot_results: bool = False):
    """
    SkylineHilltopMatching locates a set of matching features between two skyline contours. K hilltop-
    features are extracted from skyline_1_contour and matched areas of equal length are found in
    skyline_2_contour. The best of all K matches is returned. For each of the K hilltops, all possible
    feasible matches in skyline_2_contour are evaluated to find the most similar one. A feasible match 
    is considered a continous, unbroken set of pixels in skyline_2_contour, of the same length as the 
    currently evaluated hilltop. The deploymed similarity measure is a Sum of Absolute Differences (SAD). 

    :param skyline_1_contour: (np.array) A 2D numpy array with the points of the contour. Each row indicates
    a point in the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    :param skyline_2_contour: (np.array) A 2D numpy array with the points of the contour. Each row indicates
    a point in the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    :param hilltop_width_threshold: (int) Integer denoting the minimum width of a hilltop.
    :param hilltop_height_threshold: (int) Integer denoting the minimum height of a hilltop.
    :param skyline_horizontal_boundaries: (np.array) A 1D numpy array containing the horizontal boundaries
    of the skyline image. Element 0 is the leftmost boundary and element 1 is the rightmost boundary. 
    :param plot_results: (bool = False) Boolean flag indicating wheter or not to visualize the results of
    the algorithm upon completion. 
    :return best_match_skyline_1: (np.array) A 2D numpy array with the points of the match found in skyline_1.
    Each row indicates a point on the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    :return best_match_skyline_2: (np.array) A 2D numpy array with the points of the match found in skyline_2.
    Each row indicates a point on the image plane. Column 0 are column-values (x) and column 1 are row-values (y).
    """ 
    # Extract hilltops from the first skyline contour
    skyline_1_hilltops = LocateHilltopsInSkylineContour(skyline_1_contour, hilltop_width_threshold, hilltop_height_threshold, skyline_horizontal_boundaries, plot_results)

    # If no hilltops could be found in the first skyline contour, return a set of 0-values
    if skyline_1_hilltops == {}:
        print("NO HILLTOPS AVAILABLE IN SKYLINE 1")
        return None, None
    
    # Make sure the second skyline is sorted w.r.t column values
    skyline_2_contour = skyline_2_contour[skyline_2_contour[:, 0].argsort()] # sort based on the column-values

    # Preprovess the second skyline so as to not have any identical column values. I.e, only keep the
    # topmost value -> The lowest value in OpenCV conventions.
    # declare an intermediate array to contain points in the skyline
    skyline = []
    # go through all columns, returning only the bottom value of each 
    for i in range(skyline_2_contour.shape[0]):
        # extract the current column (x-value)
        column_value = skyline_2_contour[i, 0]
        # find all rows (y-values) corresponding the column (x-value)
        row_index_same_column_value = np.where(skyline_2_contour[:, 0] == skyline_2_contour[i,0])[0]
        corresponding_rows = skyline_2_contour[row_index_same_column_value, 1]
        # if there are multiple rows containing the same column-values, save the one on top to the skyline array
        if corresponding_rows.shape[0] > 1:
            # add to the current skyline
            skyline.append([column_value, corresponding_rows[np.argmin(corresponding_rows)]])
        else:
        # else, return the only value
            skyline.append([column_value, corresponding_rows[0]])


    # replace the current skyline_2_contour with the preprocessed skyline
    skyline_2_contour = np.array(skyline)

    # Find an approximate best match for each hilltop in skyline_1_hilltops and add them to a second dictionary.
    # Matches are quantitatively described by a Sum Of Absolute Differences (SAD), essentially the sum
    # of errors between two skylines. To find the best matching area, the hilltops are tested from left->right
    # on skyline_2_contour to find the area with the smallest SAD value. 
    # NOTE: The code is complicated by the fact that the second skyline can have discontinous pixel values.
    # Matching with a discontinous part of the skyline must be avoided.
    best_match_skyline_1, best_match_skyline_2 , best_sad = np.zeros((1, 1)), np.zeros((1, 1)), np.inf
    for key in skyline_1_hilltops.keys():
        # Extract the currently evaluated hilltop and it's length
        hilltop_skyline_1 = skyline_1_hilltops[key]

        # Preprovess the extracted hilltop so as to not have any identical column values. I.e, only keep the
        # topmost value -> The lowest value in OpenCV conventions.
        # declare an intermediate array to contain points of the hilltop
        hilltop = []
        # go through all columns of the hilltop, returning only the bottom value of each 
        for i in range(hilltop_skyline_1.shape[0]):
            # extract the current column (x-value)
            column_value = hilltop_skyline_1[i, 0]
            # find all rows (y-values) corresponding the column (x-value)
            row_index_same_column_value = np.where(hilltop_skyline_1[:, 0] == hilltop_skyline_1[i,0])[0]
            corresponding_rows = hilltop_skyline_1[row_index_same_column_value, 1]
            # if there are multiple rows containing the same column-values, save the one on top to the hilltop array
            if corresponding_rows.shape[0] > 1:
                # add to the current hilltop
                hilltop.append([column_value, corresponding_rows[np.argmin(corresponding_rows)]])
            else:
            # else, return the only value
                hilltop.append([column_value, corresponding_rows[0]])


        # replace the current skyline_2_contour with the preprocessed skyline
        hilltop_skyline_1 = np.array(hilltop)

        hilltop_skyline_1_length = hilltop_skyline_1.shape[0]

        # Iterate through skyline_2_contour. Done with a while-loop to quicky skip where discontinuities occur
        j = 0
        while j <= skyline_2_contour.shape[0] - hilltop_skyline_1_length:
            # Extract the area to be evaluated from the second skyline contour by an array-slice
            area_skyline_2 = skyline_2_contour[j:j+hilltop_skyline_1_length, :]
            # Check if this area contains one or more discontinous values. This is done by adding all 
            # column_values together and dividing with a linear arrangement from the first value. If
            # the product does not equal 0 we have a discontinuity. 
            sum_columns = np.sum(area_skyline_2[:, 0])
            sum_control = np.sum(np.arange(np.min(area_skyline_2[:, 0]), np.min(area_skyline_2[:, 0]) + hilltop_skyline_1_length))

            if sum_columns/sum_control != 1:
                # find out where the discontinuity happened and set j = first_pixel_after_discontinuity. 
                # values found by a smart solution using sets. 
                # CREDIT: https://stackoverflow.com/a/16974075
                initial_value, last_value = area_skyline_2[0, 0], area_skyline_2[-1, 0]
                first_gap = sorted(set(range(initial_value, last_value + 1)).difference(area_skyline_2[:, 0]))[0]
                first_gap_index = np.argwhere(skyline_2_contour[:, 0] == first_gap-1)[0][0]

                # Set the next index to be the one after the gap and continue
                j = first_gap_index + 1
                continue

            # if the area is continous, calculate the SAD value between that and the current hilltop
            sad = np.sum(np.abs(np.subtract(hilltop_skyline_1[:, 1], area_skyline_2[:, 1])))
            
            # if the SAD value is better than the previously recorded best, save the value and the matches
            if sad < best_sad:
                best_sad = sad
                best_match_skyline_1 = hilltop_skyline_1
                best_match_skyline_2 = area_skyline_2

            # increment the iterator
            j += 1

    # if 'plot_results', visualize the two matching areas on their respective skylines
    if plot_results:
        # instanciate figure
        plt.figure()
        # instanciate subplot for the match in skyline_1
        plt.subplot(1, 2, 1).set_title("Matched hilltop on intended skyline contour", fontsize = 7)
        # define a canvas and draw the entirety of skyline_1 onto it
        canvas = np.zeros((1080, 1920))
        canvas[skyline_1_contour[:,1], skyline_1_contour[:,0]] = 1
        plt.imshow(canvas, cmap = 'gray')
        # scatter the best match from skyline_1 on top of the canvas
        plt.scatter(best_match_skyline_1[:, 0], best_match_skyline_1[:, 1], s = 1)
        # define axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')

        # instanciate subplot for the match in skyline_2
        plt.subplot(1, 2, 2).set_title("Matching area in actual skyline", fontsize = 7)
        # define a canvas and draw the entirety of skyline_2 onto it
        canvas = np.zeros((1080, 1920))
        canvas[skyline_2_contour[:,1], skyline_2_contour[:,0]] = 1
        plt.imshow(canvas, cmap = 'gray')
        # scatter the best match from skyline_2 on top of the canvas
        plt.scatter(best_match_skyline_2[:, 0], best_match_skyline_2[:, 1], s = 1)
        # define axis labels
        plt.xlabel('[px]')
        plt.ylabel('[px]')

        # show the figure
        plt.show()        

    return best_match_skyline_1, best_match_skyline_2