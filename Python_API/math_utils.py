import numpy as np

def AngularAddition(original_angle: float, 
                    additional_angle: float, 
                    degrees: bool = False):
    
    # naively add angles together
    angle = original_angle + additional_angle
    
    # adjust range to pi to -pi - range
    if angle > np.pi:
        angle -= 2*np.pi
    
    if angle < -np.pi:
        angle += 2*np.pi

    # return the new angle as radians or degrees
    if degrees: 
        return angle * 180/np.pi

    return angle

def AngularDifference(angle_1: float, 
                      angle_2: float, 
                      absolute_angle: bool = False,
                      degrees: bool = False):
    # calculate angular deviation in radians
    a = angle_2 - angle_1
    rad = (a + np.pi) % (2*np.pi) - np.pi
    
    # given the flag 'absolute_angle', return the absolute difference or directional difference
    if absolute_angle:
         rad = np.abs(rad)

    # return according to boolean flag "degrees"
    if degrees: 
        return rad*180/np.pi
    
    return rad

# CREDIT: https://jccraig.medium.com/we-must-draw-the-line-1820d49d19dd
def DrawLine2D(origin: np.array, end: np.array):
    # declare output
    line = []

    # unwrap input
    x1, y1 = origin[0], origin[1]
    x2, y2 = end[0], end[1]
    
    x = x1
    y = y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1 if x1 > x2 else 0
    sy = 1 if y1 < y2 else -1 if y1 > y2 else 0
    ix = dy // 2
    iy = dx // 2
    pixels = dx + 1 if dx > dy else dy + 1
    while pixels:
        line.append([x, y])
        ix += dx
        if ix >= dy:
            ix -= dy
            x += sx
        iy += dy
        if iy >= dx:
            iy -= dx
            y += sy
        pixels -= 1
    
    return np.array(line, dtype=int)[:-1, :]

def ProjectVectorToImageBorder(image: np.array,
                               point: np.array,
                               theta: float):
    # Calculate unit-vector from origo in the direction theta
    unit_directional_vector = np.array([np.cos(theta), np.sin(theta)])

    # Add the unit vector to the point until it breaches one of the image bounds
    out_of_bounds = False
    current_vector = np.array([float(point[0]), float(point[1])])
    while not out_of_bounds:
        # check if any image-bounds are broken by the unit-vector addition
        if (current_vector[0] < 0 or current_vector[1] < 0 or current_vector[0] > image.shape[0] or current_vector[1] > image.shape[1]):
            # go back one step to be within bounds
            current_vector -= unit_directional_vector
            out_of_bounds = True
        # if no bounds are broken, add the unit vector again    
        else:
            current_vector += unit_directional_vector

    # discretize the point to an integer-array and return
    border_point = np.array([int(current_vector[0]), int(current_vector[1])], dtype=int)
    return border_point

def HorizontalCameraFrustumProjectionToImageEdge(image: np.array,
                                                position: np.array, 
                                                heading: float, 
                                                horizontal_camera_FOV: float):
    # convert the horizontal camera fov to radians and divide it by 2 to give the cross-section
    horizontal_camera_FOV_cross_section = (horizontal_camera_FOV*np.pi/180)/2

    # Project two vectors starting in 'current position' and heading to the extremities of the frustum
    # to the image edges. Return these points. 
    frustum_bound_1 = ProjectVectorToImageBorder(image, position, AngularAddition(heading, -horizontal_camera_FOV_cross_section, degrees = False))
    frustum_bound_2 = ProjectVectorToImageBorder(image, position, AngularAddition(heading, horizontal_camera_FOV_cross_section, degrees = False))

    return frustum_bound_1, frustum_bound_2

def IdentifyImageCornersInCameraFrustum(image: np.array, 
                                        position: np.array, 
                                        heading: float, 
                                        horizontal_camera_FOV: float):
    # isolate the possible corners in any arbitrary image
    possible_corners = np.array([[0, 0],
                                 [0, image.shape[1]],
                                 [image.shape[0], 0],
                                 [image.shape[0], image.shape[1]]])
    
    # loop through the corners, checking if any fall into the camera frustum
    corners_in_frustum = []
    for corner in possible_corners:
        # find the angular difference between the camera-heading and the corner
        corner_vector = corner - position
        corner_angle = np.arctan2(corner_vector[1], corner_vector[0])

        # if the angular difference is smaller than the cross section of the horizontal FOV, consider
        # the corner to be within the frustum
        if (AngularDifference(heading, corner_angle, absolute_angle=True, degrees=True) <= horizontal_camera_FOV/2):
            corners_in_frustum.append(corner)

    return np.array(corners_in_frustum)
