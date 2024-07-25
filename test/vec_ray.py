import numpy as np
np.seterr(all="ignore")

def line_segment_intersections(segments1, segments2):
    """
    Calculate intersections between two batches of line segments.
    
    Parameters:
    segments1 : numpy array of shape (n, 4)
        Each row contains [x1, y1, x2, y2] defining a line segment in batch 1.
    segments2 : numpy array of shape (m, 4)
        Each row contains [x1, y1, x2, y2] defining a line segment in batch 2.
        
    Returns:
    intersections : numpy array
        Array containing intersection points (x, y) found between segments1 and segments2.
    """
    # Reshape segments to easily compute intersections
    segments1 = segments1.reshape(-1, 1, 4)
    segments2 = segments2.reshape(1, -1, 4)

    # Extract coordinates
    x1, y1, x2, y2 = segments1[..., 0], segments1[..., 1], segments1[..., 2], segments1[..., 3]
    x3, y3, x4, y4 = segments2[..., 0], segments2[..., 1], segments2[..., 2], segments2[..., 3]
    
    # Parametric equations of the lines
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    
    # Calculate intersections
    intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    
    # Mask invalid intersections (parallel lines or out-of-bounds)
    mask = (denom != 0) & \
           (intersect_x >= np.minimum(x1, x2)) & (intersect_x <= np.maximum(x1, x2)) & \
           (intersect_y >= np.minimum(y1, y2)) & (intersect_y <= np.maximum(y1, y2)) & \
           (intersect_x >= np.minimum(x3, x4)) & (intersect_x <= np.maximum(x3, x4)) & \
           (intersect_y >= np.minimum(y3, y4)) & (intersect_y <= np.maximum(y3, y4))

    intersect_x = np.where(mask, intersect_x, -1) #some large negative number
    intersect_y = np.where(mask, intersect_y, -1) #some large negative number

    intersections = np.stack((intersect_x.flatten(), intersect_y.flatten()), axis=-1).reshape(intersect_x.shape + (2,))

    print(mask)
    
    return intersections

# Example usage:
segments1 = np.array([[0, 0, -1, 1], [1, 0, 6, 5]])
segments2 = np.array([[2, 0, 0, 2], [5, 0, 0, 5]])

intersections = line_segment_intersections(segments1, segments2)
print()
print("Intersections:", intersections)
print()
print(np.linalg.norm(intersections, axis=-1))















# import numpy as np

# def line_segment_intersections(segments1, segments2):
#     """
#     Calculate intersections between two batches of line segments.
    
#     Parameters:
#     segments1 : numpy array of shape (n, 4)
#         Each row contains [x1, y1, x2, y2] defining a line segment in batch 1.
#     segments2 : numpy array of shape (m, 4)
#         Each row contains [x1, y1, x2, y2] defining a line segment in batch 2.
        
#     Returns:
#     intersections : numpy array
#         Array containing intersection points (x, y) found between segments1 and segments2,
#         with NaNs where there is no intersection.
#     """
#     # Reshape segments to easily compute intersections
#     segments1 = segments1.reshape(-1, 1, 4)
#     segments2 = segments2.reshape(1, -1, 4)
    
#     # Extract coordinates
#     x1, y1, x2, y2 = segments1[..., 0], segments1[..., 1], segments1[..., 2], segments1[..., 3]
#     x3, y3, x4, y4 = segments2[..., 0], segments2[..., 1], segments2[..., 2], segments2[..., 3]
    
#     # Parametric equations of the lines
#     denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    
#     # Calculate intersections
#     intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
#     intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    
#     # Mask invalid intersections (parallel lines or out-of-bounds)
#     mask = (denom != 0) & \
#            (intersect_x >= np.minimum(x1, x2)) & (intersect_x <= np.maximum(x1, x2)) & \
#            (intersect_y >= np.minimum(y1, y2)) & (intersect_y <= np.maximum(y1, y2)) & \
#            (intersect_x >= np.minimum(x3, x4)) & (intersect_x <= np.maximum(x3, x4)) & \
#            (intersect_y >= np.minimum(y3, y4)) & (intersect_y <= np.maximum(y3, y4))
    
#     # Create an array for intersections, initially filled with NaNs
#     intersections = np.full_like(intersect_x, np.nan, dtype=float)
    
#     # Assign valid intersection points
#     intersections[mask] = np.column_stack((intersect_x[mask], intersect_y[mask]))
    
#     return intersections