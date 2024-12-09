import numpy as np

def point_in_polygon(point: np.ndarray, points: np.ndarray):


    point = point.reshape((2))
    points = points.reshape((-1, 2))

    # Check if test point lies on a vertex
    if point.tolist() in points.tolist():
        print("hello")
        return True

    inside = False

    # For each segment in the polygon
    for i in range(points.shape[0]):

        seg = points[(i-1,i),:]

        # Check if test point's y value lies between segment's y values (or equal to the segment's lowest y value)
        #
        #
        # So that this only counts as one intersection:
        #  
        #                  0---
        #                 /
        #                /  
        #   p ----------0------->         
        #                \
        #                 \
        #                  0---
        #

        if (point[1] < seg[0][1]) != (point[1] < seg[1][1]):

            # Find where segment intersects horizontal line through test point
            x = ((point[1]-seg[0][1])*(seg[1][0]-seg[0][0])/(seg[1][1]-seg[0][1])) + seg[0][0]

            # If test point is to the left of (or on) the intersection point, then we have an intersection
            if point[0] <= x:
                inside = not inside
            
    return inside



if __name__ == "__main__":

    # import time
    # import matplotlib.pyplot as plt


    # num_sides = []
    # times = []

    # for i in range(3, 3000, 100):
    #     point = np.random.uniform(0, 100, (1, 2))
    #     poly = np.random.uniform(0, 100, (i, 2))

    #     start_time = time.time()
    #     point_in_polygon(point, poly)
    #     end_time = time.time()
    #     num_sides.append(i)
    #     times.append(end_time - start_time)
    
    # plt.scatter(num_sides, times)
    # plt.show()

    point = np.array((-2, 0))
    poly = np.array(((-1, 0),(0, 1), (1, 1), (1, -1), (0, -1)))
    print(point_in_polygon(point, poly))
